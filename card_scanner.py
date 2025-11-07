from typing import Tuple, List
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH
from common import Card

class CardScanner:
    def __init__(self):
        self.model = None

    async def load_model(self, model_path=MODEL_PATH):
        try:
            self.model = YOLO(model_path)
            print(f"Model loadsucceed from {model_path}")
        except Exception as e:
            print(f"Err loading model: {e}")
            raise

    async def _detect_cards(self, image: Image.Image) -> List[List[Tuple[int, int]]]:
        if self.model is None:
            raise RuntimeError("model not loaded. Call load_model() first.")
        
        image_np = np.array(image)
        results = self.model(image_np)
        
        card_corners = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    corners = [
                        (int(x1), int(y1)),
                        (int(x2), int(y1)), 
                        (int(x2), int(y2)),
                        (int(x1), int(y2))
                    ]
                    card_corners.append(corners)
        
        return card_corners

    def _crop_card(self, image: Image.Image, points: List[Tuple[int, int]]) -> Card:
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        rect = self._order_points(np.array(points))
        width, height = 400, 560
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(image_np, matrix, (width, height))
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        card_image = Image.fromarray(warped_rgb)
        
        return Card(card_image)

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _generate_phash(self, image: Image.Image) -> bytes:
        """Generate 8-byte PHash using OpenCV"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        dct = cv2.dct(np.float32(resized))
        dct_roi = dct[:8, :8]
        
        median = np.median(dct_roi)
        hash_bits = (dct_roi > median).flatten()
        
        hash_bytes = bytearray()
        for i in range(0, len(hash_bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(hash_bits) and hash_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            hash_bytes.append(byte_val)
        
        return bytes(hash_bytes)

    def _generate_block_mean_hash(self, image: Image.Image) -> bytes:
        """Generate 32-byte Block Mean Hash using OpenCV"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        resized = cv2.resize(gray, (16, 16))  # 256 blocks total
        blocks = []
        
        for i in range(16):
            for j in range(16):
                block_mean = resized[i, j]
                blocks.append(block_mean)
        
        hash_bytes = bytearray()
        for i in range(0, len(blocks), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(blocks):
                    bit_val = 1 if blocks[i + j] > np.mean(blocks) else 0
                    byte_val |= (bit_val << (7 - j))
            hash_bytes.append(byte_val)
        
        return bytes(hash_bytes)

    def _generate_warr_hildreth_hash(self, image: Image.Image) -> bytes:
        """Generate 72-byte Warr-Hildreth style hash using OpenCV"""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        resized = cv2.resize(laplacian, (24, 24))  # 576 pixels = 72 bytes
        
        hash_bytes = bytearray()
        flat_laplacian = resized.flatten()
        
        for i in range(0, len(flat_laplacian), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(flat_laplacian):
                    bit_val = 1 if flat_laplacian[i + j] > 0 else 0
                    byte_val |= (bit_val << (7 - j))
            hash_bytes.append(byte_val)
        
        return bytes(hash_bytes)

    def _generate_combined_hash(self, image: Image.Image) -> int:
        """Generate combined 112-byte (896-bit) hash from all three algorithms"""
        phash = self._generate_phash(image)           # 8 bytes
        block_hash = self._generate_block_mean_hash(image)  # 32 bytes
        warr_hash = self._generate_warr_hildreth_hash(image)  # 72 bytes
        
        combined_bytes = phash + block_hash + warr_hash  # 112 bytes total
        
        combined_int = 0
        for byte in combined_bytes:
            combined_int = (combined_int << 8) | byte
        
        return combined_int

    def _to_bigints(self, input_hash: int) -> List[int]:
        """Split 896-bit hash into 14 × 64-bit integers"""
        bigints = []
        mask_64bit = (1 << 64) - 1 
        
        for i in range(14):
            chunk = (input_hash >> (64 * (13 - i))) & mask_64bit
            bigints.append(chunk)
        
        return bigints

    async def scan_cards(self, image: Image.Image) -> List[Card]:
        ret = []
        points = await self._detect_cards(image)
        
        for corner_points in points:
            card = self._crop_card(image, corner_points)
            card.hash = self._generate_combined_hash(card.image)
            card.hash_bigints = self._to_bigints(card.hash)
            ret.append(card)
            
        return ret