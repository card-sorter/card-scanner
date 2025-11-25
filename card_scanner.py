from typing import Tuple, List
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH
from common import Card
from rgb_hasher import RGBHasher  # Import the shared hasher

class CardScanner:
    def __init__(self):
        self.model = None
        self.hasher = RGBHasher()  # Use the same hasher as database

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
    def _phash(self, image: Image.Image) -> int:
        try:

            image_gray = image.convert('L').resize((8, 8), Image.LANCZOS)
            # Calculate average hash
            hash_value = imagehash.average_hash(image_gray)
            return int(str(hash_value), 16)
        except Exception as e:
            print(f"Error generating phash: {e}")
            return 0

    def _to_bigints(self, input_hash: int) -> List[int]:
        bigints = []
        mask_64bit = (1 << 64) - 1 
        
        if input_hash == 0:
            return [0]
        
        temp_hash = input_hash
        while temp_hash > 0:
            chunk = temp_hash & mask_64bit
            bigints.append(chunk)
            temp_hash >>= 64
        
        return bigints

    def _phash_rgb(self, image: Image.Image) -> int:
        image_resized = image.resize((16, 16), Image.LANCZOS)
        
        img_array = np.array(image_resized)

        r_hash = imagehash.average_hash(Image.fromarray(img_array[:,:,0]), hash_size=8)
        g_hash = imagehash.average_hash(Image.fromarray(img_array[:,:,1]), hash_size=8)  
        b_hash = imagehash.average_hash(Image.fromarray(img_array[:,:,2]), hash_size=8)

        r_int = int(str(r_hash), 16)
        g_int = int(str(g_hash), 16)
        b_int = int(str(b_hash), 16)
        
        combined_hash = (r_int << 128) | (g_int << 64) | b_int
        
        return combined_hash
    
    async def scan_cards(self, image: Image.Image) -> List[Card]:
        ret = []
        points = await self._detect_cards(image)
        
        for corner_points in points:
            card = self._crop_card(image, corner_points)
            card.hash = self.hasher.generate_phash_rgb(card.image)
            card.hash_bigints = self.hasher.to_bigints(card.hash)  # Use as many columns as needed
            ret.append(card)
            
        return ret