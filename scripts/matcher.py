import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from sklearn.neighbors import NearestNeighbors
import pickle
from itertools import product

# Configuration
CSV_DIR = "sets"
IMAGE_DIR = "images"
TARGET_IMG = "dusknoir1.jpg"
CACHE_FILE = "feature_cache.pkl"
SIMILARITY_THRESHOLD = 0.8
FEATURE_WEIGHTS = {'phash': 0.6, 'color': 0.4}
ROTATION_ANGLES = [0, 90, 180, 270]  # Test these rotation angles
FLIP_OPTIONS = [True]  # Test both flipped and unflipped versions

class CardMatcher:
    def __init__(self):
        self.feature_cache = self.load_cache()
        self.card_db = self.load_database()
        self.matcher = NearestNeighbors(n_neighbors=1, metric='cosine')
        
    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def save_cache(self):
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(self.feature_cache, f)

    def load_database(self):
        cards = []
        for csv_file in os.listdir(CSV_DIR):
            if csv_file.endswith('.csv'):
                try:
                    df = pd.read_csv(f"{CSV_DIR}/{csv_file}", usecols=['productId', 'name'])
                    cards.append(df[df['productId'].notna() & df['name'].notna()])
                except Exception as e:
                    print(f"Skipping {csv_file}: {str(e)}")
        return pd.concat(cards, ignore_index=True) if cards else pd.DataFrame()

    def apply_transformations(self, img):
        """Generate all possible transformed versions of the image"""
        transformed = []
        
        # Perspective correction first
        card_img = self.detect_card(img.copy())
        if card_img is None:
            card_img = img
            
        for angle, flip in product(ROTATION_ANGLES, FLIP_OPTIONS):
            transformed_img = card_img.copy()
            
            # Rotate
            if angle != 0:
                h, w = transformed_img.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                transformed_img = cv2.warpAffine(transformed_img, M, (w, h))
            
            # Flip
            if flip:
                transformed_img = cv2.flip(transformed_img, 1)
                
            transformed.append(transformed_img)
            
        return transformed

    def detect_card(self, img):
        """Find and extract card from image using contour detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = approx.sum(axis=2)
            rect[0] = approx[np.argmin(s)]
            rect[2] = approx[np.argmax(s)]
            
            diff = np.diff(approx, axis=2)
            rect[1] = approx[np.argmin(diff)]
            rect[3] = approx[np.argmax(diff)]
            
            # Calculate dimensions
            width = max(
                np.linalg.norm(rect[1] - rect[0]),
                np.linalg.norm(rect[3] - rect[2])
            )
            height = max(
                np.linalg.norm(rect[3] - rect[0]),
                np.linalg.norm(rect[2] - rect[1])
            )
            
            # Destination points for transform
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]], dtype="float32")
            
            # Apply perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (int(width), int(height)))
            return warped
        return None

    def preprocess_image(self, img):
        """Apply color processing to a single image"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.merge((l, a, b))

    def extract_features(self, img):
        """Extract features from a single image"""
        # Convert LAB to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # PHash feature
        phash = imagehash.phash(pil_img, hash_size=16)
        phash_binary = np.unpackbits(np.array(phash.hash, dtype=np.uint8))

        # Color histogram in LAB space
        hist_l = cv2.calcHist([img], [0], None, [32], [0,256])
        hist_a = cv2.calcHist([img], [1], None, [32], [0,256])
        hist_b = cv2.calcHist([img], [2], None, [32], [0,256])
        color_hist = np.concatenate([hist_l, hist_a, hist_b]).flatten()
        color_hist /= (color_hist.sum() + 1e-7)

        return {
            'phash': phash_binary,
            'color': color_hist
        }

    def build_database(self):
        features = []
        valid_ids = []
        
        for _, row in self.card_db.iterrows():
            img_path = f"{IMAGE_DIR}/{row['productId']}.jpg"
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    feat = self.extract_features(self.preprocess_image(img))
                    if feat:
                        combined = np.concatenate([
                            feat['phash'] * FEATURE_WEIGHTS['phash'],
                            feat['color'] * FEATURE_WEIGHTS['color']
                        ])
                        features.append(combined)
                        valid_ids.append(row['productId'])
        
        if features:
            self.matcher.fit(np.array(features))
        return valid_ids

    def find_best_match(self, query_img_path):
        """Find the best match across all possible orientations"""
        img = cv2.imread(query_img_path)
        if img is None:
            return None, 0.0
            
        valid_ids = self.build_database()
        if not valid_ids:
            return None, 0.0
            
        best_match = None
        best_score = 0
        
        # Test all transformed versions
        for transformed_img in self.apply_transformations(img):
            try:
                # Preprocess and extract features
                processed = self.preprocess_image(transformed_img)
                features = self.extract_features(processed)
                
                # Prepare query vector
                query_vec = np.concatenate([
                    features['phash'] * FEATURE_WEIGHTS['phash'],
                    features['color'] * FEATURE_WEIGHTS['color']
                ]).reshape(1, -1)
                
                # Find nearest neighbor
                distances, indices = self.matcher.kneighbors(query_vec)
                current_score = 1 - distances[0][0]
                
                # Update best match if better
                if current_score > best_score:
                    best_score = current_score
                    best_id = valid_ids[indices[0][0]]
                    best_match = self.card_db[self.card_db['productId'] == best_id]['name'].values[0]
                    
                    # Early exit if perfect match
                    if best_score >= 0.95:
                        break
                        
            except Exception as e:
                print(f"Error processing transformed image: {str(e)}")
                continue
                
        return best_match, best_score

def main():
    print("=== Advanced Card Matcher ===")
    matcher = CardMatcher()
    
    if not os.path.exists(TARGET_IMG):
        print(f"Error: Target image {TARGET_IMG} not found")
        return
        
    print("Matching card across multiple orientations...")
    best_match, confidence = matcher.find_best_match(TARGET_IMG)
    
    print("\n=== RESULTS ===")
    if best_match and confidence >= SIMILARITY_THRESHOLD:
        print(f"Best Match: {best_match} ({confidence*100:.1f}% confidence)")
    else:
        print("No confident match found")
        if best_match:
            print(f"Closest Candidate: {best_match} ({confidence*100:.1f}%)")
    
    matcher.save_cache()

if __name__ == "__main__":
    main()
