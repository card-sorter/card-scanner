import os
import cv2
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import pickle
from skimage.feature import local_binary_pattern
from sklearn.neighbors import NearestNeighbors

CSV_DIR = "sets"
IMAGE_DIR = "images"
TARGET_IMG = "king.jpg"
HASH_CACHE = "hash_cache.pkl"
SIMILARITY_THRESHOLD = 10
MAX_WORKERS = 6
FEATURE_WEIGHTS = {
    'phash': 0.4,
    'lbp': 0.3,
    'color': 0.3
}

class AdvancedCardMatcher:
    def __init__(self):
        self.hash_cache = self.load_hash_cache()
        self.card_db = self.load_card_database()
        self.nn_model = None
        
    def load_hash_cache(self):
        if os.path.exists(HASH_CACHE):
            try:
                with open(HASH_CACHE, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def save_hash_cache(self):
        with open(HASH_CACHE, 'wb') as f:
            pickle.dump(self.hash_cache, f)

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(normalized, -1, kernel)
        
        return sharpened

    def extract_features(self, img_path):
        cache_key = f"{img_path}-{os.path.getmtime(img_path)}"
        if cache_key in self.hash_cache:
            return self.hash_cache[cache_key]
            
        img = self.preprocess_image(img_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray)
        phash_obj = imagehash.phash(pil_img, hash_size=16)
        phash_array = np.array(phash_obj.hash).flatten()
        
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype('float32')
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0,256])
        color_hist = np.concatenate([hist_h, hist_s]).flatten().astype('float32')
        color_hist /= (color_hist.sum() + 1e-7)
        
        features = {
            'phash': phash_array,
            'lbp': lbp_hist,
            'color': color_hist,
            'path': img_path
        }
        
        self.hash_cache[cache_key] = features
        return features

    def build_feature_database(self):
        features = []
        valid_ids = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for _, row in self.card_db.iterrows():
                img_path = f"{IMAGE_DIR}/{row['productId']}.jpg"
                if os.path.exists(img_path):
                    futures.append(executor.submit(self.extract_features, img_path))
                    valid_ids.append(row['productId'])
            
            for future in futures:
                features.append(future.result())
        
        valid_features = [f for f in features if f is not None]
        
        phash_vecs = [f['phash'] for f in valid_features]
        lbp_vecs = [f['lbp'] for f in valid_features]
        color_vecs = [f['color'] for f in valid_features]
        
        combined_features = []
        for ph, lbp, col in zip(phash_vecs, lbp_vecs, color_vecs):
            combined = np.concatenate([
                ph * FEATURE_WEIGHTS['phash'],
                lbp * FEATURE_WEIGHTS['lbp'],
                col * FEATURE_WEIGHTS['color']
            ])
            combined_features.append(combined)
            
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn_model.fit(combined_features)
        return valid_ids, combined_features

    def load_card_database(self):
        card_data = []
        for csv_file in os.listdir(CSV_DIR):
            if csv_file.endswith('.csv'):
                try:
                    df = pd.read_csv(
                        f"{CSV_DIR}/{csv_file}",
                        usecols=['productId', 'name', 'imageUrl'],
                        dtype={'productId': str, 'name': str}
                    )
                    df = df[df['productId'].notna() & df['name'].notna()]
                    card_data.append(df)
                except Exception as e:
                    print(f"Skipping {csv_file}: {str(e)}")
                    continue
        return pd.concat(card_data, ignore_index=True)

    def find_match(self, query_img_path):
        query_features = self.extract_features(query_img_path)
        if query_features is None:
            return None, float('inf')
            
        valid_ids, db_features = self.build_feature_database()
        
        combined_query = np.concatenate([
            query_features['phash'] * FEATURE_WEIGHTS['phash'],
            query_features['lbp'] * FEATURE_WEIGHTS['lbp'],
            query_features['color'] * FEATURE_WEIGHTS['color']
        ]).reshape(1, -1)
        
        distances, indices = self.nn_model.kneighbors(combined_query)
        
        best_idx = indices[0][0]
        best_distance = distances[0][0]
        best_id = valid_ids[best_idx]
        best_name = self.card_db[self.card_db['productId'] == best_id]['name'].values[0]
        
        return best_name, best_distance

def main():
    print("=== Advanced Card Matcher ===")
    matcher = AdvancedCardMatcher()
    
    if not os.path.exists(TARGET_IMG):
        print(f"Error: Target image {TARGET_IMG} not found")
        return
        
    print("Analyzing card with multi-feature matching...")
    best_match, score = matcher.find_match(TARGET_IMG)
    
    print("\n=== RESULTS ===")
    if best_match and score < SIMILARITY_THRESHOLD:
        confidence = max(0, 100 - (score * 10))
        print(f"Match Found: {best_match}")
        print(f"Confidence: {confidence:.1f}%")
    else:
        print("No confident match found")
        if best_match:
            print(f"Closest Candidate: {best_match} (Score: {score:.2f})")
        
    matcher.save_hash_cache()

if __name__ == "__main__":
    main()