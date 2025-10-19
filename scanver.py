from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import time
import mysql.connector
from mysql.connector import Error
import io
import traceback

#config p1
MODEL_PATH = "best.pt"
INPUT_IMAGE = "hy.jpg"
OUTPUT_DIR = "detected_cards"
TARGET_SIZE = (600, 840)

#database config p2
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'csv_import_db'
}

class CardIdentifier:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.model = YOLO(MODEL_PATH)
        self.conn = self._create_db_connection()
        self.card_names = self._load_card_names()
        self.model.fuse()
        self.model.conf = 0.7

    def _create_db_connection(self):
        """Create a database connection"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            if conn.is_connected():
                print(" Successfully connected to MySQL database")
                return conn
        except Error as e:
            print(f" Database connection failed: {e}")
            return None

    def _load_card_names(self):
        """Load card names from MySQL database"""
        card_data = {}
        if not self.conn:
            print("No database connection available")
            return card_data
            
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT productId, name FROM `cards`")
            rows = cursor.fetchall()
            
            for row in rows:
                card_id = str(row[0]).strip()
                card_data[card_id] = row[1].strip()
                
            cursor.close()
            print(f" Loaded {len(card_data)} card names from database")
            
        except Error as e:
            print(f"Error loading card names from database: {e}")
            
        return card_data

    def _compute_image_hashes_simple(self, image):
        """Compute simplified hashes for an image"""
        try:
            #image resize compo
            img_resized = image.resize((300, 420))
            
            #phash, more reliability
            phash = imagehash.phash(img_resized)
            
            return str(phash)
            
        except Exception as e:
            print(f" Error computing image hash: {str(e)}")
            return None

    def _identify_card_simple(self, cropped_img):
        """Simplified card matching using only phash"""
        try:
            #PIL image converter
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            #process the hash for target card
            query_hash = self._compute_image_hashes_simple(pil_img)
            
            if query_hash is None:
                return "Error computing hash", float('inf')
            
            #match the closest phash
            cursor = self.conn.cursor()
            cursor.execute("SELECT productId, phash FROM `card_hashes`")
            hash_records = cursor.fetchall()
            
            best_match = None
            min_distance = float('inf')
            
            query_hash_obj = imagehash.hex_to_hash(query_hash)
            
            for record in hash_records:
                try:
                    product_id, db_phash_str = record
                    
                    if not db_phash_str:
                        continue
                    
                    #string hash to image hash
                    db_phash = imagehash.hex_to_hash(str(db_phash_str))
                    
                    #distance calculator
                    distance = query_hash_obj - db_phash
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = product_id
                        
                except Exception as e:
                    #skip problem records
                    continue
            
            cursor.close()
            
            #conf threshold
            if min_distance < 15 and best_match:
                card_name = self.card_names.get(best_match, f"Unknown ({best_match})")
                return card_name, min_distance
            else:
                return "No confident match found", min_distance
                    
        except Exception as e:
            print(f" Error identifying card: {str(e)}")
            traceback.print_exc()
            return "Error", float('inf')

    def _identify_card_fallback(self, cropped_img):
        """Fallback identification method using direct image comparison"""
        try:
            #pil conversion and rehash
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            query_img = pil_img.resize((300, 420))
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT c.productId, c.name, c.image_data FROM cards c")
            records = cursor.fetchall()
            
            best_match = None
            min_diff = float('inf')
            
            for record in records:
                try:
                    product_id, name, image_data = record
                    
                    if not image_data:
                        continue
                    
                    #blob conversion to image
                    db_image = Image.open(io.BytesIO(image_data))
                    db_image_resized = db_image.resize((300, 420))
                    
                    #using numpy for hash comparison
                    query_array = np.array(query_img)
                    db_array = np.array(db_image_resized)
                    
                    #mean squared error
                    if query_array.shape == db_array.shape:
                        diff = np.mean((query_array - db_array) ** 2)
                        
                        if diff < min_diff:
                            min_diff = diff
                            best_match = (product_id, name)
                            
                except Exception as e:
                    continue
            
            cursor.close()
            
            #adj. threshold based on tests
            if min_diff < 5000 and best_match:  #threshold can be tweaked for different purposes
                return f"{best_match[1]} ({best_match[0]})", min_diff
            else:
                return "No match found", min_diff
                
        except Exception as e:
            print(f" Error in fallback identification: {e}")
            return "Error", float('inf')

    def _get_card_image_from_database(self, product_id):
        """Retrieve card image from database for display"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT image_data FROM cards WHERE productId = %s", (product_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0]:
                image_data = result[0]
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
            else:
                return None
                
        except Exception as e:
            print(f" Error retrieving image from database: {e}")
            return None

    def _order_points(self, pts):
        """Order points for perspective transform"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _crop_and_deskew(self, frame, box, mask):
        """Crop and deskew card using mask"""
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            x1, y1, x2, y2 = map(int, box)
            
            #makes sure coordinates are in frame boundary
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            mask_roi = mask_uint8[y1:y2, x1:x2]
            
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < 100:
                return None
                
            rect = cv2.minAreaRect(largest_contour)
            box_points = cv2.boxPoints(rect)
            box_points = box_points.astype(np.int32)
            
            ordered_points = self._order_points(box_points)
            ordered_points[:, 0] += x1
            ordered_points[:, 1] += y1
            
            #perspective transform if need be
            src_pts = ordered_points.astype('float32')
            dst_pts = np.array([
                [0, 0],
                [TARGET_SIZE[0], 0],
                [TARGET_SIZE[0], TARGET_SIZE[1]],
                [0, TARGET_SIZE[1]]], dtype='float32')
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            aligned = cv2.warpPerspective(frame, matrix, TARGET_SIZE)
            return aligned
            
        except Exception as e:
            print(f" Error in deskewing: {e}")
            return None

    def _process_detection(self, frame):
        """Process frame and detect cards"""
        results = self.model(frame)[0]
        annotated = results.plot()
        
        detected_cards = []
        
        if results.boxes and results.masks:
            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                box_coords = box.xyxy[0].cpu().numpy()
                mask_data = mask.data.cpu().numpy().squeeze()
                
                aligned = self._crop_and_deskew(frame, box_coords, mask_data)
                
                if aligned is not None and aligned.size > 0:
                    timestamp = int(time.time())
                    output_path = os.path.join(OUTPUT_DIR, f"card_{timestamp}_{i}.jpg")
                    cv2.imwrite(output_path, aligned)
                    
                    #test with simple hash method first
                    card_name, confidence = self._identify_card_simple(aligned)
                    
                    #fallback if no confident matchs
                    if "No confident match" in card_name or confidence > 20:
                        print("🔄 Trying fallback identification method...")
                        card_name, confidence = self._identify_card_fallback(aligned)
                    
                    #id extraction
                    product_id = None
                    if "(" in card_name and ")" in card_name and "No match" not in card_name:
                        product_id = card_name.split("(")[-1].split(")")[0]
                    
                    detected_cards.append({
                        'card_name': card_name,
                        'confidence': confidence,
                        'image_path': output_path,
                        'aligned_image': aligned,
                        'product_id': product_id
                    })
                    
                    #text to annotated image
                    label = f"{card_name} ({confidence:.1f})"
                    x1, y1, x2, y2 = map(int, box_coords)
                    cv2.putText(annotated, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated, detected_cards

    def process_image(self, image_path):
        """Process a single image file"""
        if not os.path.exists(image_path):
            print(f" Error: Image file '{image_path}' not found!")
            return
        
        print(f" Processing image: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f" Error: Could not read image '{image_path}'")
            return
        
        annotated_frame, detected_cards = self._process_detection(frame)
        
        print(f"\n RESULTS")
        print(f" Found {len(detected_cards)} card(s) in the image:")
        
        for i, card in enumerate(detected_cards):
            if isinstance(card['confidence'], (int, float)):
                if card['confidence'] < 15:
                    confidence_level = "HIGH"
                elif card['confidence'] < 25:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
            else:
                confidence_level = "UNKNOWN"
                
            print(f"   Card {i+1}: {card['card_name']}")
            print(f"      Confidence: {card['confidence']} ({confidence_level})")
            print(f"      Saved to: {card['image_path']}")
            
            #diplay algined card
            cv2.imshow(f"Detected Card {i+1}", card['aligned_image'])
            
            #will also show referenced card if possible
            if card['product_id'] and "No match" not in card['card_name']:
                db_image = self._get_card_image_from_database(card['product_id'])
                if db_image is not None:
                    db_image_resized = cv2.resize(db_image, (card['aligned_image'].shape[1], card['aligned_image'].shape[0]))
                    cv2.imshow(f"Reference {i+1}", db_image_resized)
        
        cv2.imshow("Detected Cards - Press any key to close", annotated_frame)
        print("\n Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detected_cards

    def __del__(self):
        if hasattr(self, 'conn') and self.conn and self.conn.is_connected():
            self.conn.close()
            print(" Database connection closed")

def test_database_hashes():
    """Test the hash data in the database"""
    print(" Testing database hash data...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        #recorded hash sample check
        cursor.execute("SELECT productId, phash, dhash, colorhash FROM `card_hashes` LIMIT 5")
        sample_hashes = cursor.fetchall()
        
        print(" Sample hash data:")
        for record in sample_hashes:
            product_id, phash, dhash, colorhash = record
            print(f"   Product ID: {product_id}")
            print(f"   phash type: {type(phash)}, length: {len(str(phash)) if phash else 'None'}")
            print(f"   dhash type: {type(dhash)}, length: {len(str(dhash)) if dhash else 'None'}")
            print(f"   colorhash type: {type(colorhash)}, length: {len(str(colorhash)) if colorhash else 'None'}")
            print("   ---")
        
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        print(f" Error testing hash data: {e}")
        return False

if __name__ == "__main__":
    print(" Pokemon Card Identifier (Fixed Version)")
    print("=" * 50)
    
    #database connection test and hashes data
    if test_database_hashes():
        print("\n Starting card identification...")
        identifier = CardIdentifier()
        
        detected_cards = identifier.process_image(INPUT_IMAGE)
        
        if detected_cards:
            print(f"\n Successfully processed {len(detected_cards)} card(s)!")
        else:
            print("\n No cards detected or an error occurred.")
    else:
        print("\n Cannot start application due to database issues.")