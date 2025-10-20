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

MODEL_PATH = "best.pt"
INPUT_IMAGE = "42355.jpg"
OUTPUT_DIR = "detected_cards"
TARGET_SIZE = (600, 840)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'big_int_hashes'
}

class CardIdentifier:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.model = YOLO(MODEL_PATH)
        self.conn = self._create_db_connection()
        self.hash_table_info = self._detect_hash_columns()
        self.card_names = self._load_card_names()
        self.model.fuse()
        self.model.conf = 0.7

    def _create_db_connection(self):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            if conn.is_connected():
                print("Successfully connected to MySQL database")
                return conn
        except Error as e:
            print(f"Database connection failed: {e}")
            return None

    def _detect_hash_columns(self):
        if not self.conn:
            return None
            
        try:
            cursor = self.conn.cursor()
            cursor.execute("DESCRIBE image_hashes")
            columns = cursor.fetchall()
            
            hash_columns = {}
            for column in columns:
                col_name = column[0]
                col_type = column[1]
                hash_columns[col_name] = col_type
            
            hash_column = None
            id_column = None
            
            possible_hash_columns = ['phash_bigint', 'hash', 'image_hash', 'phash', 'hash_value', 'bigint_hash']
            for col in possible_hash_columns:
                if col in hash_columns:
                    hash_column = col
                    break
            
            if not hash_column:
                for col_name, col_type in hash_columns.items():
                    if 'bigint' in str(col_type).lower() and 'id' not in col_name.lower():
                        hash_column = col_name
                        break
            
            possible_id_columns = ['cardID', 'id', 'productId', 'card_id', 'product_id']
            for col in possible_id_columns:
                if col in hash_columns:
                    id_column = col
                    break
            
            if not id_column or not hash_column:
                column_names = list(hash_columns.keys())
                if len(column_names) >= 2:
                    id_column = column_names[0]
                    hash_column = column_names[1]
                    print(f"Using first column '{id_column}' as ID and second column '{hash_column}' as hash")
                else:
                    print("Cannot determine column structure")
                    return None
            
            print(f"Identified columns - ID: '{id_column}', Hash: '{hash_column}'")
            
            cursor.execute(f"SELECT {id_column}, {hash_column} FROM image_hashes LIMIT 3")
            samples = cursor.fetchall()
            for sample in samples:
                print(f"{id_column}: {sample[0]}, {hash_column}: {sample[1]}")
            
            cursor.close()
            
            return {
                'id_column': id_column,
                'hash_column': hash_column
            }
            
        except Error as e:
            print(f"Error detecting hash columns: {e}")
            return None

    def _load_card_names(self):
        card_data = {}
        if not self.conn:
            print("No database connection available")
            return card_data
            
        try:
            cursor = self.conn.cursor()
            cursor.execute("DESCRIBE cards")
            cards_columns = [col[0] for col in cursor.fetchall()]
            
            id_column = 'productId'
            name_column = 'name'
            
            if 'productId' not in cards_columns:
                possible_id_columns = ['id', 'product_id', 'cardID', 'card_id']
                for col in possible_id_columns:
                    if col in cards_columns:
                        id_column = col
                        break
            
            if 'name' not in cards_columns:
                possible_name_columns = ['cardName', 'product_name', 'title']
                for col in possible_name_columns:
                    if col in cards_columns:
                        name_column = col
                        break
            
            print(f"Using columns - ID: '{id_column}', Name: '{name_column}'")
            
            cursor.execute(f"SELECT {id_column}, {name_column} FROM `cards`")
            rows = cursor.fetchall()
            
            for row in rows:
                product_id = str(row[0]).strip()
                card_data[product_id] = row[1].strip()
                
            cursor.close()
            print(f"Loaded {len(card_data)} card names from 'cards' table")
            
        except Error as e:
            print(f"Error loading card names from database: {e}")
            
        return card_data

    def _compute_image_hash_bigint(self, image):
        try:
            img_resized = image.resize((32, 32))
            phash = imagehash.phash(img_resized, hash_size=8, highfreq_factor=4)
            hash_bigint = int(str(phash), 16)
            return hash_bigint
            
        except Exception as e:
            print(f"Error computing image hash: {str(e)}")
            return None

    def _identify_card_by_bigint_hash(self, cropped_img):
        if not self.hash_table_info:
            return "Hash table info not available", float('inf')
            
        try:
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            query_hash_bigint = self._compute_image_hash_bigint(pil_img)
            
            if query_hash_bigint is None:
                return "Error computing hash", float('inf')
            
            print(f"Query hash (bigint): {query_hash_bigint}")
            print(f"Query hash (hex): {hex(query_hash_bigint)}")
            
            cursor = self.conn.cursor()
            id_column = self.hash_table_info['id_column']
            hash_column = self.hash_table_info['hash_column']
            
            query = f"SELECT {id_column}, {hash_column} FROM `image_hashes` WHERE {hash_column} IS NOT NULL"
            print(f"Executing query: {query}")
            
            cursor.execute(query)
            hash_records = cursor.fetchall()
            
            print(f"Found {len(hash_records)} hash records in 'image_hashes' table")
            
            if not hash_records:
                print("No hash records found in image_hashes table")
                return "No hash data in database", float('inf')
            
            best_match = None
            min_distance = float('inf')
            distances = []
            
            for record in hash_records:
                try:
                    card_id, db_hash_bigint = record
                    
                    if db_hash_bigint is None:
                        continue
                    
                    xor_result = query_hash_bigint ^ db_hash_bigint
                    distance = bin(xor_result).count('1')
                    
                    distances.append((card_id, distance, db_hash_bigint))
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = card_id
                        
                except Exception as e:
                    print(f"Error processing record {card_id}: {e}")
                    continue
            
            cursor.close()
            
            distances.sort(key=lambda x: x[1])
            print("Top 5 matches:")
            for card_id, dist, db_hash in distances[:5]:
                card_name = self.card_names.get(str(card_id), f"Card_{card_id}")
                print(f"{card_name} (ID: {card_id}): distance {dist}")
            
            if min_distance <= 8:
                card_name = self.card_names.get(str(best_match), f"Card_{best_match}")
                return f"{card_name}", min_distance
            elif min_distance <= 15:
                card_name = self.card_names.get(str(best_match), f"Card_{best_match}")
                return f"{card_name} (Low Confidence)", min_distance
            else:
                return "No confident match found", min_distance
                
        except Exception as e:
            print(f"Error identifying card with bigint hash: {str(e)}")
            traceback.print_exc()
            return "Error", float('inf')

    def _analyze_database_structure(self):
        try:
            cursor = self.conn.cursor()
            
            print("Analyzing database structure...")
            
            print("'cards' table structure:")
            cursor.execute("DESCRIBE cards")
            cards_columns = cursor.fetchall()
            for column in cards_columns:
                print(f"{column[0]} - {column[1]}")
            
            print("'image_hashes' table structure:")
            cursor.execute("DESCRIBE image_hashes")
            image_hashes_columns = cursor.fetchall()
            for column in image_hashes_columns:
                print(f"{column[0]} - {column[1]}")
            
            print("Sample from 'cards' table:")
            cursor.execute("SELECT * FROM cards LIMIT 3")
            cards_samples = cursor.fetchall()
            for sample in cards_samples:
                print(f"{sample}")
            
            print("Sample from 'image_hashes' table:")
            cursor.execute("SELECT * FROM image_hashes LIMIT 3")
            hash_samples = cursor.fetchall()
            for sample in hash_samples:
                print(f"{sample}")
            
            cursor.close()
            return True
            
        except Error as e:
            print(f"Error analyzing database structure: {e}")
            return False

    def _get_card_image_from_database(self, product_id):
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
                
        except Error as e:
            print(f"Error retrieving image from database: {e}")
            return None

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _crop_and_deskew(self, frame, box, mask):
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            x1, y1, x2, y2 = map(int, box)
            
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
            print(f"Error in deskewing: {e}")
            return None

    def _process_detection(self, frame):
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
                    
                    card_name, confidence = self._identify_card_by_bigint_hash(aligned)
                    
                    product_id = None
                    if "(" in card_name and ")" in card_name and "No match" not in card_name:
                        try:
                            product_id = card_name.split("(")[-1].split(")")[0]
                        except:
                            pass
                    
                    detected_cards.append({
                        'card_name': card_name,
                        'confidence': confidence,
                        'image_path': output_path,
                        'aligned_image': aligned,
                        'product_id': product_id
                    })
                    
                    label = f"{card_name} ({confidence:.1f})"
                    x1, y1, x2, y2 = map(int, box_coords)
                    cv2.putText(annotated, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated, detected_cards

    def process_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found!")
            return
        
        print(f"Processing image: {image_path}")
        
        self._analyze_database_structure()
        
        if not self.hash_table_info:
            print("Could not detect hash table columns. Please check your database structure.")
            return
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image '{image_path}'")
            return
        
        annotated_frame, detected_cards = self._process_detection(frame)
        
        print(f"RESULTS")
        print(f"Found {len(detected_cards)} card(s) in the image:")
        
        for i, card in enumerate(detected_cards):
            if isinstance(card['confidence'], (int, float)):
                if card['confidence'] <= 8:
                    confidence_level = "HIGH"
                elif card['confidence'] <= 15:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
            else:
                confidence_level = "UNKNOWN"
                
            print(f"Card {i+1}: {card['card_name']}")
            print(f"Confidence: {card['confidence']} {confidence_level}")
            print(f"Saved to: {card['image_path']}")
            
            cv2.imshow(f"Detected Card {i+1}", card['aligned_image'])
            
            if card['product_id'] and "No match" not in card['card_name']:
                db_image = self._get_card_image_from_database(card['product_id'])
                if db_image is not None:
                    db_image_resized = cv2.resize(db_image, (card['aligned_image'].shape[1], card['aligned_image'].shape[0]))
                    cv2.imshow(f"Reference {i+1}", db_image_resized)
        
        cv2.imshow("Detected Cards - Press any key to close", annotated_frame)
        print("Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detected_cards

    def __del__(self):
        if hasattr(self, 'conn') and self.conn and self.conn.is_connected():
            self.conn.close()
            print("Database connection closed")

def test_hash_matching():
    print("Testing hash matching...")
    
    identifier = CardIdentifier()
    if not identifier.conn:
        return
    
    try:
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 128
        pil_image = Image.fromarray(test_image)
        
        test_hash = identifier._compute_image_hash_bigint(pil_image)
        print(f"Test hash computed: {test_hash} (hex: {hex(test_hash)})")
        
        cursor = identifier.conn.cursor()
        cursor.execute("SELECT cardID, hash_value FROM image_hashes LIMIT 5")
        db_hashes = cursor.fetchall()
        
        print("Sample database hashes:")
        for card_id, db_hash in db_hashes:
            if db_hash:
                distance = bin(test_hash ^ db_hash).count('1')
                card_name = identifier.card_names.get(str(card_id), f"Card_{card_id}")
                print(f"{card_name} (ID: {card_id}):")
                print(f"DB Hash: {db_hash}")
                print(f"Distance: {distance}")
        
        cursor.close()
        
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Pokemon Card Identifier")
    print("=" * 50)
    print("Database: big_int_hashes")
    print("Tables: cards (product info), image_hashes (bigint hashes)")
    print("=" * 50)
    
    identifier = CardIdentifier()
    
    if identifier.conn and identifier.hash_table_info:
        detected_cards = identifier.process_image(INPUT_IMAGE)
        
        if detected_cards:
            print(f"Successfully processed {len(detected_cards)} card(s)!")
        else:
            print("No cards detected or an error occurred.")
    else:
        print("Cannot start application due to database connection or column detection issues.")