from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
import os
import pandas as pd
from glob import glob
import time
import mysql.connector
import csv
import re

# Configuration
MODEL_PATH = "best.pt"
IMAGE_DIR = "images"
SETS_DIR = "sets"
TEST_IMAGE_PATH = ".jpg"
OUTPUT_DIR = "detected_cards"
TARGET_SIZE = (600, 840)
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'pokemon_card_db',
    'user': 'Arc',
    'password': 'Pelican2001',
    'allow_local_infile': True
}

class CardIdentifier:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()
        self.model.conf = 0.5
        
        self.db_conn = self._init_db_connection()
        self._import_csv_tables()
        self.card_names = self._load_card_names_from_db()
        self.hash_db = self._build_hash_database()

    def _init_db_connection(self):
        """Initialize database connection only"""
        config_without_db = MYSQL_CONFIG.copy()
        database_name = config_without_db.pop('database')
        
        try:
            conn = mysql.connector.connect(**config_without_db)
            cursor = conn.cursor()
            
            cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in cursor.fetchall()]
            
            if database_name not in databases:
                print(f"Database '{database_name}' not found. Creating it...")
                cursor.execute(f"CREATE DATABASE {database_name}")
                print(f"Database '{database_name}' created successfully!")
            
            cursor.close()
            conn.close()
            
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            print(" Database connection established successfully!")
            return conn
            
        except mysql.connector.Error as e:
            print(f"MySQL connection error: {e}")
            raise

    def _infer_data_type(self, value):
        value = str(value).strip()
        if re.fullmatch(r"-?\d+\.\d+", value):
            return "DECIMAL(10,2)"
        elif value.isdigit():
            return "INT"
        elif value.lower() in ('true', 'false'):
            return "BOOLEAN"
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
            return "DATETIME"
        else:
            return "VARCHAR(255)"

    def _create_table(self, cursor, table_name, col_def):
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} ({', '.join(col_def)});") 
        self.db_conn.commit()

    def _read_csv_columns(self, csv_file):
        col_def = []
        
        try:
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                columns = next(reader, None)  
                
                if not columns:
                    print(f"Warning: {csv_file} has no headers, skipping...")
                    return None
                
                try:
                    sample_data = next(reader, [])
                except StopIteration:
                    sample_data = ['' for _ in columns]
                
                for col_name, sample_value in zip(columns, sample_data):
                    inferred_type = self._infer_data_type(sample_value)
                    col_def.append(f"`{col_name}` {inferred_type}")
            
            return col_def
            
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            return None

    def _csv_to_mysql(self, cursor, csv_file, table_name):
        column_definitions = self._read_csv_columns(csv_file)
        if column_definitions is None:
            return False
            
        self._create_table(cursor, table_name, column_definitions)  

        try:
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                placeholders = ', '.join(['%s'] * len(row))
                columns = ', '.join([f"`{col}`" for col in df.columns])
                insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                
                try:
                    cursor.execute(insert_query, tuple(row))
                except Exception as e:
                    print(f"Error inserting row in {table_name}: {e}")
            
            self.db_conn.commit()
            print(f" Data imported successfully into {table_name} ({len(df)} rows)")
            return True
            
        except Exception as e:
            print(f"Error importing {csv_file}: {e}")
            return False

    def _import_csv_tables(self):
        csv_files = glob(os.path.join(SETS_DIR, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {SETS_DIR}")
            return
        
        cursor = self.db_conn.cursor()
        imported_count = 0
        
        for csv_file in csv_files:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"Importing {csv_file} as table '{table_name}'...")
            try:
                if self._csv_to_mysql(cursor, csv_file, table_name):
                    imported_count += 1
            except Exception as e:
                print(f"Error importing {csv_file}: {e}")
        
        cursor.close()
        print(f" Imported {imported_count} out of {len(csv_files)} CSV files")

    def _load_card_names_from_db(self):
        card_data = {}
        
        try:
            cursor = self.db_conn.cursor()
            
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            for table in tables:
                try:
                    cursor.execute(f"""
                        SELECT COLUMN_NAME 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s 
                        AND COLUMN_NAME IN ('productId', 'name')
                    """, (MYSQL_CONFIG['database'], table))
                    
                    columns = [col[0] for col in cursor.fetchall()]
                    
                    if 'productId' in columns and 'name' in columns:
                        cursor.execute(f"SELECT productId, name FROM {table}")
                        rows = cursor.fetchall()
                        
                        for product_id, name in rows:
                            card_id = str(product_id).strip()
                            card_name = str(name).strip()
                            card_data[card_id] = card_name
                            print(f"  Loaded card: {card_id} -> {card_name}")
                            
                except Exception as e:
                    print(f"Error reading from table {table}: {e}")
                    continue
            
            cursor.close()
            print(f" Loaded {len(card_data)} card names from database")
            
        except Exception as e:
            print(f"Error loading card names from database: {e}")
        
        return card_data

    def _preprocess_image(self, img):
        """Enhanced image preprocessing for better hashing"""
        img = img.resize((300, 420), Image.Resampling.LANCZOS)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        return img

    def _compute_advanced_hashes(self, img):
        """Compute multiple hash types with proper size limits"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = self._preprocess_image(img)
        
        r, g, b = img.split()
        
        hashes = {}
        
        hash_size = 8
        
        hashes['phash_r'] = int(str(imagehash.phash(r, hash_size=hash_size)), 16)
        hashes['phash_g'] = int(str(imagehash.phash(g, hash_size=hash_size)), 16)
        hashes['phash_b'] = int(str(imagehash.phash(b, hash_size=hash_size)), 16)

        hashes['dhash_r'] = int(str(imagehash.dhash(r, hash_size=hash_size)), 16)
        hashes['dhash_g'] = int(str(imagehash.dhash(g, hash_size=hash_size)), 16)
        hashes['dhash_b'] = int(str(imagehash.dhash(b, hash_size=hash_size)), 16)
        
        try:
            hashes['whash_r'] = int(str(imagehash.whash(r, hash_size=hash_size)), 16)
            hashes['whash_g'] = int(str(imagehash.whash(g, hash_size=hash_size)), 16)
            hashes['whash_b'] = int(str(imagehash.whash(b, hash_size=hash_size)), 16)
        except:
            hashes['whash_r'] = int(str(imagehash.average_hash(r, hash_size=hash_size)), 16)
            hashes['whash_g'] = int(str(imagehash.average_hash(g, hash_size=hash_size)), 16)
            hashes['whash_b'] = int(str(imagehash.average_hash(b, hash_size=hash_size)), 16)
        
        hashes['ahash_r'] = int(str(imagehash.average_hash(r, hash_size=hash_size)), 16)
        hashes['ahash_g'] = int(str(imagehash.average_hash(g, hash_size=hash_size)), 16)
        hashes['ahash_b'] = int(str(imagehash.average_hash(b, hash_size=hash_size)), 16)
        
        return hashes

    def _build_hash_database(self):
        """Build enhanced hash database with proper data types"""
        hash_db = {}
        
        try:
            cursor = self.db_conn.cursor()
            
            cursor.execute("DROP TABLE IF EXISTS card_hashes")
            cursor.execute("""
                CREATE TABLE card_hashes (
                    card_id VARCHAR(255) PRIMARY KEY,
                    phash_r BIGINT UNSIGNED, 
                    phash_g BIGINT UNSIGNED,
                    phash_b BIGINT UNSIGNED,
                    dhash_r BIGINT UNSIGNED, 
                    dhash_g BIGINT UNSIGNED,
                    dhash_b BIGINT UNSIGNED,
                    whash_r BIGINT UNSIGNED,
                    whash_g BIGINT UNSIGNED,
                    whash_b BIGINT UNSIGNED,
                    ahash_r BIGINT UNSIGNED,
                    ahash_g BIGINT UNSIGNED,
                    ahash_b BIGINT UNSIGNED
                )
            """)
            self.db_conn.commit()
            
            print("Computing RGB hashes from images (using 8x8 hash size)...")
            image_files = glob(os.path.join(IMAGE_DIR, "*.jpg"))
            
            if not image_files:
                print(f" Warning: No JPG images found in {IMAGE_DIR} directory!")
                return hash_db
            
            processed_count = 0
            for img_path in image_files:
                card_id = os.path.splitext(os.path.basename(img_path))[0]
                try:
                    with Image.open(img_path) as img:
                        hashes = self._compute_advanced_hashes(img)
                        
                        max_bigint = 18446744073709551615
                        for key, value in hashes.items():
                            if value > max_bigint:
                                print(f"  Hash value too large for {key}: {value}")
                                hashes[key] = value & max_bigint
                        
                        cursor.execute(
                            """INSERT INTO card_hashes 
                            (card_id, phash_r, phash_g, phash_b, dhash_r, dhash_g, dhash_b,
                             whash_r, whash_g, whash_b, ahash_r, ahash_g, ahash_b) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (card_id, 
                             hashes['phash_r'], hashes['phash_g'], hashes['phash_b'],
                             hashes['dhash_r'], hashes['dhash_g'], hashes['dhash_b'],
                             hashes['whash_r'], hashes['whash_g'], hashes['whash_b'],
                             hashes['ahash_r'], hashes['ahash_g'], hashes['ahash_b'])
                        )
                        hash_db[card_id] = hashes
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"  Processed {processed_count} images...")
                            
                except Exception as e:
                    print(f" Can't process {img_path}: {str(e)}")
            
            self.db_conn.commit()
            print(f" Successfully computed and stored {processed_count} card hashes")
            cursor.close()
            
        except Exception as e:
            print(f" Error building hash database: {e}")
        
        return hash_db

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
        mask_uint8 = (mask * 255).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        mask_roi = mask_uint8[y1:y2, x1:x2]
        
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        
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

    def _calculate_weighted_distance(self, query_hashes, db_hashes):
        """Calculate weighted distance with different importance for hash types"""
        weights = {
            'phash': 1.5,
            'dhash': 1.2,
            'whash': 1.0,
            'ahash': 0.8
        }
        
        total_distance = 0
        
        for channel in ['r', 'g', 'b']:
            for hash_type in ['phash', 'dhash', 'whash', 'ahash']:
                query_hash = query_hashes[f'{hash_type}_{channel}']
                db_hash = db_hashes[f'{hash_type}_{channel}']
                distance = bin(query_hash ^ db_hash).count('1')
                total_distance += distance * weights[hash_type]
        
        return total_distance

    def _identify_card(self, cropped_img):
        """Enhanced card identification with multiple hash types and weighting"""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            query_hashes = self._compute_advanced_hashes(pil_img)
            
            best_match = None
            min_distance = float('inf')
            second_best_match = None
            second_min_distance = float('inf')
            
            for card_id, db_hashes in self.hash_db.items():
                distance = self._calculate_weighted_distance(query_hashes, db_hashes)
                
                if distance < min_distance:
                    second_best_match = best_match
                    second_min_distance = min_distance
                    min_distance = distance
                    best_match = card_id
                elif distance < second_min_distance:
                    second_min_distance = distance
                    second_best_match = card_id
            
            confidence_ratio = 1.0
            if second_min_distance < float('inf'):
                confidence_ratio = second_min_distance / (min_distance + 0.001)
            
            high_confidence_threshold = 1.5
            
            if confidence_ratio > high_confidence_threshold and min_distance < 80:
                return self.card_names.get(best_match, "Unknown"), min_distance
            elif min_distance < 120:
                return self.card_names.get(best_match, "Unknown"), min_distance
            else:
                return "Unknown (Low Confidence)", min_distance
            
        except Exception as e:
            print(f" Error identifying card: {str(e)}")
            return "Error", float('inf')

    def _process_detection(self, frame):
        results = self.model(frame)[0]
        annotated = results.plot()
        
        if results.boxes and results.masks:
            best_aligned = None
            best_confidence = 0
            best_box = None
            
            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                confidence = box.conf.item()
                box_coords = box.xyxy[0].cpu().numpy()
                mask_data = mask.data.cpu().numpy().squeeze()
                
                aligned = self._crop_and_deskew(frame, box_coords, mask_data)
                
                if aligned is not None and aligned.size > 0:
                    card_area = (box_coords[2] - box_coords[0]) * (box_coords[3] - box_coords[1])
                    score = confidence * card_area
                    
                    if score > best_confidence:
                        best_confidence = score
                        best_aligned = aligned
                        best_box = box_coords
            
            if best_aligned is not None:
                timestamp = int(time.time())
                output_path = os.path.join(OUTPUT_DIR, f"card_{timestamp}.jpg")
                cv2.imwrite(output_path, best_aligned)
                card_name, confidence = self._identify_card(best_aligned)
                return annotated, best_aligned, f"{card_name} ({confidence:.1f})", output_path
        
        return annotated, None, None, None

    def process_image_file(self, image_path):
        if not os.path.exists(image_path):
            print(f" Error: Image file '{image_path}' not found!")
            return
        
        print(f" Processing image: {image_path}")
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f" Error: Could not read image from {image_path}")
            return
        
        processed_frame = self._preprocess_frame(frame)
        
        processed_frame, cropped, card_info, output_path = self._process_detection(processed_frame)
        
        if cropped is not None and card_info is not None:
            print(f"Card identified: {card_info}")
            print(f"Saved aligned card to: {output_path}")
            
            cv2.imshow("Original Image with Detection", processed_frame)
            cv2.imshow("Aligned Card", cropped)
            
            text_frame = processed_frame.copy()
            cv2.putText(text_frame, card_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Result", text_frame)
            
            print("Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(" No card detected in the image.")
            cv2.imshow("Original Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _preprocess_frame(self, frame):
        """Preprocess frame for better detection"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame

if __name__ == "__main__":
    try:
        identifier = CardIdentifier()
        identifier.process_image_file(TEST_IMAGE_PATH)
        identifier.db_conn.close()
        
    except Exception as e:
        print(f" Failed to initialize card identifier: {e}")