from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import imagehash
import os
import pandas as pd
from glob import glob
import time

# Configuration
MODEL_PATH = "best.pt" #Model jack trained
IMAGE_DIR = "images" #This is where it draws the thousands of images from. to be replaced with wen's database
SETS_DIR = "sets" #Same for this, CSV is where all the IDs are at, replaceable
PHONE_IP = "" #Enter your phone IP here, uh, if we need to swap to direct connection ill re-update it or whatev
DROIDCAM_PORT = "4747" #Webcam
OUTPUT_DIR = "detected_cards" #This takes the shot and save it as a jpg.
TARGET_SIZE = (600, 840) #size

class CardIdentifier:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True) #Make detected_cards directory if it don't exist
        self.model = YOLO(MODEL_PATH) #load YOLO model which is jack's model.
        self.hash_db = self._create_hash_db() #build hash database of the reference images.
        self.card_names = self._load_card_names() #loads card names from CSVs
        self._init_video_stream() #setup droidcam video feed
        self.model.fuse() #Optimizes yolo model for inference
        self.model.conf = 0.7 #threshold set at 70% confidence so only ≥ are kept

    def _init_video_stream(self):
        self.cap = cv2.VideoCapture(f"http://{PHONE_IP}:{DROIDCAM_PORT}/video")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) #decreases latency by reducing the frame buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30) #tries to get 30FPS, tries to
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #Use motion-jpg compression to attempt smoother streaming

    def _load_card_names(self):
        card_data = {}
        for csv_file in glob(os.path.join(SETS_DIR, "*.csv")): #find all csv in sets
            try:
                if os.path.getsize(csv_file) == 0: continue #skip empty files
                df = pd.read_csv(csv_file) #read csv into pandas dataframe.
                if {'productId', 'name'}.issubset(df.columns): #check required columns
                    for _, row in df.iterrows():
                        card_id = str(row['productId']).strip()
                        card_data[card_id] = row['name'].strip() 
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
        return card_data

    def _create_hash_db(self):
        hash_db = {}
        for img_path in glob(os.path.join(IMAGE_DIR, "*.jpg")): #find the jpgs in images
            card_id = os.path.splitext(os.path.basename(img_path))[0] #extract the filename which is the Id
            try:
                with Image.open(img_path) as img:
                    hash_db[card_id] = {
                        'phash': imagehash.phash(img), #percep hash, computes hash invariant to scaling + minor distortions
                        'dhash': imagehash.dhash(img) #difference hash, computes hash based on pixel differences, fast but inaccurate
                    }
            except Exception as e:
                print(f"Can't read thi {img_path}: {str(e)}")
        return hash_db

    def _order_points(self, pts): #gives 4 corner points for pers transform (req by cv2.getPerspectiveTransform)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 
        return rect

    def _crop_and_deskew(self, frame, box, mask):
        mask_uint8 = (mask * 255).astype(np.uint8) #yolo mask 0-1 floats to 0-255 for opencv convert
        x1, y1, x2, y2 = map(int, box)
        mask_roi = mask_uint8[y1:y2, x1:x2]
        
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find the outline for the card
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour) #fits rectangle to any rotations of the card
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        
        ordered_points = self._order_points(box_points) #reorder and offsets the points
        ordered_points[:, 0] += x1
        ordered_points[:, 1] += y1
        
        # Pers Tran
        src_pts = ordered_points.astype('float32') #card corners
        dst_pts = np.array([
            [0, 0],
            [TARGET_SIZE[0], 0],
            [TARGET_SIZE[0], TARGET_SIZE[1]],
            [0, TARGET_SIZE[1]]], dtype='float32')
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts) #compute transforms
        aligned = cv2.warpPerspective(frame, matrix, TARGET_SIZE) #apply
        return aligned

    def _identify_card(self, cropped_img):
        """Match card using dual hashing"""
        try:
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            query_phash = imagehash.phash(pil_img)
            query_dhash = imagehash.dhash(pil_img) #compute hashes for detected card
            
            best_match = None
            min_distance = float('inf')
            
            for card_id, hashes in self.hash_db.items():
                distance = (query_phash - hashes['phash']) + (query_dhash - hashes['dhash'])
                if distance < min_distance:
                    min_distance = distance
                    best_match = card_id #measure similiarity between hashes, lower better
                    
            return self.card_names.get(best_match, "Unknown"), min_distance #return card name or unknown if cant find
        except Exception as e:
            print(f"Can't read thi: {str(e)}")
            return "Rong", float('inf')

    def _process_detection(self, frame):
        results = self.model(frame)[0] #yolo inference
        annotated = results.plot() #draw bounding boxes on frame
        
        if results.boxes and results.masks:
            confidences = [box.conf.item() for box in results.boxes]
            best_idx = np.argmax(confidences) #picks most confident detection
            
            box = results.boxes[best_idx].xyxy[0].cpu().numpy() #bounding box
            mask = results.masks[best_idx].data.cpu().numpy().squeeze() #segmentation mask
            
            aligned = self._crop_and_deskew(frame, box, mask) #de-skew the card
            
            if aligned is not None and aligned.size > 0:
                timestamp = int(time.time())
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"card_{timestamp}.jpg"), aligned)# this saves it
                card_name, confidence = self._identify_card(aligned) #identify the card
                return annotated, aligned, f"{card_name} ({confidence:.1f})" #return result
        
        return annotated, None, None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Frame read error")
                time.sleep(0.1)
                continue
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): #press 's' key to screenshot etc
                processed_frame, cropped, card_info = self._process_detection(frame)
                
                if cropped is not None:
                    cv2.imshow("Aligned Card", cropped) #show de-skewed car
                    cv2.putText(processed_frame, card_info, (10, 30), #overlay card name
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            else:
                processed_frame = self.model(frame)[0].plot() #only show detection
            
            cv2.imshow("Pokemon Scanner - 'S' to Scan | 'Q' to Quit", processed_frame)
            
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    identifier = CardIdentifier()
    identifier.run()
