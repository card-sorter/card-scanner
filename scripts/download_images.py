import os
import urllib.request
import csv
from multiprocessing import Pool

if not os.path.exists("images"):
    os.makedirs("images")

def fetch_image(card_id):
    """Download a single card image if it doesn't exist"""
    image_path = f"./images/{card_id}.jpg"
    if not os.path.isfile(image_path):
        try:
            image_url = f"https://tcgplayer-cdn.tcgplayer.com/product/{card_id}_400w.jpg"
            urllib.request.urlretrieve(image_url, image_path)
            return f"Downloaded {card_id}"
        except Exception as e:
            print(f"Failed to download card {card_id}: {e}")
            return None
    return f"Skipped {card_id} (already exists)"

def fetch_images():
    """Download images for all cards in all sets"""
    print("Starting image download for all cards...")
    
    with open("groups.csv", newline='', encoding='utf-8') as group_file:
        group_reader = csv.reader(group_file)
        next(group_reader) 
        
        for set_row in group_reader:
            set_id, set_name = set_row[0], set_row[1]
            print(f"\nProcessing set: {set_name} ({set_id})")
            
            set_filepath = f"./sets/{set_id}.csv"
            if not os.path.exists(set_filepath):
                print(f"Set file not found: {set_filepath}")
                continue
                
            with open(set_filepath, newline='', encoding='utf-8') as set_file:
                set_reader = csv.reader(set_file)
                headers = next(set_reader)
                
                try:
                    product_id_index = headers.index("productId")
                except ValueError:
                    print("No productId column found in CSV")
                    continue
                
                card_ids = []
                for card_row in set_reader:
                    if len(card_row) > product_id_index:
                        card_ids.append(card_row[product_id_index])
                
                print(f"Found {len(card_ids)} cards in set")

                with Pool(30) as p:
                    results = p.map(fetch_image, card_ids)

                downloaded = sum(1 for r in results if r and "Downloaded" in r)
                skipped = sum(1 for r in results if r and "Skipped" in r)
                failed = len(card_ids) - downloaded - skipped
                
                print(f"Set complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")

if __name__ == "__main__":
    print("Starting card image download for ALL cards...")
    fetch_images()
    print("\nImage download process completed.")
