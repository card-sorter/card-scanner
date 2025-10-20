import mysql.connector
import pandas as pd
import os
from glob import glob
import csv
from PIL import Image
import imagehash

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'csv_import_db'
}

def setup_database():
    """Set up the database structure and import CSV data"""
    try:
        # Connect to MySQL (without specific database first)
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}`")
        cursor.execute(f"USE `{DB_CONFIG['database']}`")
        
        # Create main cards table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `cards` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                productId VARCHAR(50) NOT NULL,
                name VARCHAR(255) NOT NULL,
                cleanName VARCHAR(255),
                imageUrl VARCHAR(500),
                categoryId INT,
                groupId INT,
                url VARCHAR(500),
                modifiedOn DATETIME,
                imageCount INT,
                extNumber VARCHAR(50),
                extRarity VARCHAR(50),
                extCardType VARCHAR(50),
                extHP INT,
                extStage VARCHAR(50),
                extAttack1 TEXT,
                extAttack2 TEXT,
                extWeakness VARCHAR(50),
                extRetreatCost VARCHAR(50),
                lowPrice DECIMAL(10,2),
                midPrice DECIMAL(10,2),
                highPrice DECIMAL(10,2),
                marketPrice DECIMAL(10,2),
                directLowPrice DECIMAL(10,2),
                subTypeName VARCHAR(100),
                extResistance VARCHAR(50),
                extCardText TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_productId (productId)
            )
        """)
        
        # Create card_hashes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `card_hashes` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                productId VARCHAR(50) NOT NULL,
                phash VARCHAR(64) NOT NULL,
                dhash VARCHAR(64) NOT NULL,
                colorhash VARCHAR(64) NOT NULL,
                image_width INT,
                image_height INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (productId) REFERENCES cards(productId),
                INDEX idx_hashes (phash, dhash),
                UNIQUE KEY unique_product (productId)
            )
        """)
        
        print(" Database structure created successfully!")
        
        # Import all CSV files from sets directory
        sets_dir = "sets"
        if os.path.exists(sets_dir):
            csv_files = glob(os.path.join(sets_dir, "*.csv"))
            total_cards = 0
            
            for csv_file in csv_files:
                print(f" Processing {os.path.basename(csv_file)}...")
                
                try:
                    # Read CSV file with error handling
                    df = pd.read_csv(csv_file, low_memory=False)
                    
                    # Insert data into cards table
                    inserted_count = 0
                    for _, row in df.iterrows():
                        try:
                            # Extract only the columns we need, handle missing columns gracefully
                            card_data = {
                                'productId': str(row.get('productId', '')),
                                'name': str(row.get('name', '')),
                                'cleanName': str(row.get('cleanName', '')) if 'cleanName' in row else None,
                                'imageUrl': str(row.get('imageUrl', '')) if 'imageUrl' in row else None,
                                'categoryId': int(row.get('categoryId', 0)) if 'categoryId' in row and pd.notna(row.get('categoryId')) else None,
                                'groupId': int(row.get('groupId', 0)) if 'groupId' in row and pd.notna(row.get('groupId')) else None,
                                'url': str(row.get('url', '')) if 'url' in row else None,
                                'modifiedOn': row.get('modifiedOn') if 'modifiedOn' in row and pd.notna(row.get('modifiedOn')) else None,
                                'imageCount': int(row.get('imageCount', 0)) if 'imageCount' in row and pd.notna(row.get('imageCount')) else None,
                                'extNumber': str(row.get('extNumber', '')) if 'extNumber' in row else None,
                                'extRarity': str(row.get('extRarity', '')) if 'extRarity' in row else None,
                                'extCardType': str(row.get('extCardType', '')) if 'extCardType' in row else None,
                                'extHP': int(row.get('extHP', 0)) if 'extHP' in row and pd.notna(row.get('extHP')) else None,
                                'extStage': str(row.get('extStage', '')) if 'extStage' in row else None,
                                'extAttack1': str(row.get('extAttack1', '')) if 'extAttack1' in row else None,
                                'extAttack2': str(row.get('extAttack2', '')) if 'extAttack2' in row else None,
                                'extWeakness': str(row.get('extWeakness', '')) if 'extWeakness' in row else None,
                                'extRetreatCost': str(row.get('extRetreatCost', '')) if 'extRetreatCost' in row else None,
                                'lowPrice': float(row.get('lowPrice', 0)) if 'lowPrice' in row and pd.notna(row.get('lowPrice')) else None,
                                'midPrice': float(row.get('midPrice', 0)) if 'midPrice' in row and pd.notna(row.get('midPrice')) else None,
                                'highPrice': float(row.get('highPrice', 0)) if 'highPrice' in row and pd.notna(row.get('highPrice')) else None,
                                'marketPrice': float(row.get('marketPrice', 0)) if 'marketPrice' in row and pd.notna(row.get('marketPrice')) else None,
                                'directLowPrice': float(row.get('directLowPrice', 0)) if 'directLowPrice' in row and pd.notna(row.get('directLowPrice')) else None,
                                'subTypeName': str(row.get('subTypeName', '')) if 'subTypeName' in row else None,
                                'extResistance': str(row.get('extResistance', '')) if 'extResistance' in row else None,
                                'extCardText': str(row.get('extCardText', '')) if 'extCardText' in row else None,
                            }
                            
                            # Insert the card (ignore duplicates)
                            cursor.execute("""
                                INSERT IGNORE INTO `cards` (
                                    productId, name, cleanName, imageUrl, categoryId, groupId, url, 
                                    modifiedOn, imageCount, extNumber, extRarity, extCardType, extHP, 
                                    extStage, extAttack1, extAttack2, extWeakness, extRetreatCost,
                                    lowPrice, midPrice, highPrice, marketPrice, directLowPrice,
                                    subTypeName, extResistance, extCardText
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                card_data['productId'], card_data['name'], card_data['cleanName'],
                                card_data['imageUrl'], card_data['categoryId'], card_data['groupId'],
                                card_data['url'], card_data['modifiedOn'], card_data['imageCount'],
                                card_data['extNumber'], card_data['extRarity'], card_data['extCardType'],
                                card_data['extHP'], card_data['extStage'], card_data['extAttack1'],
                                card_data['extAttack2'], card_data['extWeakness'], card_data['extRetreatCost'],
                                card_data['lowPrice'], card_data['midPrice'], card_data['highPrice'],
                                card_data['marketPrice'], card_data['directLowPrice'], card_data['subTypeName'],
                                card_data['extResistance'], card_data['extCardText']
                            ))
                            
                            if cursor.rowcount > 0:
                                inserted_count += 1
                                total_cards += 1
                            
                        except Exception as e:
                            print(f"     Error inserting row: {e}")
                            continue
                    
                    print(f"   Imported {inserted_count} cards from {os.path.basename(csv_file)}")
                    
                except Exception as e:
                    print(f"   Error processing {csv_file}: {e}")
                    continue
            
            print(f"\n Successfully imported {total_cards} total cards!")
        
        else:
            print(f" Sets directory '{sets_dir}' not found!")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n Database setup completed!")
        return True
        
    except Exception as e:
        print(f" Database setup failed: {e}")
        return False

def compute_and_store_image_hashes():
    """Compute hashes for all images and store them in the database"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("\n Computing image hashes...")
        images_dir = "images"
        if not os.path.exists(images_dir):
            print(f" Images directory '{images_dir}' not found!")
            return False
        
        # Look for all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(images_dir, ext)))
            image_files.extend(glob(os.path.join(images_dir, ext.upper())))
        
        print(f" Found {len(image_files)} image files")
        
        processed_count = 0
        error_count = 0
        
        for img_path in image_files:
            # Extract card ID from filename (remove extension)
            filename = os.path.basename(img_path)
            card_id = os.path.splitext(filename)[0]
            
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Get image dimensions
                    width, height = img.size
                    
                    # Resize image to standard size for consistent hashing
                    img_resized = img.resize((300, 420))
                    
                    # Compute hashes
                    phash = str(imagehash.phash(img_resized))
                    dhash = str(imagehash.dhash(img_resized))
                    colorhash = str(imagehash.colorhash(img_resized))
                    
                    # Insert or update hash in database
                    cursor.execute("""
                        INSERT INTO `card_hashes` 
                        (productId, phash, dhash, colorhash, image_width, image_height) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        phash = VALUES(phash), 
                        dhash = VALUES(dhash), 
                        colorhash = VALUES(colorhash),
                        image_width = VALUES(image_width),
                        image_height = VALUES(image_height)
                    """, (card_id, phash, dhash, colorhash, width, height))
                    
                    processed_count += 1
                    print(f"   Processed: {filename}")
                        
            except Exception as e:
                error_count += 1
                print(f"   Can't process image {filename}: {str(e)}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n Successfully processed {processed_count} images")
        if error_count > 0:
            print(f"  Failed to process {error_count} images")
        
        return True
        
    except Exception as e:
        print(f" Error computing image hashes: {e}")
        return False

def test_database():
    """Test the database connection and data"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Count total cards
        cursor.execute("SELECT COUNT(*) FROM `cards`")
        total_cards = cursor.fetchone()[0]
        print(f" Total cards in database: {total_cards}")
        
        # Count total hashes
        cursor.execute("SELECT COUNT(*) FROM `card_hashes`")
        total_hashes = cursor.fetchone()[0]
        print(f" Total image hashes in database: {total_hashes}")
        
        # Show some sample cards with hashes
        cursor.execute("""
            SELECT c.productId, c.name, h.phash, h.dhash 
            FROM `cards` c 
            LEFT JOIN `card_hashes` h ON c.productId = h.productId 
            LIMIT 5
        """)
        sample_data = cursor.fetchall()
        print(" Sample cards with hashes:")
        for card in sample_data:
            print(f"   {card[0]}: {card[1]}")
            if card[2]:
                print(f"      phash: {card[2][:20]}...")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f" Database test failed: {e}")
        return False

if __name__ == "__main__":
    print(" Pokemon Card Database Setup")
    print("=" * 50)
    
    if setup_database():
        print("\n Computing image hashes...")
        if compute_and_store_image_hashes():
            print("\n🔍 Testing database...")
            test_database()
        else:
            print("\n Hash computation failed!")
    else:

        print("\n Setup failed!")
