from PIL import Image
import imagehash
import numpy as np
from typing import List, Dict
import os  

class RGBHasher:
    def __init__(self):
        self.hash_size = 8 
        
    def generate_phash_rgb(self, image: Image.Image) -> int:
        image_resized = image.resize((16, 16), Image.LANCZOS)  
        
        img_array = np.array(image_resized)
        
        r_hash = imagehash.average_hash(Image.fromarray(img_array[:,:,0]), hash_size=self.hash_size)
        g_hash = imagehash.average_hash(Image.fromarray(img_array[:,:,1]), hash_size=self.hash_size)  
        b_hash = imagehash.average_hash(Image.fromarray(img_array[:,:,2]), hash_size=self.hash_size)
        
        r_int = int(str(r_hash), 16)
        g_int = int(str(g_hash), 16)
        b_int = int(str(b_hash), 16)
        
        combined_hash = (r_int << 128) | (g_int << 64) | b_int
        
        return combined_hash

    def to_bigints(self, input_hash: int) -> List[int]:

        bigints = []
        mask = (1 << 64) - 1  
        
        temp_hash = input_hash
        while temp_hash > 0:
            chunk = temp_hash & mask
            bigints.append(int(chunk))  
            temp_hash >>= 64
        
        if not bigints:
            bigints.append(0)
            
        return bigints

    def hash_image(self, image_path: str) -> dict:
        try:
            image = Image.open(image_path).convert('RGB')
            hash_int = self.generate_phash_rgb(image)
            hash_bigints = self.to_bigints(hash_int)
            hash_hex = format(hash_int, 'x')
            
            result = {
                'filename': os.path.basename(image_path),  
                'hash_int': hash_int,
                'hash_hex': hash_hex,
                'hash_bigints': hash_bigints,
                'num_bigints': len(hash_bigints)
            }
            
            for i, bigint in enumerate(hash_bigints):
                result[f'bigint{i+1}'] = int(bigint) 
            
            return result
        except Exception as e:
            print(f"Error hashing {image_path}: {e}")
            raise

def safe_int(value):
    """Convert value to Python int safely"""
    if hasattr(value, 'item'):  
        return int(value.item())
    return int(value)