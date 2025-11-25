from PIL import Image
from typing import Dict, Any, List, Optional

class Card:
    def __init__(self, image: Image.Image):
        self.image = image
        self.values: Dict[str, Any] = {}  
        self.hash: Optional[int] = None
        self.hash_bigints: List[int] = []  # hash as list of bigints for SQL
        self.distance: Optional[int] = None 
        
    def to_dict(self) -> Dict[str, Any]: #json serialization
        return {
            'values': self.values,
            'hash': self.hash,
            'hash_bigints': self.hash_bigints,
            'distance': self.distance,
            'image_size': self.image.size if self.image else None
        }