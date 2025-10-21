from PIL import Image

class Card:
    def __init__(self, image: Image.Image):
        self.image = image
        self.values = {}
        self.hash = None
        self.distance = None