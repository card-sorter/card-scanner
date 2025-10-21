from typing import Tuple, List
from PIL import Image
from config import MODEL_PATH
from ultralytics import YOLO
from common import Card

class CardScanner:
    def __init__(self):
        self.model = None

    async def load_model(self, model_path = MODEL_PATH):
        self.model = await YOLO(model=model_path)

    async def _detect_cards(self, image: Image.Image)->List[List[Tuple[int, int]]]:
        """
        Return a list of tuples containing the position of the card corners.
        :param card:
        :return:
        """
        pass

    def _crop_card(self, image: Image.Image, points: List[Tuple[int, int]])->Card:
        """
        Crop and deskew the image from the list of points.
        :param card:
        :param points:
        :return:
        """
        pass

    def _phash(self, image: Image.Image)->int:
        """
        Generate phash from image.
        :param image:
        :return:
        """
        pass

    def _to_bigints(self, input: int)-> List[int]:
        """
        Convert input to a number of bigints.
        SQL bigints are 64 bits
        :param input:
        :return:
        """
        pass

    async def scan_cards(self, image: Image.Image)->List[Card]:
        """
        Scan an image and return a List of cards found in the image, alongside their hashes.
        :param image:
        :return:
        """
        ret = []
        points = await self._detect_cards(image)
        for i in points:
            ret.append(self._crop_card(image, i))
        for i in ret:
            i.hash = self._phash(i.image)
            i.hash = self._to_bigints(i.hash)
        return ret