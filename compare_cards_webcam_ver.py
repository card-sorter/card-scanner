import os
import cv2
import pandas as pd
from PIL import Image
import imagehash

#To compute image hash
def compute_image_hash(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #Change opencv numpy array to PIL image objects
    return imagehash.average_hash(pil_image)

#Comparison to find closest match in the dataset
def find_match(card_hash, df):
    min_distance = float('inf')
    best_match = None

    for index, row in df.iterrows():
        card_id = row['productId']  #Use the productId column for card IDs
        card_name = row['name']  #Use the name column for card names

        #Load the card image from the images folder
        image_path = f"./images/{card_id}.jpg"
        if not os.path.exists(image_path):
            continue
        card_image = cv2.imread(image_path)
        if card_image is None:
            continue

        #Compute the image hashes
        card_image_hash = compute_image_hash(card_image) #Execute for every card in the dataset

        #Calc hamming distance betw hashes
        distance = card_hash - card_image_hash

        if distance < min_distance:
            min_distance = distance
            best_match = card_name

    return best_match, min_distance
