import os
import urllib.request
import csv
from multiprocessing import Pool

#Make folder for the images
if not os.path.exists("images"):
    os.makedirs("images")

#To download a card
def fetch_image(card_id):
    if not os.path.isfile(f"./images/{card_id}.jpg"):
        try:
            image_url = f"https://tcgplayer-cdn.tcgplayer.com/product/{card_id}_400w.jpg"
            urllib.request.urlretrieve(image_url, f"./images/{card_id}.jpg")
        except Exception as e:
            print(f"Can't DL this card {card_id}: {e}")

#To download all the cards
def fetch_images():
    with open("groups.csv", newline='', encoding='utf-8') as group_file:
        group_reader = csv.reader(group_file)
        next(group_reader)  #Skip header
        for row in group_reader:
            print(f"Downloading images for set: {row[1]}")
            with open(f"./sets/{row[0]}.csv", newline='', encoding='utf-8') as set_file:
                set_reader = csv.reader(set_file)
                target_col = 0
                cols = next(set_reader)  #Get header row
                for idx, item in enumerate(cols):
                    if item == "extRarity":
                        target_col = idx
                ids = []
                for card in set_reader:
                    if "Rare" in card[target_col]:  #Filter for rare cards
                        ids.append(card[0])
                with Pool(30) as p:  #This to help with downloading, multiprocesses please god this shit is taking so long
                    p.map(fetch_image, ids)

if __name__ == "__main__":
    print("Donloading cardm pi9cs")
    fetch_images()
    print("Card images download perfecta.")