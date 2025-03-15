import os
import urllib.request
import csv
from multiprocessing import Pool, Process

import tqdm

groups = "https://tcgcsv.com/tcgplayer/3/Groups.csv"

def fetch_list():
    urllib.request.urlretrieve(groups, "groups.csv")

def fetch_set(set_id):
    print("https://tcgcsv.com/tcgplayer/3/% s/ProductsAndPrices.csv" % set_id)
    urllib.request.urlretrieve("https://tcgcsv.com/tcgplayer/3/% s/ProductsAndPrices.csv" % set_id, "./sets/% s.csv" % set_id)

def fetch_csv():
    fetch_list()
    with open("groups.csv", newline='') as groupfile:
        groupreader = csv.reader(groupfile)
        next(groupreader)
        ids = []
        for row in groupreader:
            ids.append(row[0])
        with Pool(30) as p:
            p.map(fetch_set, ids)

def fetch_image(card_id):
    if not os.path.isfile("./images/% s.jpg" % card_id):
        try:
            urllib.request.urlretrieve("https://tcgplayer-cdn.tcgplayer.com/product/%s_400w.jpg" % card_id, "./images/% s.jpg" % card_id)
        except:
            print("https://tcgplayer-cdn.tcgplayer.com/product/%s_400w.jpg" % card_id)

def fetch_images():
    ids = []
    with open("groups.csv", newline='') as group_file:
        group_reader = csv.reader(group_file)
        next(group_reader)
        for row in group_reader:
            print(row[1])
            if "Jumbo Cards" in row[1] or "World Championship Decks" in row[1]:
                continue
            with open ("./sets/% s.csv" % row[0], encoding="utf8") as set_file:
                set_reader = csv.reader(set_file)
                target_col = 0
                cols = next(set_reader)
                for idx, item in enumerate(cols):
                    if item == "extRarity":
                        target_col = idx
                for card in set_reader:
                    if card[target_col] and "Code Card" not in card[1]:
                        ids.append(card[0])
    p = Pool(50)
    for _ in tqdm.tqdm(p.imap_unordered(fetch_image, ids), total=len(ids)):
        pass

if __name__ == "__main__":
    #fetch_csv()
    fetch_images()