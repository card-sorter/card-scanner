import os
import urllib.request
import csv
from multiprocessing import Pool, Process

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
        for row in groupreader:
            fetch_set(row[0])

def fetch_image(card_id):
    if not os.path.isfile("./images/% s.jpg" % card_id):
        try:
            urllib.request.urlretrieve("https://tcgplayer-cdn.tcgplayer.com/product/%s_400w.jpg" % card_id, "./images/% s.jpg" % card_id)
        except:
            print("https://tcgplayer-cdn.tcgplayer.com/product/%s_400w.jpg" % card_id)

def fetch_images():
    with open("groups.csv", newline='') as group_file:
        group_reader = csv.reader(group_file)
        next(group_reader)
        for row in group_reader:
            print(row[1])
            with open ("./sets/% s.csv" % row[0]) as set_file:
                set_reader = csv.reader(set_file)
                target_col = 0
                cols = next(set_reader)
                for idx, item in enumerate(cols):
                    if item == "extRarity":
                        target_col = idx
                ids = []
                for card in set_reader:
                    if "Rare" in card[target_col]:
                        ids.append(card[0])
                with Pool(30) as p:
                    p.map(fetch_image, ids)

if __name__ == "__main__":
    fetch_images()