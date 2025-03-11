import urllib.request
import csv

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

if __name__ == "__main__":
