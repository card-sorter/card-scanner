import os
import urllib.request
import csv

#Make folder
if not os.path.exists("sets"):
    os.makedirs("sets")

#Download shit from here
groups_url = "https://tcgcsv.com/tcgplayer/3/Groups.csv"

#Auto download the groups excel
def fetch_list():
    urllib.request.urlretrieve(groups_url, "groups.csv")

#To auto download a set within
def fetch_set(set_id):
    set_url = f"https://tcgcsv.com/tcgplayer/3/{set_id}/ProductsAndPrices.csv"
    urllib.request.urlretrieve(set_url, f"./sets/{set_id}.csv")

#To download all the csv files
def fetch_csv():
    fetch_list()
    with open("groups.csv", newline='', encoding='utf-8') as groupfile:
        groupreader = csv.reader(groupfile)
        next(groupreader) 
        for row in groupreader:
            fetch_set(row[0])

if __name__ == "__main__":
    print("CSV files... Dadadadadaddownloading")
    fetch_csv()
    print("CSV files download done.")
