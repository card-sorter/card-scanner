import csv
import mysql.connector
import pandas as pd

username = "########"  # Your mysql username
password = "########"  # Your mysql password
databasename = "pokemonDatabase"  # name of database we want to create

def create_db_connection():
    mydb = mysql.connector.connect(
        host="localhost",
        user= username,
        password=password,
        database=databasename,
        allow_local_infile=True
    )
    return mydb

def create_tables(mydb):
    with mydb.cursor() as cursor:
        cursor.execute('DROP TABLE IF EXISTS PokemonCards')
        cursor.execute("""
            CREATE TABLE PokemonCards (
                productId INT,
                name VARCHAR(255),
                cleanName VARCHAR(255),
                imageUrl VARCHAR(255),
                categoryId INT,
                groupId INT,
                url VARCHAR(255),
                modifiedOn VARCHAR(255),
                imageCount INT,
                extNumber VARCHAR(255),
                extRarity VARCHAR(255),
                extCardType VARCHAR(255),
                extHP INT,
                extStage VARCHAR(255),
                extAttack1 VARCHAR(255),
                extAttack2 VARCHAR(255),
                extWeakness VARCHAR(255),
                extRetreatCost INT,
                lowPrice DECIMAL(10,2),
                midPrice DECIMAL(10,2),
                highPrice DECIMAL(10,2),
                marketPrice DECIMAL(10,2),
                directLowPrice DECIMAL(10,2),
                subTypeName VARCHAR(255),
                extResistance VARCHAR(255),
                extCardText TEXT,
                extUPC VARCHAR(255)
            )
        """)
        mydb.commit()

def csv_to_mysql(mydb, csv_file):
    with mydb.cursor() as cursor:
        cursor.execute(f"""
            LOAD DATA LOCAL INFILE '{csv_file}'
            INTO TABLE PokemonCards
            FIELDS TERMINATED BY ','
            ENCLOSED BY '"'
            LINES TERMINATED BY '\n'
            IGNORE 1 ROWS
            (productId,
            name,
            cleanName,
            imageUrl,
            categoryId,
            groupId,
            url,
            modifiedOn,
            imageCount,
            extNumber,
            extRarity,
            extCardType,
            extHP,
            extStage,
            extAttack1,
            extAttack2,
            extWeakness,
            extRetreatCost,
            lowPrice,
            midPrice,
            highPrice,
            marketPrice,
            directLowPrice,
            subTypeName,
            extResistance,
            extCardText,
            extUPC)
        """)

        cursor.execute("""
                    DELETE FROM PokemonCards
                    WHERE extRarity IS NULL 
                    OR extRarity = ''
                    OR extRarity = 'Code Card'
                    OR subTypeName = 'Reverse Holofoil'
                """)

        mydb.commit()
        print("Data imported successfully with filters applied")

def get_row_count(mydb):
    with mydb.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM PokemonCards")
        row_count = cursor.fetchone()[0]
        print(f"Total rows in PokemonCards table: {row_count}")
        return row_count

def main():
    CSV_FILE = "######" #CSV File that has all the pokemon card data 
    
    mydb = create_db_connection()
    if mydb or mydb.is_connected():
        print("Database connection established successfully ( ˶ˆᗜˆ˵ )")
        create_tables(mydb)
        print(f"PokemonCards tables created successfully")
        csv_to_mysql(mydb, CSV_FILE)
        print(f"'{CSV_FILE}' successfully imported into MySQL")
        get_row_count(mydb)
        
        if mydb.is_connected():
            mydb.close()
            print("Database connection closed (×_×)")
    else: 
        print("Failed to establish database connection (｡•́︿•̀｡)")
        return
if __name__ == "__main__":
    main()
