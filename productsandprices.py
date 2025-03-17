import csv,re
import mysql.connector

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

def create_tables(mydb, table_name, col_def):
    with mydb.cursor() as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"CREATE TABLE {table_name} ({', '.join(col_def)});") 
        mydb.commit()

def read_csv(csv_file):
    col_def = []
    
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        columns = next(reader, None)  
        sample_data = next(reader, [])  

        if not columns:
            raise ValueError("CSV file is empty or has no headers!")

        for col_name, sample_value in zip(columns, sample_data):
            inferred_type = infer_data_type(sample_value) if sample_value else "VARCHAR(255)"
            col_def.append(f"{col_name} {inferred_type}")
    
    return col_def

def csv_to_mysql(mydb, csv_file, table_name):
    column_definitions = read_csv(csv_file)
    create_tables(mydb, table_name, column_definitions)  

    with mydb.cursor() as cursor:
        cursor.execute(f"""
        LOAD DATA LOCAL INFILE '{csv_file}'
        INTO TABLE {table_name}
        FIELDS TERMINATED BY ',' 
        ENCLOSED BY '"'
        LINES TERMINATED BY '\n'
        IGNORE 1 ROWS;
        """)
        mydb.commit()

    print("Data imported successfully with filters applied")

def get_row_count(mydb, table_name):
    with mydb.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"Total rows in {table_name} table: {row_count}")
        return row_count

def infer_data_type(value):
    value = value.strip()
    if value.isdigit():
        return "INT"

    if re.fullmatch(r"-?\d+\.\d+", value):
        return "DECIMAL(10,2)"

    return "TEXT" if len(value) > 255 else "VARCHAR(255)"

def main():
    CSV_FILE = "######" #CSV File that has all the pokemon card data 
    csv_tablename = "PokemonCards"
    
    mydb = create_db_connection()
    if mydb.is_connected():
        print("Database connection established successfully ( ˶ˆᗜˆ˵ )")
        print(f"Creating table '{csv_tablename}'...")
        csv_to_mysql(mydb, CSV_FILE, csv_tablename)
        print(f"'{CSV_FILE}' successfully imported into MySQL")
        get_row_count(mydb, csv_tablename)
        
        mydb.close()
        print("Database connection closed (×_×)")
    else: 
        print("Failed to establish database connection (｡•́︿•̀｡)")
if __name__ == "__main__":
    main()
