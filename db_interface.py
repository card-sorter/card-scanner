import asyncio
from typing import List

from PIL import Image
from hamming_matcher import HammingMatcher

import aiosqlite
from git import Repo, InvalidGitRepositoryError
import os
import aiohttp
import csv
from io import StringIO, BytesIO
import aiofiles
import json

from common import Card
from config import DATABASE, GAME_CATEGORY, HASH_PATH, HASH_REPOSITORY

class DBInterface:
    """
    Manages interactions with the SQLite database, including data synchronization
    from external sources and image-based card search using Hamming distance.
    """
    def __init__(
            self,
            database_info = DATABASE,
            hash_repository = HASH_REPOSITORY,
            hash_path = HASH_PATH,
            game_category = GAME_CATEGORY
    ):
        """
        Initialize the DBInterface.

        Args:
            database_info (dict): Configuration for the database, including the path.
            hash_repository (str): URL of the git repository containing image hashes.
            hash_path (str): Local path to clone/store the hash repository.
            game_category (int): The TCGPlayer category ID for the game (e.g., Pokemon).
        """
        self.db = None
        self.hamming_matcher = HammingMatcher()
        self.hash_path = hash_path
        self.hash_repository = hash_repository
        self.database_info = database_info
        self.game_category = game_category

    @property
    def connected(self):
        return bool(self.db)

    async def open(self, path: str|None = None)->bool:
        """
        Connect to the SQLite database.

        Args:
            path (str | None): Path to the database file. If None, uses the path from self.database_info.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if path is None:
            path = self.database_info["path"]
        try: 
            self.db = await aiosqlite.connect(path)
            print(f"Connected to database: {path}")
            return True
        
        except Exception as e: 
            print(f"Failed to connect to database: {e}")
            self.db = None
            return False

    async def close(self)->bool:
        """
        Close the database connection.

        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        try:    
            await self.db.close()
            self.db = None
            print("Database connection closed")
            return True
        
        except Exception as e:
            print(f"Failed to disconnect database: {e}")
            return False

    async def _execute(self, statement: str)-> aiosqlite.Cursor | None:
        """
        Execute a SQL statement and return the cursor.

        Args:
            statement (str): The SQL statement to execute.

        Returns:
            aiosqlite.Cursor | None: The cursor object if successful, None if failed.

        Raises:
            Exception: If the database is not connected.
        """
        if not self.connected: 
            raise Exception("Database not connected")

        
        try: 
            cursor = await self.db.execute(statement)
            await self.db.commit()
            return cursor 
        
        except Exception as e: 
            print(f"Execute failed: {e}")
            return None

    async def _initialize_table(self) -> bool:
        """
        Initialize the database schema.

        Creates the 'hashes', 'groups', and 'products' tables if they do not exist.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            statement = 'CREATE TABLE IF NOT EXISTS hashes (\nCard_ID INTEGER PRIMARY KEY'
            for i in range(1,15):
                statement = statement + f",\nBigint{i} BIGINT DEFAULT 0"
            statement = statement + ",\nFOREIGN KEY (Card_ID) REFERENCES products (productId)\n);"
            print(statement)
            await self._execute(statement)
            statement = '''
            CREATE TABLE IF NOT EXISTS groups (
                groupId INTEGER NOT NULL PRIMARY KEY,
                name TEXT,
                abbreviation TEXT,
                isSupplemental BOOLEAN,
                publishedOn DATETIME,
                modifiedOn DATETIME,
                categoryId INTEGER
                );
            '''
            print(statement)
            await self._execute(statement)
            statement = '''
            CREATE TABLE IF NOT EXISTS products (
                productId INTEGER PRIMARY KEY, 
                name TEXT, 
                cleanName TEXT, 
                imageUrl TEXT, 
                categoryId INTEGER, 
                groupId INTEGER, 
                url TEXT, 
                modifiedOn DATETIME, 
                imageCount INTEGER, 
                lowPrice FLOAT, 
                midPrice FLOAT, 
                highPrice FLOAT, 
                marketPrice FLOAT, 
                directLowPrice FLOAT, 
                subTypeName TEXT, 
                data TEXT,
                FOREIGN KEY (groupId) REFERENCES groups (groupId)
                );
            '''
            print(statement)
            await self._execute(statement)
            return True

        except Exception as e:
            print(f"Failed to initialize table: {e}")
            return False

    async def _download_csv(self, url: str, path: str) -> bool:
        """
        Download a CSV file from a URL and save it to a local path.

        Args:
            url (str): The URL to download from.
            path (str): The local file path to save the CSV.

        Returns:
            bool: True if download and save were successful, False otherwise.
        """
        try: 
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        #TODO:csv files
                        csv_text = await response.text()

                        os.makedirs(os.path.dirname(path), exist_ok=True) 

                        async with aiofiles.open(path, 'w', encoding='utf-8') as file:
                            await file.write(csv_text) #Raw data

                        #print(f"CSV successfully saved to: {path}")
                        return True
                    else: 
                        print(f"Failed to fetch CSV: HTTP {response.status}")
                        return False
                
        except Exception as e: 
            print(f"Error fetching CSV: {e}")
            return False
        
    async def _read_file(self, path: str, column_name: str) -> list:
        """
        Read a CSV file and extract values from a specific column.

        Args:
            path (str): Path to the CSV file.
            column_name (str): The name of the column to extract.

        Returns:
            list: A list of values from the specified column. Returns empty list on error.
        """
        try: 
            async with aiofiles.open(path, 'r', encoding='utf-8') as file: 
                content = await file.read()

            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)

            groups = [row[column_name] for row in reader if column_name in row and row[column_name]]
            return groups
        
        except Exception as e: 
            print(f"Error reading category file: {e}")
            return []

    async def _load_data(self, content: str, table_name: str) -> bool:
        """
        Load CSV content into a database table.

        Args:
            content (str): The raw CSV content string.
            table_name (str): The name of the table to insert data into.

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        try:
            lines = content.strip().split('\n')
            if len(lines) < 2:
                print("CSV file has no data rows")
                return False

            header = [col.strip() for col in lines[0].split(',')]
            print(f"CSV Header: {header}")

            data_to_insert = []
            for line in lines[1:]:
                if line.strip():
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(header):
                        data_to_insert.append(tuple(values))

            if data_to_insert:
                columns = ', '.join([f'"{col}"' for col in header])
                placeholders = ', '.join(['?'] * len(header))
                sql = f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"

                await self.db.executemany(sql, data_to_insert)
                await self.db.commit()
                #print(f"Loaded {len(data_to_insert)} rows into {table_name} table")
                return True
            else:
                print(f"No data to load")
                return False

        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    async def _fetch_group_ids(self, category: int | None = None) -> list[int]:
        """
        Fetch all group IDs for a given category from the database.

        Args:
            category (int | None): The category ID.

        Returns:
            list[int]: A list of group IDs.
        """
        statement = "SELECT groupId FROM groups WHERE categoryId = ?;"
        cur = await self.db.execute(statement, (category,))
        rows = await cur.fetchall()
        ret = [r[0] for r in rows]
        return ret


    async def _load_category(self, category: int | None = None)->bool:
        """
        Load category information from a local CSV file into the database.

        Args:
            category (int | None): The category ID. If None, uses self.game_category.

        Returns:
            bool: True if successful, False otherwise.
        """
        if category is None:
            category = self.game_category
        try:
            async with aiofiles.open(f"./categories/category{category}.csv", 'r', encoding='utf-8') as file:
                content = await file.read()

                if await self._load_data(content, "groups"):
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Failed to load products: {e}")
            return False

    async def _update_category(self, category: int | None = None) -> bool:
        """
        Download and update category and group data from TCGCSV.

        Downloads the Groups.csv for the category, loads it, then downloads
        ProductsAndPrices.csv for each group and loads them.

        Args:
            category (int | None): The category ID. If None, uses self.game_category.

        Returns:
            bool: True if all updates were successful, False otherwise.
        """
        if category is None:
            category = self.game_category
        try:
            url = f"https://tcgcsv.com/tcgplayer/{category}/Groups.csv"
            path = f"./categories/category{category}.csv"
            success = await self._download_csv(url, path)
            if not success:
                print(f"No Category CSV file downloaded")
                return False

            await self._load_category(category)

            groups = await self._fetch_group_ids(category)

            async with asyncio.TaskGroup() as tg:
                for group_id in groups:
                    group_url = f"https://tcgcsv.com/tcgplayer/{category}/{group_id}/ProductsAndPrices.csv"
                    group_path = f"./categories/category{category}/{group_id}.csv"
                    tg.create_task(self._download_csv(group_url, group_path))

            success = await self._load_groups(path)
            return success

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Network issue in category {category}: {e}")
            return False

        except OSError as e:
            print(f"File system error in category {category}: {e}")
            return False

        except Exception as e:
            print(f"Failed to fetch category {category}: {e}")
            return False

    async def _load_groups(self, path: str, category: int | None = None) -> bool:
        """
        Load product data from downloaded group CSV files into the database.

        Args:
            path (str): Path to the category CSV file (unused but kept for signature compatibility).
            category (int | None): The category ID. If None, uses self.game_category.

        Returns:
            bool: True if successful, False otherwise.
        """
        if category is None:
            category = self.game_category
        try:
            group_ids = await self._fetch_group_ids(category)
            if not group_ids:
                print("Failed to load groups")
                return False

            await self.db.execute("PRAGMA synchronous = OFF;") #Disables waiting for disk writes
            await self.db.execute("PRAGMA journal_mode = MEMORY;") # Keeps journal in memory, not disk
            await self.db.execute("BEGIN TRANSACTION;") #Wraps all inserts in one atomic operation
            card_columns = [
                "productId", "name", "cleanName", "imageUrl", "categoryId",
                "groupId", "url", "modifiedOn", "imageCount", "lowPrice",
                "midPrice", "highPrice", "marketPrice", "directLowPrice",
                "subTypeName"
            ]

            for group_id in group_ids:
                group_path = f"./categories/category{category}/{group_id}.csv"
                if not os.path.exists(group_path):
                    print(f"Group {group_id} does not exist")
                    continue

                async with aiofiles.open(group_path, 'r', encoding='utf-8') as file:
                    content = await file.read()

                csv_file = StringIO(content)
                reader = csv.DictReader(csv_file)

                insert_data = []
                for row in reader:
                    if not row.get("productId"):
                        continue

                    normal_values = [row.get(col, None) for col in card_columns]
                    ext_fields = {k: v for k, v in row.items() if k.startswith("ext")}
                    data_json = json.dumps(ext_fields, sort_keys=True, indent=4)
                    insert_data.append(tuple(normal_values + [data_json]))

                if insert_data:
                    columns = ", ".join(card_columns + ["data"])
                    placeholders = ", ".join(["?"] * len(card_columns + ["data"]))
                    sql = f"INSERT OR REPLACE INTO products ({columns}) VALUES ({placeholders})"
                    await self.db.executemany(sql, insert_data)
                    #print(f"Loaded {len(insert_data)} products from {group_id}.csv")
                else:
                    print(f"No valid data in {group_id}.csv")

            await self.db.commit()
            await self.db.execute("PRAGMA synchronous = FULL;") #Resets back to safe default
            await self.db.execute("PRAGMA journal_mode = DELETE;") #Returns to normal disk journaling
            return True

        except Exception as e:
            print(f"Failed to load card CSV files: {e}")
            return False



    async def _initialize_hash_repository(self)->bool:
        """
        Clone the image hash repository if it does not exist.

        Returns:
            bool: True if successful (cloned or already exists), False otherwise.
        """
        try: 
            if not os.path.exists(os.path.join(self.hash_path, ".git")):
                os.makedirs(os.path.dirname(self.hash_path), exist_ok=True)
                Repo.clone_from(self.hash_repository, self.hash_path)
                print(f"Successfully cloned repository to: {self.hash_path}")
                return True
            else:
                print(f"Repository already cloned to: {self.hash_path}")
                return True
        
        except Exception as e: 
            print(f"Failed to clone repository: {e}")
            return False

    async def _download_hashes(self)->bool:
        """
        Pull the latest changes from the hash repository.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            git_repo = Repo(self.hash_path)
            origin = git_repo.remote(name='origin')
            origin.fetch()
            current_branch = git_repo.active_branch.name
            local_commit = git_repo.head.commit
            remote_ref = git_repo.refs[f"origin/{current_branch}"]
            remote_commit = remote_ref.commit

            if local_commit != remote_commit:
                origin.pull()
                print("Repository successfully updated.")
            else: 
                print("Repository is already up to date.")
            return True

        except InvalidGitRepositoryError:
            print(f"Directory exists but is not a git repo: {self.hash_path}")
            return False

        except Exception as e:
            print(f"Failed fetching hashes: {e}")
            return False

    async def _load_hashes(self) -> bool:
        """
        Load image hashes from the CSV file in the repository into the database.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            async with aiofiles.open("./hashes/image_hashes.csv", 'r', encoding='utf-8') as file:
                content = await file.read()

            if not self.connected:
                raise Exception("Failed to connect to database")

            await self.db.execute("PRAGMA synchronous = OFF;")
            await self.db.execute("PRAGMA journal_mode = MEMORY;")
            await self.db.execute("BEGIN TRANSACTION;")

            if await self._load_data(content, "hashes"):
                await self.db.commit()
                await self.db.execute("PRAGMA synchronous = FULL;")
                await self.db.execute("PRAGMA journal_mode = DELETE;")
                return True
            else:
                return False

        except Exception as e:
            print(f"Failed to load hashes: {e}")
            return False

    async def _update_hashes(self) -> bool:
        if await self._download_hashes():
            if await self._load_hashes():
                await self.hamming_matcher.load_from_db(DATABASE["path"], "hashes")
                return True
        return False

    async def update(self) -> bool:
        """
        Update database with latest category and product data.
        """
        try:
            cat_ok = await self._update_category(self.game_category)
            hash_ok = await self._update_hashes()

            if cat_ok and hash_ok:
                print("Database updated successfully.")
                return True
            else:
                print("Some update steps failed.")
                return False

        except Exception as e:
            print(f"Failed to update: {e}")
            return False

    async def setup(self):
        try:
            if not self.connected:
                opened = await self.open()
                if not opened:
                    print("Failed to connect to database")
                    return False

            initialized = await self._initialize_table()
            if not initialized:
                print("Failed to initialize table")
                return False

            hash_repo = await self._initialize_hash_repository()
            if not hash_repo:
                print("Failed to initialize repository")
                return False

            updated = await self.update()
            if not updated:
                print("Failed to update table")
                return False

            print("Successfully setup")
            return True

        except Exception as e:
            print(f"Failed to setup: {e}")
            return False

    async def find_cards(self, compare: Card) -> List[Card]:
        """
        Find the top 3 closest matching cards using Hamming distance on image hashes.

        Args:
            compare (Card): The card object containing the hash to compare.

        Returns:
            List[Card]: A list of the top 3 matching Card objects, populated with details and images.

        Raises:
            Exception: If database is not connected.
            ValueError: If the compare card does not have hash_bigints populated.
        """
        if not self.connected:
            raise Exception("Failed to connect to database")

        # Build query hash from compare Card's hash_bigints
        if not compare.hash_bigints:
            raise ValueError("Card object must have hash_bigints populated")

        # Use the hash_bigints list directly
        query_hash = tuple(compare.hash_bigints)

        # If hash_bigints has fewer elements than available columns, pad with zeros
        if len(query_hash) < self.hamming_matcher.num_parts:
            query_hash = query_hash + (0,) * (self.hamming_matcher.num_parts - len(query_hash))
        elif len(query_hash) > self.hamming_matcher.num_parts:
            query_hash = query_hash[:self.hamming_matcher.num_parts]

        print(f"Querying with {len(query_hash)} hash parts out of {self.hamming_matcher.num_parts} available")

        # Find top 3 matches
        matches = self.hamming_matcher.find_top_3(query_hash)

        cards: List[Card] = []

        async with aiohttp.ClientSession() as session:
            for card_id, distance in matches:
                cursor = await self.db.execute(
                    "SELECT productId, name, imageUrl FROM products WHERE productId = ?",
                    (card_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()

                if not row:
                    print(f"Card ID {card_id} not found in products table")
                    continue

                product_id, name, image_url = row
                print(f"ID: {card_id}, Distance: {distance}, Name: {name}")

                image = None
                try:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            image_bytes = await resp.read()
                            image = Image.open(BytesIO(image_bytes))
                except Exception as e:
                    print(f"Failed to fetch image for card {product_id}: {e}")

                # Create Card object
                card = Card(image=image)
                card.values = {
                    "productId": product_id,
                    "name": name,
                    "imageUrl": image_url
                }
                card.distance = distance
                cards.append(card)

        return cards

    async def initialize(self):
        """
        Perform full initialization: connect, create tables, clone repo, and update data.

        Returns:
            bool: True if all steps succeed, False otherwise.
        """
        try:
            success = await self.setup()
            if not success:
                print("Database setup failed.")
            else:
                print("Database setup complete.")
            return success
        except Exception as e:
            print(f"Initialization error: {e}")
            return False



async def main():
    db = DBInterface()
    success = await db.initialize()
    if success:
        print("Done")
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())