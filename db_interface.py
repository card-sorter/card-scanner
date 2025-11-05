import asyncio
from typing import List

import aiosqlite
from git import Repo, Remote, InvalidGitRepositoryError
import os
import aiohttp
import csv
from io import StringIO
import aiofiles
import json

from common import Card
from config import DATABASE, GAME_CATEGORY, HASH_PATH, HASH_REPOSITORY


def hamming_distance(a0, a1, a2, a3, b0, b1, b2, b3):
    return (bin(a0 ^ b0).count('1') +
            bin(a1 ^ b1).count('1') +
            bin(a2 ^ b2).count('1') +
            bin(a3 ^ b3).count('1'))

class DBInterface:
    def __init__(self):
        self.db = None

    @property
    def connected(self):
        return bool(self.db)

    async def open(self, path: str = DATABASE["path"])->bool:
        """
        Connect to the database.
        :return:
        """
        try: 
            self.db = await aiosqlite.connect(path)
            await self.db.create_function('HAMMINGDISTANCE', 8, hamming_distance)

            print("Connected to database: {path}")
            return True
        
        except Exception as e: 
            print(f"Failed to connect to database: {e}")
            self.db = None
            return False

    async def close(self)->bool:
        """
        Close the database connection.
        :return:
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
        Return a cursor object of the executed statement.
        Return none if not connected.
        :param statement:
        :return:
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

    async def _initialize_table(self, category: int = GAME_CATEGORY) -> bool:
        """
        Initialize the database
        """
        try:
            statement = 'CREATE TABLE IF NOT EXISTS hashes (\ncardID INTEGER PRIMARY KEY'
            for i in range(1,15):
                statement = statement + f",\nbigint{i} BIGINT DEFAULT 0"
            statement = statement + "\n);"
            print(statement)
            await self._execute(statement)
            statement = '''
            CREATE TABLE IF NOT EXISTS cards (
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
                data TEXT
                );
            '''
            print(statement)
            await self._execute(statement)
            statement = '''
            CREATE TABLE IF NOT EXISTS products (
                groupId INTEGER,
                name TEXT,
                abbreviation TEXT,
                isSupplemental BOOLEAN,
                publishedOn DATETIME,
                modifiedOn DATETIME,
                categoryId INTEGER,
                FOREIGN KEY (categoryId) REFERENCES cards (categoryId)
                );
            '''
            print(statement)
            await self._execute(statement)
            return True

        except Exception as e:
            print(f"Failed to initialize table: {e}")
            return False

    async def _fetch_csv(self, url: str, path: str) -> bool:
        """
        Download the CSV files from tcgcsv Groups file
        and save it in the corresponding path
        :return:
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

                        print(f"CSV successfully saved to: {path}")
                        return True
                    else: 
                        print(f"Failed to fetch CSV: HTTP {response.status}")
                        return False
                
        except Exception as e: 
            print(f"Error fetching CSV: {e}")
            return False
        
    async def _read_file(self, path: str, column_name: str) -> list:
        try: 
            async with aiofiles.open(path, 'r', encoding='utf-8') as file: 
                content = await file.read()

            csv_file = StringIO(content)
            reader = csv.DictReader(csv_file)

            groups = [row[column_name] for row in reader if column_name in row and row[column_name]]
            return groups
        
        except Exception as e: 
            print(f"Error reading groups file: {e}")
            return []

    async def _load_data(self, content: str, table_name: str) -> bool:
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
                print(f"Loaded {len(data_to_insert)} rows into {table_name} table")
                return True
            else:
                print(f"No data to load")
                return False

        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    async def _fetch_products(self, category: int = GAME_CATEGORY)->bool:
        try:
            async with aiofiles.open(f"./categories/group{category}.csv", 'r', encoding='utf-8') as file:
                content = await file.read()

                if await self._load_data(content, "products"):
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Failed to load products: {e}")
            return False

    async def _fetch_category(self, category: int = GAME_CATEGORY) -> bool:
        """
        Download the CSV files from tcgcsv for a category.
        """
        try:
            url = f"https://tcgcsv.com/tcgplayer/{category}/Groups.csv"
            path = f"./categories/group{category}.csv"
            success = await self._fetch_csv(url, path)
            if not success:
                print(f"No Group CSV file fetched")
                return False

            group_ids = await self._read_file(path, column_name="groupId")

            async with asyncio.TaskGroup() as tg:
                for group_id in group_ids:
                    group_url = f"https://tcgcsv.com/tcgplayer/{category}/{group_id}/ProductsAndPrices.csv"
                    group_path = f"./categories/category{category}/{group_id}.csv"
                    tg.create_task(self._fetch_csv(group_url, group_path))

            success = await self._load_card_csv(path)
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

    async def _load_card_csv(self, path: str, category: int = GAME_CATEGORY) -> bool:
        """
        Load multiple group CSV files into the database
        """
        try:
            group_ids = await self._read_file(path, column_name="groupId")
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
                    sql = f"INSERT OR REPLACE INTO cards ({columns}) VALUES ({placeholders})"
                    await self.db.executemany(sql, insert_data)
                    print(f"Loaded {len(insert_data)} cards from {group_id}.csv")
                else:
                    print(f"No valid data in {group_id}.csv")

            await self.db.commit()
            await self.db.execute("PRAGMA synchronous = FULL;") #Resets back to safe default
            await self.db.execute("PRAGMA journal_mode = DELETE;") #Returns to normal disk journaling
            return True

        except Exception as e:
            print(f"Failed to load card CSV files: {e}")
            return False

    async def _initialize_hash_repository(self, path: str = HASH_PATH, repo: str = HASH_REPOSITORY)->bool:
        """
        Clone repository if it does not exist.
        :param path:
        :param repo:
        :return:
        """
        try: 
            if os.path.exists(path):
                try:
                    if await self._fetch_hashes(): 
                        print("Repository is ready to use")
                        return True
                    else: 
                        print("Failed to initialize repository")
                        return False
                    
                except InvalidGitRepositoryError:
                    print(f"Directory exists but is not a git repo: {path}")
                    return False
                
            else:        
                os.makedirs(os.path.dirname(path), exist_ok=True)
                Repo.clone_from(repo, path)
                print(f"Successfully cloned repository to: {path}")
                return True
        
        except Exception as e: 
            print(f"Failed to clone repository: {e}")
            return False

    async def _fetch_hashes(self, path: str = HASH_PATH)->bool:
        """
        Pull from remote
        :param path:
        :return:
        """
        try:
            git_repo = Repo(path)
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
            print(f"Directory exists but is not a git repo: {path}")
            return False

        except Exception as e:
            print(f"Failed fetching hashes: {e}")
            return False

    async def _load_hashes(self, path: str = HASH_PATH) -> bool:
        """
        Load hashes into the database table
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

    async def update(self)->bool:
        try:
            success = await self._fetch_category()
            if not success:
                print("Failed to fetch categories new data")
                return False

            success = await self._fetch_products()
            if not success:
                print("Failed to fetch products new data")
                return False

            print("Successfully updated")
            return True

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

            updated = await self.update()
            if not updated:
                print("Failed to update table")
                return False

            hash_repo = await self._initialize_hash_repository()
            if not hash_repo:
                print("Failed to initialize repository")
                return False

            hash_loaded = await self._load_hashes()
            if not hash_loaded:
                print("Failed to load hashes")
                return False

            print("Successfully setup")
            return True

        except Exception as e:
            print(f"Failed to setup: {e}")
            return False

    async def find_cards(self, compare: Card) -> List[Card]:
        '''CREATE FUNCTION HAMMINGDISTANCE(
          A0 BIGINT, A1 BIGINT, A2 BIGINT, A3 BIGINT,
          B0 BIGINT, B1 BIGINT, B2 BIGINT, B3 BIGINT
        )
        RETURNS INT DETERMINISTIC
        RETURN
          BIT_COUNT(A0 ^ B0) +
          BIT_COUNT(A1 ^ B1) +
          BIT_COUNT(A2 ^ B2) +
          BIT_COUNT(A3 ^ B3);'''
        pass

async def main():
    db = DBInterface()
    success = await db.setup()
    if success:
        print("Done")
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())