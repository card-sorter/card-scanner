import asyncio
from typing import List

import aiosqlite
from git import Repo, InvalidGitRepositoryError
import os
import aiohttp
import csv
from io import StringIO
import aiofiles
import json

from common import Card
from config import DATABASE, GAME_CATEGORY, HASH_PATH, HASH_REPOSITORY

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
            for i in range(1,5          ):
                statement = statement + f",\nbigint{i} BIGINT DEFAULT 0"
            statement = statement + "\n);"
            print(statement)
            await self._execute(statement)
            statement = '''
            CREATE TABLE IF NOT EXISTS cards (
                productId INTEGER PRIMARY KEY, 
                name TEXT, 
                cleanName TEXT, 
                imageURL TEXT, 
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
                '''
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

                        async with aiofiles.open(path, 'w', encoding = 'utf=8') as file:
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

    async def _fetch_category(self, category: int = GAME_CATEGORY)->bool:
        """
        Download the CSV files from tcgcsv for a category.
        Use asyncio tasks to fetch in parallel.
        :return:
        """

        try:
            url = f"https://tcgcsv.com/tcgplayer/{category}/Groups.csv"
            path = f"./categories/group{category}.csv"
            success = await self._fetch_csv(url, path)
            if not success: 
                print(f"No Group CSV file fetched")
                return False 
            
            group_ids = await self._read_file(path, column_name="groupId")
            os.makedirs(f"./category{category}", exist_ok=True)

            async with asyncio.TaskGroup() as tg:
                for groupId in group_ids:
                    group_url = f"https://tcgcsv.com/tcgplayer/{category}/{groupId}/ProductsAndPrices.csv"
                    group_path = f"./categories/category{category}/{groupId}.csv"
                    tg.create_task(self._fetch_csv(group_url, group_path))
            return True
        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e: 
            print(f"Network issue in category {category}: {e}")
            return False 
        
        except OSError as e:
            print(f"File system error in category {category}: {e}")
            return False
        
        except Exception as e: 
            print(f"Failed to fetch category {category}: {e}")
            return False

    async def _load_card_csv(self, path: str, category: int = GAME_CATEGORY)->bool:
        """
        Load a csv file into the table containing card information.
        Include game category as a column.
        Keep it non-blocking somehow.
        :param path:
        :return:
        """
        try:

            return True
        except Exception as e: 
            print(f"Failed to load category {category}: {e}")
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
            origin = git_repo.remotes.origin
            origin.fetch()
            current_branch = git_repo.active_branch.name
            local_commit = git_repo.head.commit
            remote_commit = git_repo.remotes.origin.refs[current_branch].commit

            if local_commit != remote_commit:
                origin.pull()
                print("Repository successfully updated.")
            else: 
                print("Repository is already up to date.")
            return True
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

            lines = content.strip().split('\n')
            if len(lines) < 2:
                print("CSV file has no data rows")
                return False

            header = [col.strip() for col in lines[0].split(',')]
            print(f"CSV Header: {header}")

            await self._initialize_table()

            data_to_insert = []
            for line in lines[1:]:
                if line.strip():
                    values = [v.strip() for v in line.split(',')]
                    if len(values) == len(header):
                        converted_values = [int(v) if v.isdigit() else 0 for v in values]
                        data_to_insert.append(tuple(converted_values))

            if data_to_insert:
                columns = ', '.join([f'"{col}"' for col in header])
                placeholders = ', '.join(['?'] * len(header))
                sql = f"INSERT OR REPLACE INTO hashes ({columns}) VALUES ({placeholders})"

                await self.db.executemany(sql, data_to_insert)
                await self.db.commit()
                print(f"Loaded {len(data_to_insert)} rows into hashes table")
            else:
                print("No data to insert")

            return True

        except Exception as e:
            print(f"Failed to load hashes: {e}")
            return False

    async def update(self):
        pass

    async def setup(self):
        await self.update()
        pass

    async def find_cards(self, compare: Card) -> List[Card]:
        pass

async def main():
    db = DBInterface()
    if await db.open():
        await db._initialize_table()
        success = await db._fetch_category()
        if success:
            print("Hashes loaded successfully!")
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())