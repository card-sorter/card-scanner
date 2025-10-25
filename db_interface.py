import asyncio
import aiosqlite
from git import Repo
import os
import aiohttp
import csv
from io import StringIO
import aiofiles
import git

from config import DATABASE, GAME_CATEGORY, HASH_PATH, HASH_REPOSITORY

class DBInterface:
    def __init__(self):
        self.db = None

    @property
    def connected(self):
        return bool(self.db)

    async def open(self, path: str = DATABASE)->bool:
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
        
    #async def _initialize_db_table(self, statement: str):
     #   pass

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
                    group_path = f"./category{category}/{groupId}.csv"
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
                    
                except git.InvalidGitRepositoryError:
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

    async def _load_hashes(self):
        pass

async def main():
    result = DBInterface()
    await result._fetch_category()

asyncio.run(main())        