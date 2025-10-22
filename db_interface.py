import asyncio
import aiosqlite
from sympy import false
from git import Repo
import os
import pandas as pd
import aiohttp

from config import DATABASE, GAME_CATEGORY, HASH_PATH, HASH_REPOSITORY

http = "https://tcgcsv.com/tcgplayer/Categories.csv"

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
            print(f"x Execute failed: {e}")
            return None

    async def _fetch_csv(self, url: str, path: str) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    #TODO:csv files
                    return True
                return False

    async def _fetch_category(self, category: int = GAME_CATEGORY)->bool:
        """
        Download the CSV files from tcgcsv for a category.
        Use asyncio tasks to fetch in parallel.
        :return:
        """

        try:
            url = f"https://tcgcsv.com/tcgplayer/{category}/Groups.csv"
            path = f"./categories/group{category}.csv"
            await self._fetch_csv(url, path)

            group = [] # fill in from file
            tasks = []
            async with asyncio.TaskGroup() as tg:
                for cat in group:
                    tasks.append(tg.create_task(self._fetch_csv(cat, f"./category{category}/{group}.csv")))

            # Then load CSVs
            return True

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

    async def _initialize_hash_repository(self, path: str = HASH_PATH, repo: str = HASH_REPOSITORY)->bool:
        """
        Clone repository if it does not exist.
        :param path:
        :param repo:
        :return:
        """
        pass

    async def _fetch_hashes(self, path: str = HASH_PATH)->bool:
        """
        Pull from remote
        :param path:
        :return:
        """
        pass

    async def _load_hashes(self):
        pass