import asyncio
import aiosqlite
from sympy import false
from git import Repo
import os

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
        pass

    async def close(self)->bool:
        """
        Close the database connection.
        :return:
        """
        pass

    async def _execute(self, statement: str)-> aiosqlite.Cursor | None:
        """
        Return a cursor object of the executed statement.
        Return none if not connected.
        :param statement:
        :return:
        """

    async def _fetch_category(self, category: int = GAME_CATEGORY)->bool:
        """
        Download the CSV files from tcgcsv for a category.
        Use asyncio tasks to fetch in parallel.
        :return:
        """
        pass

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