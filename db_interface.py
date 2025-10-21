import asyncio
import aiosqlite
from config import DATABASE

class DBInterface:
    def __init__(self):
        self.db = None