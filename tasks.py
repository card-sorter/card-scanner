from db_interface import DBInterface
from config import DATABASE, UPDATE_HOUR, UPDATE_MINUTE
import asyncio
import aioschedule as schedule
from datetime import datetime

async def daily_task(db: DBInterface = None): #daily maintainence/ updates
    print(f"Starting update task at {datetime.now()}")
    
    if db is None:
        db = DBInterface()
        await db.open(DATABASE)
    
    try:
        print("Fetching latest card data...") # Update card data from TCGCSV
        await db._update_category()
        
        print("Updating hash repository...") # Update hash repository
        await db._download_hashes()
        await db._load_hashes()
        
        print("Daily update completed successfully")
        
    except Exception as e:
        print(f"Update failed: {e}")
    
    finally:
        if db.connected:
            await db.close()

async def scheduler(): # Schedule daily update
    schedule.every().day.at(f"{UPDATE_HOUR:02d}:{UPDATE_MINUTE:02d}").do(
        lambda: asyncio.create_task(daily_task())
    )
    
    while True:
        await schedule.run_pending()
        await asyncio.sleep(60)  # Check every minute

async def run_scheduled_tasks():
    await scheduler()