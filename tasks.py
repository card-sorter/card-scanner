from db_interface import DBInterface
from config import UPDATE_HOUR, UPDATE_MINUTE
import asyncio
from datetime import datetime, timedelta

async def daily_task(db: DBInterface = None):
    """
    Daily maintenance task to update the database.
    """
    print(f"Starting update task at {datetime.now()}")
    
    should_close = False
    if db is None:
        db = DBInterface()
        should_close = True
    
    try:
        # initialize() handles connection, table creation, repo cloning, and data updates
        success = await db.initialize()
        
        if success:
            print("Daily update completed successfully")
        else:
            print("Daily update failed")
        
    except Exception as e:
        print(f"Update failed with exception: {e}")
    
    finally:
        if should_close and db.connected:
            await db.close()

async def scheduler(db: DBInterface = None):
    """
    Schedules the daily task to run at the configured time.
    Waits without polling.
    """
    while True:
        now = datetime.now()
        target = now.replace(hour=UPDATE_HOUR, minute=UPDATE_MINUTE, second=0, microsecond=0)
        
        # If target time has passed for today, schedule for tomorrow
        if target <= now:
            target += timedelta(days=1)
            
        wait_seconds = (target - now).total_seconds()
        print(f"Scheduler sleeping for {wait_seconds:.2f} seconds until {target}")
        
        await asyncio.sleep(wait_seconds)
        
        # Run the task
        await daily_task(db)


def run_scheduled_tasks(db: DBInterface = None):
    asyncio.create_task(scheduler(db))