import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, time

import main
from main_v2 import OptionsPositionModel



eastern = pytz.timezone("US/Eastern")

def is_market_open():
    now = datetime.now(eastern)
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    return time(9, 30) <= now.time() <= time(16, 0)

def market_task():
    if is_market_open():
        now = datetime.now(eastern)
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market is open – running task.")
        opm = OptionsPositionModel(send_message = True)
        opm.run()
    else:
        print("Skipped – outside market hours")



def start_scheduler():
    scheduler = BlockingScheduler(timezone=eastern)

    # Run every hour from 9 to 16 (4 PM), Monday to Friday
    trigger = CronTrigger(hour='6-16', minute='0', day_of_week='mon-fri')
    # trigger = CronTrigger(hour='1-23', minute='0')

    scheduler.add_job(market_task, trigger)
    print("Scheduler started (runs hourly during market hours)...")
    scheduler.start()

start_scheduler()