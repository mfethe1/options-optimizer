"""
Background Scheduler - Routine data refresh and maintenance
"""
from typing import Dict, List, Any
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import os

logger = logging.getLogger(__name__)


class DataScheduler:
    """
    Background scheduler for routine data updates.
    Runs earnings calendar updates, research refreshes, and cache maintenance.
    """
    
    def __init__(self, research_service=None, earnings_service=None, position_manager=None):
        self.scheduler = BackgroundScheduler()
        self.research_service = research_service
        self.earnings_service = earnings_service
        self.position_manager = position_manager
        self.is_running = False
        
        logger.info("DataScheduler initialized")
    
    def start(self):
        """Start the scheduler with all jobs"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        # Earnings calendar: Daily at 6 AM
        self.scheduler.add_job(
            self.update_earnings_calendar,
            CronTrigger(hour=6, minute=0),
            id='earnings_calendar_daily',
            name='Update earnings calendar',
            replace_existing=True
        )
        
        # Research updates: Hourly during market hours (9 AM - 4 PM ET)
        self.scheduler.add_job(
            self.update_research_market_hours,
            CronTrigger(hour='9-16', minute=0),
            id='research_market_hours',
            name='Update research (market hours)',
            replace_existing=True
        )
        
        # Research updates: Every 4 hours outside market hours
        self.scheduler.add_job(
            self.update_research_off_hours,
            IntervalTrigger(hours=4),
            id='research_off_hours',
            name='Update research (off hours)',
            replace_existing=True
        )
        
        # Cache cleanup: Daily at midnight
        self.scheduler.add_job(
            self.cleanup_old_caches,
            CronTrigger(hour=0, minute=0),
            id='cache_cleanup',
            name='Clean up old caches',
            replace_existing=True
        )
        
        # Position analysis: Every 15 minutes during market hours
        self.scheduler.add_job(
            self.analyze_positions,
            CronTrigger(hour='9-16', minute='*/15'),
            id='position_analysis',
            name='Analyze positions',
            replace_existing=True
        )
        
        self.scheduler.start()
        self.is_running = True
        logger.info("Scheduler started with all jobs")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        self.scheduler.shutdown()
        self.is_running = False
        logger.info("Scheduler stopped")
    
    def update_earnings_calendar(self):
        """Update earnings calendar for next 60 days"""
        if not self.earnings_service:
            logger.warning("Earnings service not available")
            return
        
        try:
            logger.info("Updating earnings calendar...")
            from datetime import timedelta
            
            start_date = datetime.now().strftime('%Y-%m-%d')
            end_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
            
            earnings = self.earnings_service.get_earnings_calendar(start_date, end_date)
            
            # Save to parquet
            if earnings:
                filename = f"calendar_{datetime.now().strftime('%Y%m%d')}.parquet"
                self.earnings_service.save_to_parquet(earnings, filename)
                logger.info(f"Updated earnings calendar: {len(earnings)} events")
            else:
                logger.warning("No earnings data fetched")
        except Exception as e:
            logger.error(f"Error updating earnings calendar: {e}")
    
    def update_research_market_hours(self):
        """Update research for all positions during market hours"""
        self._update_research(max_age_hours=1)
    
    def update_research_off_hours(self):
        """Update research for all positions outside market hours"""
        self._update_research(max_age_hours=4)
    
    def _update_research(self, max_age_hours: int):
        """Internal method to update research"""
        if not self.research_service or not self.position_manager:
            logger.warning("Research service or position manager not available")
            return
        
        try:
            logger.info(f"Updating research (max age: {max_age_hours}h)...")
            
            # Get unique symbols from positions
            symbols = self.position_manager.get_unique_symbols()
            
            if not symbols:
                logger.info("No positions to research")
                return
            
            # Update research for each symbol
            updated = 0
            for symbol in symbols:
                try:
                    self.research_service.research_symbol(symbol, max_age_hours=max_age_hours)
                    updated += 1
                except Exception as e:
                    logger.error(f"Error researching {symbol}: {e}")
            
            logger.info(f"Updated research for {updated}/{len(symbols)} symbols")
        except Exception as e:
            logger.error(f"Error updating research: {e}")
    
    def cleanup_old_caches(self):
        """Clean up cache files older than 30 days"""
        try:
            logger.info("Cleaning up old caches...")
            from pathlib import Path
            from datetime import timedelta
            
            cache_dirs = [
                Path("data/research"),
                Path("data/research/earnings")
            ]
            
            cutoff_date = datetime.now() - timedelta(days=30)
            deleted = 0
            
            for cache_dir in cache_dirs:
                if not cache_dir.exists():
                    continue
                
                for file in cache_dir.glob("*.json"):
                    file_time = datetime.fromtimestamp(file.stat().st_mtime)
                    if file_time < cutoff_date:
                        file.unlink()
                        deleted += 1
            
            logger.info(f"Deleted {deleted} old cache files")
        except Exception as e:
            logger.error(f"Error cleaning up caches: {e}")
    
    def analyze_positions(self):
        """Analyze all positions for alerts and risks"""
        if not self.position_manager or not self.earnings_service:
            logger.warning("Position manager or earnings service not available")
            return
        
        try:
            logger.info("Analyzing positions...")
            
            # Get all positions
            stocks = self.position_manager.get_all_stock_positions()
            options = self.position_manager.get_all_option_positions()
            
            alerts = []
            
            # Check options expiring soon
            for option in options:
                days_left = option.days_to_expiry()
                if days_left <= 7 and days_left >= 0:
                    alerts.append({
                        'type': 'expiration',
                        'symbol': option.symbol,
                        'message': f"Option expires in {days_left} days",
                        'severity': 'high' if days_left <= 3 else 'medium'
                    })
            
            # Check earnings risk
            symbols = self.position_manager.get_unique_symbols()
            for symbol in symbols:
                next_earnings = self.earnings_service.get_next_earnings(symbol)
                if next_earnings:
                    earnings_date = datetime.strptime(next_earnings['date'], '%Y-%m-%d').date()
                    days_to_earnings = (earnings_date - datetime.now().date()).days
                    
                    if days_to_earnings <= 7 and days_to_earnings >= 0:
                        alerts.append({
                            'type': 'earnings',
                            'symbol': symbol,
                            'message': f"Earnings in {days_to_earnings} days",
                            'severity': 'high' if days_to_earnings <= 3 else 'medium',
                            'date': next_earnings['date']
                        })
            
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts")
                # TODO: Store alerts or send notifications
            else:
                logger.info("No alerts generated")
        except Exception as e:
            logger.error(f"Error analyzing positions: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            })
        
        return {
            'running': self.is_running,
            'jobs': jobs,
            'timestamp': datetime.now().isoformat()
        }

