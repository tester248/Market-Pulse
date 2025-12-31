"""
Fast & Smart RSS Trigger System

High-performance RSS feed monitoring that triggers targeted scraping jobs
for Tier-1 financial sources with sub-minute latency detection.
"""

import asyncio
import aiohttp
import asyncpg
import feedparser
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

# Import configuration management
from config_manager import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RSSFeedEntry:
    """Represents a single RSS feed entry"""
    title: str
    url: str
    published_date: Optional[datetime]
    description: str
    author: Optional[str]
    guid: Optional[str]
    source_feed: str

@dataclass
class ScrapingJob:
    """Represents a scraping job to be queued"""
    url: str
    source_name: str
    rss_feed_id: str
    priority: int = 3
    max_retries: int = 3

class RSSFeedMonitor:
    """High-performance RSS feed monitoring system"""
    
    def __init__(self, 
                 db_config: Dict[str, str],
                 max_concurrent_feeds: int = 20,
                 default_timeout: int = 30,
                 user_agent: str = "FinanceAI-FastScraper/1.0"):
        
        self.db_config = db_config
        self.max_concurrent_feeds = max_concurrent_feeds
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # Rate limiting and performance
        self.semaphore = asyncio.Semaphore(max_concurrent_feeds)
        self.session_timeout = aiohttp.ClientTimeout(total=default_timeout)
        self.user_agent = user_agent
        
        # Tracking
        self.seen_guids: Set[str] = set()
        self.monitoring_active = False
        
        # Performance metrics
        self.metrics = {
            'feeds_checked': 0,
            'articles_found': 0,
            'jobs_created': 0,
            'errors': 0,
            'avg_response_time': 0.0
        }

    async def initialize(self):
        """Initialize database connection and HTTP session"""
        try:
            # Create database connection pool
            self.db_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
            
            # Create HTTP session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool
                limit_per_host=10,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.session_timeout,
                headers={
                    'User-Agent': self.user_agent,
                    'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                    'Accept-Encoding': 'gzip, deflate',
                    'Cache-Control': 'no-cache'
                }
            )
            
            # Load existing GUIDs to prevent reprocessing
            await self._load_existing_guids()
            
            logger.info("✅ RSS Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RSS Monitor: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.db_pool:
            await self.db_pool.close()

    async def _load_existing_guids(self):
        """Load existing article GUIDs to prevent reprocessing"""
        try:
            async with self.db_pool.acquire() as conn:
                # Load GUIDs from last 7 days to balance memory vs accuracy
                cutoff_date = datetime.now() - timedelta(days=7)
                
                rows = await conn.fetch("""
                    SELECT DISTINCT url FROM articles 
                    WHERE scraped_at > $1
                """, cutoff_date)
                
                self.seen_guids = {row['url'] for row in rows}
                logger.info(f"📚 Loaded {len(self.seen_guids)} existing article URLs")
                
        except Exception as e:
            logger.error(f"❌ Error loading existing GUIDs: {e}")
            self.seen_guids = set()

    async def get_active_feeds(self) -> List[Dict]:
        """Get list of active RSS feeds from database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, name, url, category, priority, check_interval_seconds,
                           last_checked, consecutive_failures
                    FROM rss_feeds 
                    WHERE enabled = true
                      AND (consecutive_failures < 5 OR consecutive_failures IS NULL)
                    ORDER BY priority ASC, last_checked ASC NULLS FIRST
                """
                
                rows = await conn.fetch(query)
                feeds = [dict(row) for row in rows]
                
                logger.info(f"📡 Found {len(feeds)} active RSS feeds")
                return feeds
                
        except Exception as e:
            logger.error(f"❌ Error getting active feeds: {e}")
            return []

    async def should_check_feed(self, feed: Dict) -> bool:
        """Determine if feed should be checked based on interval and last check"""
        if not feed['last_checked']:
            return True
            
        interval = timedelta(seconds=feed['check_interval_seconds'])
        next_check = feed['last_checked'] + interval
        
        return datetime.now() > next_check

    async def fetch_rss_feed(self, feed: Dict) -> List[RSSFeedEntry]:
        """Fetch and parse a single RSS feed with performance tracking"""
        start_time = asyncio.get_event_loop().time()
        
        async with self.semaphore:
            try:
                logger.debug(f"🔄 Fetching RSS feed: {feed['name']}")
                
                async with self.session.get(feed['url']) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse RSS content
                        entries = await self._parse_rss_content(content, feed)
                        
                        # Update performance metrics
                        response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                        await self._update_feed_metrics(feed['id'], True, response_time, len(entries))
                        
                        logger.info(f"✅ {feed['name']}: {len(entries)} entries found")
                        return entries
                        
                    else:
                        logger.warning(f"⚠️ {feed['name']}: HTTP {response.status}")
                        await self._update_feed_metrics(feed['id'], False, 0, 0)
                        return []
                        
            except asyncio.TimeoutError:
                logger.error(f"⏰ {feed['name']}: Request timeout")
                await self._update_feed_metrics(feed['id'], False, 0, 0)
                return []
                
            except Exception as e:
                logger.error(f"❌ {feed['name']}: {e}")
                await self._update_feed_metrics(feed['id'], False, 0, 0)
                return []

    async def _parse_rss_content(self, content: str, feed: Dict) -> List[RSSFeedEntry]:
        """Parse RSS/Atom content and extract entries"""
        try:
            # Use feedparser for robust RSS/Atom parsing
            parsed_feed = feedparser.parse(content)
            entries = []
            
            for entry in parsed_feed.entries:
                # Extract publication date
                pub_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                    except:
                        pass
                elif hasattr(entry, 'published'):
                    try:
                        pub_date = parsedate_to_datetime(entry.published)
                    except:
                        pass
                
                # Create RSS entry
                rss_entry = RSSFeedEntry(
                    title=entry.get('title', ''),
                    url=entry.get('link', ''),
                    published_date=pub_date,
                    description=entry.get('summary', entry.get('description', '')),
                    author=entry.get('author', ''),
                    guid=entry.get('id', entry.get('link', '')),
                    source_feed=feed['name']
                )
                
                # Only include if URL is valid and not seen before
                if rss_entry.url and rss_entry.url not in self.seen_guids:
                    entries.append(rss_entry)
                    self.seen_guids.add(rss_entry.url)
            
            return entries
            
        except Exception as e:
            logger.error(f"❌ Error parsing RSS content: {e}")
            return []

    async def _update_feed_metrics(self, feed_id: str, success: bool, 
                                 response_time_ms: float, articles_found: int):
        """Update feed performance metrics in database"""
        try:
            async with self.db_pool.acquire() as conn:
                if success:
                    await conn.execute("""
                        UPDATE rss_feeds 
                        SET last_checked = NOW(),
                            last_successful_check = NOW(),
                            consecutive_failures = 0,
                            total_articles_found = total_articles_found + $2,
                            avg_response_time_ms = COALESCE(
                                (avg_response_time_ms * 0.8) + ($3 * 0.2), 
                                $3
                            )
                        WHERE id = $1
                    """, feed_id, articles_found, response_time_ms)
                    
                    if articles_found > 0:
                        await conn.execute("""
                            UPDATE rss_feeds 
                            SET last_article_found = NOW()
                            WHERE id = $1
                        """, feed_id)
                else:
                    await conn.execute("""
                        UPDATE rss_feeds 
                        SET last_checked = NOW(),
                            consecutive_failures = consecutive_failures + 1
                        WHERE id = $1
                    """, feed_id)
                    
        except Exception as e:
            logger.error(f"❌ Error updating feed metrics: {e}")

    async def create_scraping_jobs(self, entries: List[RSSFeedEntry], feed: Dict) -> int:
        """Create scraping jobs for new articles"""
        if not entries:
            return 0
            
        try:
            async with self.db_pool.acquire() as conn:
                jobs_created = 0
                
                for entry in entries:
                    try:
                        # Determine priority based on feed priority and content
                        priority = self._calculate_job_priority(entry, feed)
                        
                        # Insert scraping job
                        await conn.execute("""
                            INSERT INTO scraping_jobs 
                            (url, source_name, rss_feed_id, priority, max_retries)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (url) DO NOTHING
                        """, entry.url, feed['name'], feed['id'], priority, 3)
                        
                        jobs_created += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Error creating job for {entry.url}: {e}")
                        continue
                
                logger.info(f"📋 Created {jobs_created} scraping jobs for {feed['name']}")
                return jobs_created
                
        except Exception as e:
            logger.error(f"❌ Error creating scraping jobs: {e}")
            return 0

    def _calculate_job_priority(self, entry: RSSFeedEntry, feed: Dict) -> int:
        """Calculate scraping job priority based on content and source"""
        base_priority = feed.get('priority', 3)
        
        # Boost priority for urgent keywords
        urgent_keywords = [
            'breaking', 'urgent', 'alert', 'emergency', 'crisis',
            'earnings', 'merger', 'acquisition', 'ipo', 'bankruptcy',
            'fed', 'federal reserve', 'interest rate', 'inflation'
        ]
        
        title_lower = entry.title.lower()
        description_lower = entry.description.lower()
        
        # Check for urgent content
        if any(keyword in title_lower or keyword in description_lower 
               for keyword in urgent_keywords):
            base_priority = max(1, base_priority - 1)
        
        # Boost Tier-1 sources
        if feed.get('category') == 'tier1_news':
            base_priority = max(1, base_priority - 1)
        
        # Recent articles get higher priority
        if entry.published_date:
            age_hours = (datetime.now() - entry.published_date.replace(tzinfo=None)).total_seconds() / 3600
            if age_hours < 1:
                base_priority = max(1, base_priority - 1)
        
        return min(base_priority, 5)  # Cap at priority 5

    async def monitor_feeds_once(self) -> Dict[str, int]:
        """Single iteration of feed monitoring"""
        feeds = await self.get_active_feeds()
        
        if not feeds:
            logger.warning("⚠️ No active feeds found")
            return {'feeds_checked': 0, 'jobs_created': 0}
        
        # Filter feeds that need checking
        feeds_to_check = [feed for feed in feeds if await self.should_check_feed(feed)]
        
        if not feeds_to_check:
            logger.debug("📡 No feeds need checking at this time")
            return {'feeds_checked': 0, 'jobs_created': 0}
        
        logger.info(f"🔄 Checking {len(feeds_to_check)} RSS feeds")
        
        # Fetch all feeds concurrently
        feed_tasks = [self.fetch_rss_feed(feed) for feed in feeds_to_check]
        feed_results = await asyncio.gather(*feed_tasks, return_exceptions=True)
        
        # Process results and create jobs
        total_jobs_created = 0
        successful_checks = 0
        
        for feed, result in zip(feeds_to_check, feed_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Feed {feed['name']} failed: {result}")
                continue
                
            if isinstance(result, list) and result:
                jobs_created = await self.create_scraping_jobs(result, feed)
                total_jobs_created += jobs_created
                successful_checks += 1
            else:
                successful_checks += 1
        
        # Update global metrics
        self.metrics['feeds_checked'] += successful_checks
        self.metrics['jobs_created'] += total_jobs_created
        
        return {
            'feeds_checked': successful_checks,
            'jobs_created': total_jobs_created,
            'total_articles': sum(len(r) for r in feed_results if isinstance(r, list))
        }

    async def start_monitoring(self, check_interval: int = 60):
        """Start continuous RSS feed monitoring"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring already active")
            return
            
        self.monitoring_active = True
        logger.info(f"🚀 Starting RSS feed monitoring (check every {check_interval}s)")
        
        try:
            while self.monitoring_active:
                start_time = asyncio.get_event_loop().time()
                
                # Monitor feeds
                results = await self.monitor_feeds_once()
                
                # Log results
                elapsed_time = asyncio.get_event_loop().time() - start_time
                logger.info(
                    f"📊 Monitoring cycle: {results['feeds_checked']} feeds checked, "
                    f"{results['jobs_created']} jobs created in {elapsed_time:.2f}s"
                )
                
                # Log performance metrics to database
                await self._log_performance_metrics(results, elapsed_time)
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.info("⏹️ RSS monitoring cancelled")
        except Exception as e:
            logger.error(f"❌ RSS monitoring error: {e}")
        finally:
            self.monitoring_active = False

    async def stop_monitoring(self):
        """Stop RSS feed monitoring"""
        logger.info("⏹️ Stopping RSS feed monitoring")
        self.monitoring_active = False

    async def _log_performance_metrics(self, results: Dict, elapsed_time: float):
        """Log performance metrics to database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    SELECT log_scraper_metric($1, $2, $3, $4, $5)
                """, 
                'feeds_checked_per_cycle', results['feeds_checked'], 'count', 'rss_monitor',
                '{"cycle_time": ' + str(elapsed_time) + '}')
                
                await conn.execute("""
                    SELECT log_scraper_metric($1, $2, $3, $4, $5)
                """, 
                'jobs_created_per_cycle', results['jobs_created'], 'count', 'rss_monitor',
                '{"cycle_time": ' + str(elapsed_time) + '}')
                
                await conn.execute("""
                    SELECT log_scraper_metric($1, $2, $3, $4, $5)
                """, 
                'monitor_cycle_time', elapsed_time, 'seconds', 'rss_monitor',
                '{"feeds_checked": ' + str(results['feeds_checked']) + '}')
                
        except Exception as e:
            logger.error(f"❌ Error logging performance metrics: {e}")

    async def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent stats
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_feeds,
                        COUNT(*) FILTER (WHERE enabled = true) as enabled_feeds,
                        COUNT(*) FILTER (WHERE consecutive_failures >= 5) as failed_feeds,
                        AVG(avg_response_time_ms) as avg_response_time,
                        SUM(total_articles_found) as total_articles_found
                    FROM rss_feeds
                """)
                
                # Get recent job stats
                job_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) FILTER (WHERE status = 'queued') as queued_jobs,
                        COUNT(*) FILTER (WHERE status = 'processing') as processing_jobs,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed_jobs,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_jobs
                    FROM scraping_jobs
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                
                return {
                    'feeds': dict(stats) if stats else {},
                    'jobs': dict(job_stats) if job_stats else {},
                    'monitoring_active': self.monitoring_active,
                    'runtime_metrics': self.metrics
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting monitoring stats: {e}")
            return {}

# Utility functions
async def setup_rss_monitor(db_config: Dict[str, str]) -> RSSFeedMonitor:
    """Setup and initialize RSS monitor"""
    monitor = RSSFeedMonitor(db_config)
    await monitor.initialize()
    return monitor

# Demo and testing
async def demo_rss_monitor():
    """Demonstrate RSS monitor functionality"""
    print("📡 RSS Feed Monitor Demo")
    print("=" * 40)
    
    # Use configuration management for database settings
    config = get_config()
    db_config_obj = config.database_config
    db_config = {
        'host': db_config_obj.host,
        'port': db_config_obj.port,
        'database': db_config_obj.name,
        'user': db_config_obj.user,
        'password': db_config_obj.password
    }
    
    try:
        monitor = await setup_rss_monitor(db_config)
        
        # Test single monitoring cycle
        print("🔄 Running single monitoring cycle...")
        results = await monitor.monitor_feeds_once()
        
        print(f"✅ Results:")
        print(f"   Feeds checked: {results['feeds_checked']}")
        print(f"   Jobs created: {results['jobs_created']}")
        print(f"   Articles found: {results.get('total_articles', 0)}")
        
        # Get stats
        stats = await monitor.get_monitoring_stats()
        print(f"\n📊 System Stats:")
        print(f"   Total feeds: {stats.get('feeds', {}).get('total_feeds', 0)}")
        print(f"   Enabled feeds: {stats.get('feeds', {}).get('enabled_feeds', 0)}")
        print(f"   Queued jobs: {stats.get('jobs', {}).get('queued_jobs', 0)}")
        
        await monitor.cleanup()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_rss_monitor())