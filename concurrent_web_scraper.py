"""
Fast & Smart Concurrent Web Scraper

High-performance web scraper using asyncio/aiohttp for concurrent fetching
of hundreds of financial articles with intelligent rate limiting and error handling.
"""

import asyncio
import aiohttp
import asyncpg
import logging
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import hashlib
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    success: bool
    content: Optional[str] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    response_time_ms: float = 0.0
    error_message: Optional[str] = None
    redirect_url: Optional[str] = None
    
class RateLimiter:
    """Intelligent rate limiter for different domains"""
    
    def __init__(self):
        self.domain_delays: Dict[str, float] = {}
        self.last_request_time: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = {}
        self.failure_counts: Dict[str, int] = {}
        
        # Default delays per domain type
        self.default_delays = {
            'reuters.com': 1.0,
            'bloomberg.com': 2.0,
            'ft.com': 1.5,
            'wsj.com': 2.0,
            'marketwatch.com': 0.5,
            'sec.gov': 3.0,  # Government sites - be respectful
            'federalreserve.gov': 3.0,
            'default': 1.0
        }
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return 'unknown'
    
    def get_delay(self, url: str) -> float:
        """Get appropriate delay for domain"""
        domain = self.get_domain(url)
        
        # Check for specific domain rules
        for known_domain, delay in self.default_delays.items():
            if known_domain in domain:
                return self.domain_delays.get(domain, delay)
        
        return self.domain_delays.get(domain, self.default_delays['default'])
    
    async def wait_if_needed(self, url: str):
        """Wait if rate limiting is needed for this domain"""
        domain = self.get_domain(url)
        current_time = time.time()
        
        if domain in self.last_request_time:
            time_since_last = current_time - self.last_request_time[domain]
            required_delay = self.get_delay(url)
            
            # Add jitter to avoid thundering herd
            jitter = random.uniform(0.1, 0.3)
            required_delay += jitter
            
            if time_since_last < required_delay:
                wait_time = required_delay - time_since_last
                await asyncio.sleep(wait_time)
        
        self.last_request_time[domain] = time.time()
        self.request_counts[domain] = self.request_counts.get(domain, 0) + 1
    
    def record_failure(self, url: str):
        """Record a failure and potentially increase delay"""
        domain = self.get_domain(url)
        self.failure_counts[domain] = self.failure_counts.get(domain, 0) + 1
        
        # Increase delay for domains with failures
        if self.failure_counts[domain] > 3:
            current_delay = self.get_delay(url)
            self.domain_delays[domain] = min(current_delay * 1.5, 10.0)
    
    def record_success(self, url: str):
        """Record a success and potentially decrease delay"""
        domain = self.get_domain(url)
        
        # Decrease delay for successful domains
        if self.request_counts.get(domain, 0) > 10:
            current_delay = self.get_delay(url)
            self.domain_delays[domain] = max(current_delay * 0.9, 0.5)

class ConcurrentWebScraper:
    """High-performance concurrent web scraper"""
    
    def __init__(self, 
                 db_config: Dict[str, str],
                 max_concurrent: int = 50,
                 max_concurrent_per_domain: int = 5,
                 timeout: int = 30,
                 max_retries: int = 3):
        
        self.db_config = db_config
        self.max_concurrent = max_concurrent
        self.max_concurrent_per_domain = max_concurrent_per_domain
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Async components
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.global_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Worker management
        self.worker_id = f"scraper-{int(time.time())}"
        self.active_workers = 0
        self.should_stop = False
        
        # Performance tracking
        self.stats = {
            'jobs_processed': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'total_bytes_downloaded': 0,
            'avg_response_time': 0.0,
            'start_time': None
        }

    async def initialize(self):
        """Initialize scraper components"""
        try:
            # Create database connection pool
            self.db_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create HTTP session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent * 2,  # Total connection pool
                limit_per_host=self.max_concurrent_per_domain,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                ssl=False  # Disable SSL verification for speed (adjust as needed)
            )
            
            # Custom timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.timeout,
                connect=10,
                sock_read=20
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            self.stats['start_time'] = time.time()
            logger.info("✅ Concurrent Web Scraper initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize scraper: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        self.should_stop = True
        
        if self.session:
            await self.session.close()
        if self.db_pool:
            await self.db_pool.close()

    def get_domain_semaphore(self, url: str) -> asyncio.Semaphore:
        """Get or create semaphore for domain"""
        domain = self.rate_limiter.get_domain(url)
        
        if domain not in self.domain_semaphores:
            self.domain_semaphores[domain] = asyncio.Semaphore(
                self.max_concurrent_per_domain
            )
        
        return self.domain_semaphores[domain]

    async def fetch_url(self, url: str, max_retries: Optional[int] = None) -> ScrapingResult:
        """Fetch a single URL with retries and rate limiting"""
        retries = max_retries or self.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed(url)
                
                # Use domain and global semaphores
                domain_semaphore = self.get_domain_semaphore(url)
                
                async with self.global_semaphore:
                    async with domain_semaphore:
                        start_time = time.time()
                        
                        async with self.session.get(url) as response:
                            content = await response.text()
                            response_time = (time.time() - start_time) * 1000
                            
                            # Create result
                            result = ScrapingResult(
                                url=url,
                                success=True,
                                content=content,
                                status_code=response.status,
                                content_type=response.headers.get('Content-Type'),
                                content_length=len(content),
                                response_time_ms=response_time,
                                redirect_url=str(response.url) if str(response.url) != url else None
                            )
                            
                            # Update rate limiter
                            self.rate_limiter.record_success(url)
                            
                            # Update stats
                            self.stats['successful_scrapes'] += 1
                            self.stats['total_bytes_downloaded'] += len(content)
                            
                            return result
                            
            except asyncio.TimeoutError as e:
                last_error = f"Timeout after {self.timeout}s"
                self.rate_limiter.record_failure(url)
                
            except aiohttp.ClientError as e:
                last_error = f"Client error: {e}"
                self.rate_limiter.record_failure(url)
                
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                self.rate_limiter.record_failure(url)
            
            # Wait before retry with exponential backoff
            if attempt < retries:
                wait_time = (2 ** attempt) + random.uniform(0.1, 1.0)
                await asyncio.sleep(wait_time)
        
        # All retries failed
        self.stats['failed_scrapes'] += 1
        return ScrapingResult(
            url=url,
            success=False,
            error_message=last_error
        )

    async def get_pending_jobs(self, limit: int = 100) -> List[Dict]:
        """Get pending scraping jobs from database"""
        try:
            async with self.db_pool.acquire() as conn:
                jobs = await conn.fetch("""
                    SELECT job_id, url, source_name, priority
                    FROM get_next_scraping_jobs($1, $2)
                """, self.worker_id, limit)
                
                return [dict(job) for job in jobs]
                
        except Exception as e:
            logger.error(f"❌ Error getting pending jobs: {e}")
            return []

    async def update_job_status(self, job_id: str, status: str, 
                              result: Optional[ScrapingResult] = None):
        """Update job status in database"""
        try:
            async with self.db_pool.acquire() as conn:
                if status == 'completed' and result:
                    await conn.execute("""
                        UPDATE scraping_jobs 
                        SET status = $2,
                            completed_at = NOW(),
                            processing_time_ms = $3
                        WHERE id = $1
                    """, job_id, status, int(result.response_time_ms))
                    
                elif status == 'failed' and result:
                    await conn.execute("""
                        UPDATE scraping_jobs 
                        SET status = $2,
                            completed_at = NOW(),
                            error_message = $3,
                            last_error_at = NOW(),
                            retry_count = retry_count + 1
                        WHERE id = $1
                    """, job_id, status, result.error_message)
                    
                else:
                    await conn.execute("""
                        UPDATE scraping_jobs 
                        SET status = $2
                        WHERE id = $1
                    """, job_id, status)
                    
        except Exception as e:
            logger.error(f"❌ Error updating job status: {e}")

    async def store_scraped_content(self, job: Dict, result: ScrapingResult):
        """Store scraped content in database"""
        if not result.success or not result.content:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                # Create content hash for deduplication
                content_hash = hashlib.md5(result.content.encode()).hexdigest()
                
                await conn.execute("""
                    INSERT INTO articles 
                    (url, title, content, source_name, scraped_at, 
                     word_count, content_hash, processing_status)
                    VALUES ($1, $2, $3, $4, NOW(), $5, $6, 'pending')
                    ON CONFLICT (url) DO UPDATE SET
                        content = EXCLUDED.content,
                        scraped_at = EXCLUDED.scraped_at,
                        word_count = EXCLUDED.word_count,
                        processing_status = 'pending'
                """, 
                result.url,
                'Title extraction pending',  # Will be extracted later
                result.content,
                job['source_name'],
                len(result.content.split()),
                content_hash
                )
                
                logger.debug(f"💾 Stored content for {result.url}")
                
        except Exception as e:
            logger.error(f"❌ Error storing scraped content: {e}")

    async def process_single_job(self, job: Dict) -> bool:
        """Process a single scraping job"""
        try:
            # Fetch the URL
            result = await self.fetch_url(job['url'])
            
            if result.success:
                # Store the content
                await self.store_scraped_content(job, result)
                await self.update_job_status(job['job_id'], 'completed', result)
                
                logger.debug(f"✅ Scraped {job['url']} ({result.content_length} bytes)")
                return True
                
            else:
                await self.update_job_status(job['job_id'], 'failed', result)
                logger.warning(f"❌ Failed to scrape {job['url']}: {result.error_message}")
                return False
                
        except Exception as e:
            error_result = ScrapingResult(
                url=job['url'],
                success=False,
                error_message=str(e)
            )
            await self.update_job_status(job['job_id'], 'failed', error_result)
            logger.error(f"❌ Error processing job {job['job_id']}: {e}")
            return False

    async def worker_loop(self, batch_size: int = 50):
        """Main worker loop for processing scraping jobs"""
        logger.info(f"🚀 Starting scraper worker {self.worker_id}")
        
        while not self.should_stop:
            try:
                # Get batch of jobs
                jobs = await self.get_pending_jobs(batch_size)
                
                if not jobs:
                    # No jobs available, wait a bit
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"📋 Processing batch of {len(jobs)} scraping jobs")
                
                # Process jobs concurrently
                tasks = [self.process_single_job(job) for job in jobs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successes
                successes = sum(1 for r in results if r is True)
                self.stats['jobs_processed'] += len(jobs)
                
                logger.info(f"✅ Batch complete: {successes}/{len(jobs)} successful")
                
                # Log performance metrics
                await self._log_performance_metrics(len(jobs), successes)
                
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                await asyncio.sleep(10)  # Back off on errors

    async def _log_performance_metrics(self, jobs_processed: int, successful: int):
        """Log performance metrics to database"""
        try:
            runtime = time.time() - self.stats['start_time']
            
            async with self.db_pool.acquire() as conn:
                # Log batch metrics
                await conn.execute("""
                    SELECT log_scraper_metric($1, $2, $3, $4, $5)
                """, 
                'batch_success_rate', successful / jobs_processed if jobs_processed > 0 else 0,
                'ratio', 'scraper', f'{{"worker_id": "{self.worker_id}"}}')
                
                # Log throughput
                throughput = self.stats['jobs_processed'] / runtime if runtime > 0 else 0
                await conn.execute("""
                    SELECT log_scraper_metric($1, $2, $3, $4, $5)
                """, 
                'scraper_throughput', throughput, 'jobs_per_second', 'scraper',
                f'{{"worker_id": "{self.worker_id}"}}')
                
        except Exception as e:
            logger.error(f"❌ Error logging performance metrics: {e}")

    async def get_scraper_stats(self) -> Dict:
        """Get current scraper statistics"""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        throughput = self.stats['jobs_processed'] / runtime if runtime > 0 else 0
        
        return {
            'worker_id': self.worker_id,
            'runtime_seconds': runtime,
            'jobs_processed': self.stats['jobs_processed'],
            'successful_scrapes': self.stats['successful_scrapes'],
            'failed_scrapes': self.stats['failed_scrapes'],
            'success_rate': self.stats['successful_scrapes'] / max(self.stats['jobs_processed'], 1),
            'throughput_jobs_per_second': throughput,
            'total_bytes_downloaded': self.stats['total_bytes_downloaded'],
            'avg_bytes_per_job': self.stats['total_bytes_downloaded'] / max(self.stats['jobs_processed'], 1),
            'domain_delays': dict(self.rate_limiter.domain_delays),
            'active': not self.should_stop
        }

# Utility functions
async def setup_scraper(db_config: Dict[str, str], **kwargs) -> ConcurrentWebScraper:
    """Setup and initialize concurrent scraper"""
    scraper = ConcurrentWebScraper(db_config, **kwargs)
    await scraper.initialize()
    return scraper

# Demo and testing
async def demo_concurrent_scraper():
    """Demonstrate concurrent scraper functionality"""
    print("🚀 Concurrent Web Scraper Demo")
    print("=" * 40)
    
    # Database configuration (update with your settings)
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'financial_data_ingestion',
        'user': 'postgres',
        'password': 'your_password'
    }
    
    try:
        scraper = await setup_scraper(db_config, max_concurrent=10)
        
        # Test fetching a few URLs
        test_urls = [
            'https://httpbin.org/delay/1',
            'https://httpbin.org/delay/2',
            'https://httpbin.org/json'
        ]
        
        print("🔄 Testing concurrent URL fetching...")
        
        tasks = [scraper.fetch_url(url) for url in test_urls]
        results = await asyncio.gather(*tasks)
        
        print(f"✅ Results:")
        for url, result in zip(test_urls, results):
            status = "✅ Success" if result.success else "❌ Failed"
            print(f"   {url}: {status} ({result.response_time_ms:.1f}ms)")
        
        # Get stats
        stats = await scraper.get_scraper_stats()
        print(f"\n📊 Scraper Stats:")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Avg response time: {stats.get('avg_response_time', 0):.1f}ms")
        
        await scraper.cleanup()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_concurrent_scraper())