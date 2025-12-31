"""
Production Startup Script for Financial Seismograph System
"""

import asyncio
import logging
import sys
import signal
import sqlite3
from pathlib import Path
from datetime import datetime
import aiohttp

from config_manager import get_config
from production_api import app
from realtime_rss_manager import RealTimeRSSManager
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/financial_seismograph.log')
    ]
)
logger = logging.getLogger(__name__)

class ProductionManager:
    """Manages the complete production system"""
    
    def __init__(self):
        self.config = None
        self.api_server = None
        self.running = False
        
    async def initialize(self):
        """Initialize the production system"""
        logger.info("🚀 Initializing Financial Seismograph Production System")
        
        try:
            # Load and validate configuration
            self.config = get_config()
            logger.info("✅ Configuration loaded and validated")
            
            # Create logs directory if it doesn't exist
            Path("logs").mkdir(exist_ok=True)
            
            # Check database connectivity
            if not await self._check_database():
                logger.error("❌ Database connectivity check failed")
                return False
            
            # Check Ollama connectivity
            if not await self._check_ollama():
                logger.warning("⚠️ Ollama connectivity check failed - some features may be limited")
            
            logger.info("✅ Production system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize production system: {e}")
            return False
    
    async def _check_database(self):
        """Check database connectivity"""
        try:
            db_config = self.config.database_config
            
            if db_config.type == "sqlite":
                import sqlite3
                import os
                
                # Ensure directory exists
                db_dir = os.path.dirname(db_config.path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)
                
                # Test SQLite connection
                conn = sqlite3.connect(db_config.path)
                cursor = conn.cursor()
                
                # Create a simple test table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_check (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT
                    )
                """)
                
                # Test basic query
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                result = cursor.fetchone()[0]
                conn.close()
                
                logger.info(f"✅ SQLite database connected - {result} tables available")
                return True
                
            elif db_config.type == "postgresql":
                import asyncpg
                
                conn = await asyncpg.connect(
                    host=db_config.host,
                    port=db_config.port,
                    database=db_config.name,
                    user=db_config.user,
                    password=db_config.password
                )
                
                # Test basic query
                result = await conn.fetchval("SELECT COUNT(*) FROM rss_feeds")
                await conn.close()
                
                logger.info(f"✅ PostgreSQL connected - {result} RSS feeds configured")
                return True
            
            else:
                logger.error(f"❌ Unsupported database type: {db_config.type}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def _check_ollama(self):
        """Check Ollama connectivity"""
        try:
            import aiohttp
            
            ollama_config = self.config.ollama_config
            url = f"{ollama_config.base_url}/api/tags"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        logger.info(f"✅ Ollama connected - {len(models)} models available")
                        
                        # Check if required models are available
                        required_models = []
                        for model_type, model_config in self.config.model_configs.items():
                            required_models.append(model_config.name)
                        
                        missing_models = [model for model in required_models if model not in models]
                        if missing_models:
                            logger.warning(f"⚠️ Missing required models: {missing_models}")
                            logger.warning("   System will attempt to pull missing models automatically")
                        
                        return True
                    else:
                        logger.error(f"❌ Ollama returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            return False

    async def _start_rss_monitoring(self):
        """Start RSS feed monitoring for live data ingestion"""
        try:
            logger.info("📡 Starting RSS feed monitoring...")
            
            # Initialize RSS manager
            self.rss_manager = RealTimeRSSManager()
            
            # Add callback to store articles in database
            self.rss_manager.add_callback(self._store_rss_article)
            
            # Start monitoring
            asyncio.create_task(self.rss_manager.start_monitoring())
            
            logger.info("✅ RSS feed monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start RSS monitoring: {e}")
            raise

    async def _store_rss_article(self, entry):
        """Store RSS article in database and trigger web scraping"""
        try:
            conn = sqlite3.connect(self.config.database_config.path)
            cursor = conn.cursor()
            
            # Check if article already exists
            cursor.execute("SELECT id FROM articles WHERE url = ?", (entry.link,))
            existing = cursor.fetchone()
            if existing:
                return  # Article already exists
            
            # Get RSS feed ID from database
            cursor.execute("SELECT id FROM rss_feeds WHERE name = ?", (entry.feed_name,))
            rss_feed_result = cursor.fetchone()
            rss_feed_id = rss_feed_result[0] if rss_feed_result else None
            
            # Insert new article with initial RSS content
            cursor.execute("""
                INSERT INTO articles (title, content, url, rss_feed_id, published_at, word_count, is_scraped)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.title,
                entry.content or entry.summary,
                entry.link,
                rss_feed_id,
                entry.published_timestamp,
                len((entry.content or entry.summary).split()) if entry.content or entry.summary else 0,
                False  # Mark as not yet scraped
            ))
            
            article_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"📰 Stored article: {entry.title[:50]}...")
            
            # Trigger web scraping for full content
            asyncio.create_task(self._scrape_article_content(article_id, entry.link, entry.title))
            
        except Exception as e:
            logger.error(f"❌ Failed to store RSS article: {e}")

    async def _scrape_article_content(self, article_id: int, url: str, title: str):
        """Scrape full article content from URL"""
        try:
            # Import scraping libraries
            try:
                import trafilatura
                from readability import Document
                SCRAPING_AVAILABLE = True
            except ImportError:
                logger.warning("⚠️ Scraping libraries not available. Install with: pip install trafilatura readability-lxml")
                return
            
            # Create HTTP session if not exists
            if not hasattr(self, 'scraping_session'):
                timeout = aiohttp.ClientTimeout(total=30)
                self.scraping_session = aiohttp.ClientSession(timeout=timeout)
            
            # Fetch the webpage
            async with self.scraping_session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ Failed to fetch {url}: HTTP {response.status}")
                    return
                
                html_content = await response.text()
            
            # Extract text using trafilatura (primary method)
            extracted_text = trafilatura.extract(html_content, include_links=False, include_images=False)
            
            # Fallback to readability if trafilatura fails
            if not extracted_text or len(extracted_text) < 100:
                doc = Document(html_content)
                extracted_text = doc.summary()
                
                # Remove HTML tags
                import re
                extracted_text = re.sub(r'<[^>]+>', '', extracted_text)
            
            if extracted_text and len(extracted_text) > 100:
                # Update database with scraped content
                conn = sqlite3.connect(self.config.database_config.path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE articles 
                    SET content = ?, word_count = ?, is_scraped = ?, scraped_at = ?
                    WHERE id = ?
                """, (
                    extracted_text,
                    len(extracted_text.split()),
                    True,
                    datetime.now(),
                    article_id
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"🕷️ Scraped content for: {title[:50]}... ({len(extracted_text)} chars)")
            else:
                logger.warning(f"⚠️ Failed to extract meaningful content from {url}")
                
        except Exception as e:
            logger.error(f"❌ Failed to scrape article {url}: {e}")

    async def start_api_server(self):
        """Start the FastAPI server"""
        try:
            api_config = self.config.api_config
            
            logger.info(f"🌐 Starting API server on {api_config.host}:{api_config.port}")
            
            config = uvicorn.Config(
                app,
                host=api_config.host,
                port=api_config.port,
                log_level="info",
                reload=api_config.debug,
                access_log=True
            )
            
            server = uvicorn.Server(config)
            self.api_server = server
            
            await server.serve()
            
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            raise
    
    async def start_background_services(self):
        """Start background services"""
        try:
            # Import and start the integration
            from fast_smart_integration import FastSmartIntegration
            
            db_config = self.config.database_config
            
            if db_config.type == "sqlite":
                integration = FastSmartIntegration(
                    db_config={
                        'type': 'sqlite',
                        'path': db_config.path
                    }
                )
            elif db_config.type == "postgresql":
                integration = FastSmartIntegration(
                    db_config={
                        'type': 'postgresql',
                        'host': db_config.host,
                        'port': db_config.port,
                        'database': db_config.name,
                        'user': db_config.user,
                        'password': db_config.password
                    }
                )
            else:
                raise ValueError(f"Unsupported database type: {db_config.type}")
            
            await integration.initialize()
            
            # Start RSS feed monitoring
            await self._start_rss_monitoring()
            
            # Start continuous processing
            processing_settings = self.config.processing_settings
            asyncio.create_task(
                integration.start_continuous_processing(
                    poll_interval_seconds=30,
                    max_articles_per_batch=processing_settings['batch_size']
                )
            )
            
            logger.info("✅ Background services started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background services: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down Financial Seismograph System")
        
        self.running = False
        
        if self.api_server:
            self.api_server.should_exit = True
        
        # Give services time to clean up
        await asyncio.sleep(2)
        
        logger.info("✅ System shutdown complete")

# Global manager instance
manager = ProductionManager()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"📡 Received signal {signum}, initiating shutdown...")
    asyncio.create_task(manager.shutdown())

async def main():
    """Main entry point"""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize system
        if not await manager.initialize():
            logger.error("❌ System initialization failed")
            sys.exit(1)
        
        # Start background services
        await manager.start_background_services()
        
        # Start API server (this will block)
        manager.running = True
        await manager.start_api_server()
        
    except KeyboardInterrupt:
        logger.info("⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        await manager.shutdown()

def check_requirements():
    """Check if all requirements are met"""
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append("Python 3.8+ required")
    
    # Check required files
    required_files = [
        "config/models.yaml",
        "config/app.yaml", 
        "config/rss_feeds.yaml",
        "config_manager.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            errors.append(f"Missing required file: {file_path}")
    
    # Check required packages (using correct import names)
    required_packages = [
        'fastapi', 'uvicorn', 'asyncpg', 'aiohttp', 'pydantic', 
        'yaml', 'feedparser', 'scrapy', 'bs4'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            errors.append(f"Missing required package: {package}")
    
    if errors:
        logger.error("❌ Requirements check failed:")
        for error in errors:
            logger.error(f"   • {error}")
        return False
    
    logger.info("✅ All requirements met")
    return True

def print_startup_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                  Financial Seismograph System                ║
    ║                     Production Environment                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  🌊 Real-time Financial Sentiment Analysis                  ║
    ║  🤖 Multi-LLM AI Assembly Line                             ║
    ║  📊 Seismograph Tremor Detection                           ║
    ║  🚀 Fast & Smart Data Ingestion                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    🐍 Python: {sys.version.split()[0]}")
    print()

if __name__ == "__main__":
    print_startup_banner()
    
    if not check_requirements():
        sys.exit(1)
    
    logger.info("🎬 Starting Financial Seismograph Production System")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)