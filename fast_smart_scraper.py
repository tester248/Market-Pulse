"""
Fast & Smart Data Ingestion Layer - Main Orchestrator

Complete "Fast & Smart" scraper system that coordinates RSS monitoring, 
concurrent web scraping, smart text extraction, and Finance-LLM-13B analysis
for real-time financial insights.
"""

import asyncio
import logging
import signal
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse

# Check and install dependencies
def check_dependencies():
    """Check and provide guidance for missing dependencies"""
    missing_deps = []
    optional_deps = []
    
    try:
        import asyncpg
    except ImportError:
        missing_deps.append("asyncpg")
    
    try:
        import aiohttp
    except ImportError:
        missing_deps.append("aiohttp")
    
    try:
        import feedparser
    except ImportError:
        missing_deps.append("feedparser")
    
    try:
        import trafilatura
    except ImportError:
        optional_deps.append("trafilatura")
    
    try:
        from readability import Document
    except ImportError:
        optional_deps.append("readability-lxml")
    
    try:
        from newspaper import Article
    except ImportError:
        optional_deps.append("newspaper3k")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        print()
        return False
    
    if optional_deps:
        print("⚠️ Optional dependencies for enhanced functionality:")
        for dep in optional_deps:
            print(f"   pip install {dep}")
        print()
    
    return True

# Import our components (after dependency check)
try:
    from rss_trigger_system import RSSFeedMonitor
    from concurrent_web_scraper import ConcurrentWebScraper
    from smart_text_extraction import SmartTextExtractor
    from data_pipeline_integration import FinancialDataPipeline
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Component import error: {e}")
    COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_smart_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastSmartScraper:
    """Main orchestrator for the Fast & Smart data ingestion system"""
    
    def __init__(self, config_file: str = 'fast_smart_config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
        
        # System components
        self.rss_monitor: Optional[RSSFeedMonitor] = None
        self.web_scraper: Optional[ConcurrentWebScraper] = None
        self.text_extractor: Optional[SmartTextExtractor] = None
        self.data_pipeline: Optional[FinancialDataPipeline] = None
        
        # Control flags
        self.running = False
        self.shutdown_requested = False
        
        # Worker tasks
        self.worker_tasks = []
        
        # Performance tracking
        self.system_stats = {
            'start_time': None,
            'total_articles_processed': 0,
            'total_signals_generated': 0,
            'components_active': 0
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file or create default"""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML not installed, using JSON config fallback")
                try:
                    with open(config_path, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        # Return default configuration
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'financial_data_ingestion',
                'user': 'postgres',
                'password': 'your_password'
            },
            'rss_monitor': {
                'check_interval_minutes': 5,
                'max_concurrent_feeds': 10,
                'feed_timeout': 30
            },
            'web_scraper': {
                'max_concurrent': 50,
                'max_concurrent_per_domain': 5,
                'timeout': 30,
                'max_retries': 3
            },
            'text_extraction': {
                'min_word_count': 50,
                'min_char_count': 300,
                'batch_size': 20
            },
            'pipeline': {
                'analysis_batch_size': 10,
                'max_concurrent_analyses': 3,
                'analysis_timeout': 300
            },
            'finance_llm': {
                'model_path': 'models/finance-llm-13b.Q5_K_S.gguf',
                'context_length': 4096,
                'temperature': 0.3
            },
            'logging': {
                'level': 'INFO',
                'file': 'fast_smart_scraper.log'
            }
        }

    def _save_default_config(self):
        """Save default configuration to file"""
        config_path = Path(self.config_file)
        
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"✅ Saved default config to {config_path}")
        except ImportError:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"✅ Saved default config to {config_path} (JSON format)")

    async def initialize(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing Fast & Smart Data Ingestion System")
        
        if not COMPONENTS_AVAILABLE:
            logger.error("❌ System components not available")
            return False
        
        try:
            # Initialize RSS Monitor
            self.rss_monitor = RSSFeedMonitor(
                db_config=self.config['database'],
                max_concurrent_feeds=self.config['rss_monitor']['max_concurrent_feeds'],
                default_timeout=self.config['rss_monitor']['feed_timeout']
            )
            await self.rss_monitor.initialize()
            logger.info("✅ RSS Monitor initialized")
            
            # Initialize Web Scraper
            self.web_scraper = ConcurrentWebScraper(
                db_config=self.config['database'],
                **self.config['web_scraper']
            )
            await self.web_scraper.initialize()
            logger.info("✅ Web Scraper initialized")
            
            # Initialize Text Extractor
            self.text_extractor = SmartTextExtractor(
                db_config=self.config['database']
            )
            await self.text_extractor.initialize()
            logger.info("✅ Text Extractor initialized")
            
            # Initialize Data Pipeline
            self.data_pipeline = FinancialDataPipeline(
                db_config=self.config['database'],
                llm_config=self.config['finance_llm']
            )
            await self.data_pipeline.initialize()
            logger.info("✅ Data Pipeline initialized")
            
            self.system_stats['start_time'] = time.time()
            logger.info("🎉 All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    async def start_workers(self):
        """Start all worker processes"""
        logger.info("🔄 Starting worker processes...")
        
        try:
            # Start RSS monitoring worker
            rss_task = asyncio.create_task(
                self.rss_monitor.run_continuous_monitoring(),
                name="rss_monitor"
            )
            self.worker_tasks.append(rss_task)
            
            # Start web scraping worker
            scraper_task = asyncio.create_task(
                self.web_scraper.worker_loop(),
                name="web_scraper"
            )
            self.worker_tasks.append(scraper_task)
            
            # Start text extraction worker
            extraction_task = asyncio.create_task(
                self.text_extractor.run_extraction_worker(),
                name="text_extractor"
            )
            self.worker_tasks.append(extraction_task)
            
            # Start analysis pipeline worker
            analysis_task = asyncio.create_task(
                self.data_pipeline.run_analysis_worker(),
                name="analysis_pipeline"
            )
            self.worker_tasks.append(analysis_task)
            
            # Start status monitoring
            monitor_task = asyncio.create_task(
                self._monitor_system_status(),
                name="status_monitor"
            )
            self.worker_tasks.append(monitor_task)
            
            logger.info(f"✅ Started {len(self.worker_tasks)} worker processes")
            self.running = True
            
        except Exception as e:
            logger.error(f"❌ Failed to start workers: {e}")
            raise

    async def _monitor_system_status(self):
        """Monitor system status and log performance metrics"""
        while self.running and not self.shutdown_requested:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Get component stats
                stats = await self.get_system_stats()
                
                logger.info("📊 System Status Update:")
                logger.info(f"   Uptime: {stats['uptime_hours']:.1f} hours")
                logger.info(f"   Articles processed: {stats['total_articles_processed']}")
                logger.info(f"   Signals generated: {stats['total_signals_generated']}")
                logger.info(f"   Processing rate: {stats['articles_per_hour']:.1f}/hour")
                
                # Check component health
                unhealthy_components = []
                if not hasattr(self.rss_monitor, 'running') or not self.rss_monitor.running:
                    unhealthy_components.append("RSS Monitor")
                if self.web_scraper.should_stop:
                    unhealthy_components.append("Web Scraper")
                
                if unhealthy_components:
                    logger.warning(f"⚠️ Unhealthy components: {', '.join(unhealthy_components)}")
                
            except Exception as e:
                logger.error(f"❌ Status monitoring error: {e}")

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            uptime = time.time() - self.system_stats['start_time'] if self.system_stats['start_time'] else 0
            
            # Get component stats
            rss_stats = await self.rss_monitor.get_monitor_stats()
            scraper_stats = await self.web_scraper.get_scraper_stats()
            extraction_stats = self.text_extractor.get_extraction_stats()
            pipeline_stats = await self.data_pipeline.get_pipeline_stats()
            
            # Aggregate stats
            total_articles = (
                scraper_stats.get('jobs_processed', 0) +
                extraction_stats.get('articles_processed', 0)
            )
            
            return {
                'uptime_seconds': uptime,
                'uptime_hours': uptime / 3600,
                'total_articles_processed': total_articles,
                'total_signals_generated': pipeline_stats.get('signals_generated', 0),
                'articles_per_hour': total_articles / max(uptime / 3600, 1),
                'components': {
                    'rss_monitor': rss_stats,
                    'web_scraper': scraper_stats,
                    'text_extractor': extraction_stats,
                    'data_pipeline': pipeline_stats
                },
                'workers_running': len([t for t in self.worker_tasks if not t.done()]),
                'system_healthy': len([t for t in self.worker_tasks if t.done()]) == 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("🛑 Initiating graceful shutdown...")
        self.shutdown_requested = True
        self.running = False
        
        try:
            # Cancel all worker tasks
            for task in self.worker_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to finish
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            
            # Cleanup components
            if self.data_pipeline:
                await self.data_pipeline.cleanup()
            if self.text_extractor:
                await self.text_extractor.cleanup()
            if self.web_scraper:
                await self.web_scraper.cleanup()
            if self.rss_monitor:
                await self.rss_monitor.cleanup()
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def run(self):
        """Main run loop"""
        try:
            # Initialize system
            if not await self.initialize():
                logger.error("❌ Failed to initialize system")
                return
            
            # Start workers
            await self.start_workers()
            
            # Setup signal handlers
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating shutdown...")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            logger.info("🎉 Fast & Smart Data Ingestion System is running!")
            logger.info("   Press Ctrl+C to stop")
            
            # Keep running until shutdown
            while self.running and not self.shutdown_requested:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            await self.shutdown()

# CLI interface
async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fast & Smart Financial Data Ingestion System'
    )
    parser.add_argument(
        '--config', '-c',
        default='fast_smart_config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check system dependencies'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics and exit'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        print("🔍 Checking system dependencies...")
        deps_ok = check_dependencies()
        if deps_ok:
            print("✅ All required dependencies are available")
        else:
            print("❌ Missing required dependencies")
        return
    
    # Create config file
    if args.create_config:
        scraper = FastSmartScraper(args.config)
        scraper._save_default_config()
        return
    
    # Check dependencies before starting
    if not check_dependencies():
        print("❌ Please install missing dependencies before running")
        return
    
    # Initialize and run system
    scraper = FastSmartScraper(args.config)
    
    if args.stats:
        # Show stats and exit
        try:
            await scraper.initialize()
            stats = await scraper.get_system_stats()
            print("📊 System Statistics:")
            print(json.dumps(stats, indent=2, default=str))
            await scraper.shutdown()
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
        return
    
    # Run the system
    await scraper.run()

if __name__ == "__main__":
    print("🚀 Fast & Smart Financial Data Ingestion System")
    print("=" * 50)
    print("🎯 Real-time financial insights with Finance-LLM-13B")
    print("📡 RSS monitoring + concurrent scraping + smart extraction")
    print("🧠 AI-powered analysis + trading signals")
    print("=" * 50)
    
    asyncio.run(main())