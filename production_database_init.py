"""
Database Initialization Script for Financial Data Ingestion System
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import logging

from config_manager import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get database configuration from config system
config = get_config()
db_config = config.database_config

def create_database_and_user():
    """Create database and user if they don't exist"""
    try:
        # Connect as admin user
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            user='postgres',  # Admin user
            password=db_config.admin_password,
            database='postgres'  # Connect to default database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_config.name}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {db_config.name}")
            logger.info(f"✅ Created database: {db_config.name}")
        else:
            logger.info(f"ℹ️ Database {db_config.name} already exists")
        
        # Create user if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_user WHERE usename = '{db_config.user}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE USER {db_config.user} WITH PASSWORD '{db_config.password}'")
            cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_config.name} TO {db_config.user}")
            logger.info(f"✅ Created user: {db_config.user}")
        else:
            logger.info(f"ℹ️ User {db_config.user} already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to create database/user: {e}")
        sys.exit(1)

def create_tables():
    """Create all required tables with production schema"""
    try:
        # Connect as application user
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            user=db_config.user,
            password=db_config.password,
            database=db_config.name
        )
        cursor = conn.cursor()
        
        # Enable UUID extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        
        # RSS Feeds Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rss_feeds (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                url TEXT NOT NULL UNIQUE,
                category VARCHAR(100),
                enabled BOOLEAN DEFAULT true,
                last_fetched TIMESTAMP,
                fetch_frequency_minutes INTEGER DEFAULT 15,
                total_articles_fetched INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Articles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                rss_feed_id INTEGER REFERENCES rss_feeds(id),
                title TEXT NOT NULL,
                content TEXT,
                url TEXT NOT NULL UNIQUE,
                published_at TIMESTAMP,
                author VARCHAR(200),
                tickers TEXT[],
                content_hash VARCHAR(64),
                word_count INTEGER,
                language VARCHAR(10) DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis Results Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES articles(id),
                content_type VARCHAR(50),
                priority_score FLOAT,
                sentiment VARCHAR(20),
                sentiment_confidence FLOAT,
                market_impact VARCHAR(20),
                tickers TEXT[],
                key_insights JSONB,
                financial_metrics JSONB,
                executive_summary TEXT,
                quality_grade VARCHAR(20),
                quality_score FLOAT,
                extracted_metrics_count INTEGER DEFAULT 0,
                processing_time_ms FLOAT,
                model_versions JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Processing Queue Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_queue (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES articles(id),
                priority INTEGER DEFAULT 5,
                status VARCHAR(20) DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                assigned_worker VARCHAR(100),
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System Metrics Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT NOT NULL,
                metric_unit VARCHAR(20),
                tags JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_articles_tickers ON articles USING GIN(tickers)",
            "CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url, content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_sentiment ON analysis_results(sentiment, sentiment_confidence)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_tickers ON analysis_results USING GIN(tickers)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_created_at ON analysis_results(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_queue_status_priority ON processing_queue(status, priority DESC)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON system_metrics(metric_name, timestamp DESC)"
        ]
        
        for index in indexes:
            cursor.execute(index)
        
        # Create triggers for updated_at timestamps
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)
        
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_rss_feeds_updated_at ON rss_feeds;
            CREATE TRIGGER update_rss_feeds_updated_at
                BEFORE UPDATE ON rss_feeds
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ All tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        sys.exit(1)

def populate_rss_feeds():
    """Populate RSS feeds from configuration"""
    try:
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            user=db_config.user,
            password=db_config.password,
            database=db_config.name
        )
        cursor = conn.cursor()
        
        # Get RSS feeds from configuration
        for feed in config.rss_feeds:
            cursor.execute("""
                INSERT INTO rss_feeds (name, url, category, enabled, fetch_frequency_minutes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    name = EXCLUDED.name,
                    category = EXCLUDED.category,
                    enabled = EXCLUDED.enabled,
                    fetch_frequency_minutes = EXCLUDED.fetch_frequency_minutes,
                    updated_at = CURRENT_TIMESTAMP
            """, (feed.name, feed.url, feed.category, feed.enabled, feed.fetch_frequency))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Populated {len(config.rss_feeds)} RSS feeds")
        
    except Exception as e:
        logger.error(f"❌ Failed to populate RSS feeds: {e}")
        sys.exit(1)

def create_sample_data():
    """Create sample data for testing (optional)"""
    try:
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            user=db_config.user,
            password=db_config.password,
            database=db_config.name
        )
        cursor = conn.cursor()
        
        # Add sample system metrics
        sample_metrics = [
            ('articles_processed_total', 0, 'count'),
            ('processing_queue_size', 0, 'count'),
            ('average_processing_time', 0, 'milliseconds'),
            ('system_health_score', 1.0, 'ratio')
        ]
        
        for metric_name, value, unit in sample_metrics:
            cursor.execute("""
                INSERT INTO system_metrics (metric_name, metric_value, metric_unit)
                VALUES (%s, %s, %s)
            """, (metric_name, value, unit))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Sample data created")
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")

def test_database_connection():
    """Test database connection and basic operations"""
    try:
        conn = psycopg2.connect(
            host=db_config.host,
            port=db_config.port,
            user=db_config.user,
            password=db_config.password,
            database=db_config.name
        )
        cursor = conn.cursor()
        
        # Test basic operations
        cursor.execute("SELECT COUNT(*) FROM rss_feeds")
        feed_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        article_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        analysis_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"🧪 Database test successful:")
        logger.info(f"   📰 RSS Feeds: {feed_count}")
        logger.info(f"   📄 Articles: {article_count}")
        logger.info(f"   🔍 Analysis Results: {analysis_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

def main():
    """Main initialization process"""
    logger.info("🚀 Starting database initialization...")
    
    # Check if admin password is configured
    if not db_config.admin_password:
        logger.error("❌ Admin password not configured. Please set POSTGRES_ADMIN_PASSWORD in config/app.yaml")
        sys.exit(1)
    
    try:
        # Step 1: Create database and user
        create_database_and_user()
        
        # Step 2: Create tables and indexes
        create_tables()
        
        # Step 3: Populate RSS feeds from configuration
        populate_rss_feeds()
        
        # Step 4: Create sample data (optional)
        create_sample_data()
        
        # Step 5: Test connection
        if test_database_connection():
            logger.info("🎉 Database initialization completed successfully!")
            logger.info("📋 Next steps:")
            logger.info("   1. Update your Ollama model names in config/models.yaml")
            logger.info("   2. Run the API: python production_api.py")
            logger.info("   3. Start the integration: python -c 'from fast_smart_integration import FastSmartIntegration; import asyncio; asyncio.run(FastSmartIntegration().start_continuous_processing())'")
        else:
            logger.error("❌ Database initialization failed during testing")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Database initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()