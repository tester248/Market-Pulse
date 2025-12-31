# Fast & Smart Data Ingestion Layer Setup Guide

## 🚀 Complete "Fast & Smart" Financial Data Scraper

This is the enterprise-grade data ingestion layer you requested - a high-performance system that monitors dozens of RSS feeds, scrapes hundreds of concurrent articles, extracts clean content with trafilatura, stores everything in PostgreSQL, and feeds Finance-LLM-13B for real-time financial insights.

## 🎯 What This System Does

✅ **RSS Trigger System**: Monitors dozens of Tier-1 financial RSS feeds (Reuters, Bloomberg, SEC, etc.)  
✅ **Concurrent Web Scraper**: Fetches hundreds of articles simultaneously with intelligent rate limiting  
✅ **Smart Text Extraction**: Uses trafilatura + fallbacks to extract clean content, removing ads/navigation  
✅ **PostgreSQL Storage**: Enterprise-grade database with full-text search and performance optimization  
✅ **Finance-LLM-13B Integration**: AI-powered sentiment analysis and trading signal generation  
✅ **Real-time Pipeline**: Continuous processing from RSS → scraping → extraction → analysis → signals  

## 📋 Prerequisites

### 1. PostgreSQL Database
```bash
# Install PostgreSQL (if not already installed)
# Windows: Download from https://www.postgresql.org/download/windows/
# Linux: sudo apt-get install postgresql postgresql-contrib
# macOS: brew install postgresql

# Create database
psql -U postgres
CREATE DATABASE financial_data_ingestion;
CREATE USER financeai WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE financial_data_ingestion TO financeai;
\q
```

### 2. Python Dependencies
```bash
# Install all required packages
pip install -r requirements_fast_smart.txt

# For optimal performance on Linux/macOS, also install:
pip install uvloop

# Check installation
python fast_smart_scraper.py --check-deps
```

### 3. Database Schema Setup
```bash
# Run the PostgreSQL schema setup
psql -U financeai -d financial_data_ingestion -f database/postgresql_schema.sql
```

## ⚙️ Configuration

### 1. Create Configuration File
```bash
# Generate default config
python fast_smart_scraper.py --create-config

# This creates fast_smart_config.yaml with all settings
```

### 2. Update Database Settings
Edit `fast_smart_config.yaml`:
```yaml
database:
  host: localhost
  port: 5432
  database: financial_data_ingestion
  user: financeai
  password: your_secure_password

finance_llm:
  model_path: models/finance-llm-13b.Q5_K_S.gguf  # Update to your model path
  context_length: 4096
  temperature: 0.3
```

### 3. RSS Feed Configuration
The system uses your existing `config/rss_feeds.yaml` with 15+ Tier-1 financial sources already configured.

## 🚀 Running the System

### Option 1: Full System (Recommended)
```bash
# Start the complete Fast & Smart system
python fast_smart_scraper.py

# With custom config
python fast_smart_scraper.py --config my_config.yaml
```

### Option 2: Individual Components (Testing)
```bash
# Test RSS monitoring only
python rss_trigger_system.py

# Test web scraping only  
python concurrent_web_scraper.py

# Test text extraction only
python smart_text_extraction.py

# Test analysis pipeline only
python data_pipeline_integration.py
```

### Option 3: Check System Status
```bash
# Get system statistics
python fast_smart_scraper.py --stats
```

## 📊 System Architecture

```
RSS Feeds (Tier-1 Sources)
    ↓
RSS Trigger System (asyncio monitoring)
    ↓
Job Queue (PostgreSQL)
    ↓
Concurrent Web Scraper (aiohttp + rate limiting)
    ↓
Raw HTML Storage (PostgreSQL)
    ↓
Smart Text Extraction (trafilatura + fallbacks)
    ↓
Clean Article Storage (PostgreSQL + full-text search)
    ↓
Finance-LLM-13B Analysis (sentiment + signals)
    ↓
Trading Signals & Insights (PostgreSQL)
    ↓
Virtual Trading Sandbox Integration
```

## 🎛️ System Components

### 1. RSS Trigger System (`rss_trigger_system.py`)
- Monitors dozens of RSS feeds with sub-minute latency
- Creates scraping jobs for new articles
- Handles feed failures gracefully
- Performance metrics and health monitoring

### 2. Concurrent Web Scraper (`concurrent_web_scraper.py`)
- Fetches hundreds of articles simultaneously
- Intelligent rate limiting per domain
- Automatic retries with exponential backoff
- Respects robots.txt and site-specific delays

### 3. Smart Text Extraction (`smart_text_extraction.py`)
- Primary: trafilatura (best-in-class extraction)
- Fallback 1: readability-lxml
- Fallback 2: newspaper3k
- Fallback 3: regex-based extraction
- Financial content enhancement (ticker extraction, categorization)

### 4. Data Pipeline Integration (`data_pipeline_integration.py`)
- Finance-LLM-13B sentiment analysis
- Trading signal generation
- Market impact assessment
- Risk analysis and time horizon classification

### 5. Main Orchestrator (`fast_smart_scraper.py`)
- Coordinates all components
- Health monitoring and auto-recovery
- Performance metrics and logging
- Graceful shutdown handling

## 📈 Performance Expectations

With default settings, you can expect:
- **RSS Monitoring**: 50+ feeds checked every 5 minutes
- **Web Scraping**: 50+ concurrent requests (respecting rate limits)
- **Text Extraction**: 20+ articles per minute
- **LLM Analysis**: 10+ articles per minute (depends on model speed)
- **Total Throughput**: 200-500 articles per hour (depends on news volume)

## 🔧 Customization

### Scaling Up Performance
```yaml
# In fast_smart_config.yaml
web_scraper:
  max_concurrent: 100          # More concurrent requests
  max_concurrent_per_domain: 10  # Higher domain limits

pipeline:
  analysis_batch_size: 20      # Larger analysis batches
  max_concurrent_analyses: 5   # More parallel analyses
```

### Adding More RSS Feeds
Edit `config/rss_feeds.yaml`:
```yaml
additional_feeds:
  - name: "Custom Financial News"
    url: "https://example.com/financial-news.rss"
    category: "news"
    priority: 5
    enabled: true
```

### Custom Analysis Prompts
Modify the analysis prompts in `data_pipeline_integration.py` for specialized analysis:
```python
self.analysis_prompts = {
    'crypto_analysis': "Analyze this crypto news for...",
    'earnings_analysis': "Focus on earnings implications...",
    # Add your custom prompts
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check PostgreSQL is running
   sudo systemctl status postgresql
   
   # Test connection
   psql -U financeai -d financial_data_ingestion -c "SELECT 1;"
   ```

2. **Missing Dependencies**
   ```bash
   # Check all dependencies
   python fast_smart_scraper.py --check-deps
   
   # Install missing packages
   pip install trafilatura asyncpg aiohttp feedparser
   ```

3. **Finance-LLM-13B Not Found**
   ```bash
   # Update model path in config
   # Ensure your finance-llm-13b.Q5_K_S.gguf is in the correct location
   ```

4. **Rate Limiting Issues**
   ```bash
   # Reduce concurrent requests in config
   # Check system logs for blocked domains
   tail -f fast_smart_scraper.log
   ```

### Performance Optimization

1. **Database Performance**
   ```sql
   -- Check index usage
   SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch 
   FROM pg_stat_user_indexes;
   
   -- Optimize queries
   VACUUM ANALYZE articles;
   ```

2. **Memory Usage**
   ```bash
   # Monitor memory usage
   htop
   
   # Reduce batch sizes if needed
   ```

## 🔐 Security Considerations

1. **Database Security**
   - Use strong passwords
   - Enable SSL connections in production
   - Restrict database access by IP

2. **Web Scraping Ethics**
   - The system respects robots.txt
   - Implements rate limiting
   - Uses appropriate user agents

3. **Content Storage**
   - Articles are stored with source attribution
   - No copyright violation intended
   - For research/analysis purposes only

## 📞 Support

If you encounter issues:

1. Check the logs: `tail -f fast_smart_scraper.log`
2. Run dependency check: `python fast_smart_scraper.py --check-deps`
3. Test individual components separately
4. Check PostgreSQL logs for database issues
5. Verify Finance-LLM-13B model path and permissions

## 🎉 Success Metrics

You'll know the system is working when you see:
- ✅ RSS feeds being monitored every 5 minutes
- ✅ Scraping jobs being created and processed
- ✅ Articles being extracted and stored
- ✅ Sentiment analysis and trading signals being generated
- ✅ Real-time financial insights flowing to your trading sandbox

The "Fast & Smart" data ingestion layer is now complete and ready to power your Finance-LLM-13B system with real-time financial intelligence!