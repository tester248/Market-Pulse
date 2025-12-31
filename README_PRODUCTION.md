# Financial Seismograph 🌊📈

> Real-time Financial Sentiment Analysis & AI-Powered Market Intelligence

The **Financial Seismograph** is a production-ready system that monitors financial news in real-time, processes it through a multi-LLM AI assembly line, and provides seismograph-style visualization of market sentiment tremors.

## 🎯 What It Does

- **📰 Smart Data Ingestion**: Continuously monitors 15+ financial RSS feeds
- **🤖 AI Assembly Line**: 6-stage processing pipeline with specialized LLMs
- **🌊 Seismograph Interface**: Real-time sentiment visualization with tremor detection
- **🔍 Epicenter Analysis**: Deep-dive reports on significant market events
- **💬 Interactive Queries**: Natural language questions about market sentiment

## ⚡ Quick Start

### 1. Prerequisites
```bash
# Install Python 3.8+, PostgreSQL 12+, and Ollama
pip install -r requirements.txt
```

### 2. Configure Your Models
Edit `config/models.yaml` with your Ollama model names:
```yaml
ollama:
  default_model: "llama3.1:13b"  # Your finance model
specialized_models:
  triage: "llama3.1:8b"         # Content classification
  sentiment: "mistral:7b"       # Sentiment analysis
  extraction: "llama3.1:13b"    # Data extraction
  analysis: "llama3.1:13b"      # Final analysis
```

### 3. Setup Database
```bash
# Edit config/app.yaml with your database credentials
python production_database_init.py
```

### 4. Start the System
```bash
python production_startup.py
```

🎉 **Your Financial Seismograph is now running!**
- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/system/health
- **Real-time Data**: http://localhost:8000/api/seismograph/data

## 🏗️ System Architecture

### Fast & Smart Data Ingestion Layer
```
RSS Feeds → Smart Scraper → Content Deduplication → Priority Queue → Database
     ↓
- 15+ Financial Sources (Reuters, Bloomberg, etc.)
- Concurrent Processing (10x faster)
- Smart Content Extraction
- PostgreSQL Backend
```

### AI Core: Assembly Line Pipeline
```
Article → Triage → Sentiment → Extraction → Analysis → Integration → Results
         (LLM1)   (LLM2)      (LLM3)      (LLM4)     (LLM5)      (Database)
```

**6 Specialized Stages:**
1. **Triage Dispatcher**: Content classification & routing
2. **Sentiment Processor**: Market sentiment analysis  
3. **Data Extractor**: Financial metrics & entities
4. **Analysis Orchestrator**: Market impact assessment
5. **Integration Layer**: Result aggregation
6. **Quality Assurance**: Output validation

### Seismograph Interface
```
Tremor Detection → Peak Analysis → Epicenter Reports → Frontend API
      ↓                ↓               ↓              ↓
- Real-time Events   - Intensity      - Detailed     - Chart Data
- Confidence Scores  - Thresholds     - Analysis     - Interactive
- Market Impact      - Alerts         - Insights     - Queries
```

## 📊 API Endpoints

### Seismograph Data
```bash
# Real-time sentiment chart data
GET /api/seismograph/data?hours=24&tickers=AAPL,TSLA

# Detected tremors (significant events)  
GET /api/tremors?min_intensity=0.7

# Detailed epicenter analysis
GET /api/epicenter/{tremor_id}
```

### Interactive Queries
```bash
# Natural language questions
POST /api/query
{
  "query": "What's driving Tesla's sentiment today?",
  "tickers": ["TSLA"]
}
```

### System Status
```bash
# Health monitoring
GET /api/system/health
GET /api/system/stats
```

## 🔧 Configuration

### RSS Feeds (`config/rss_feeds.yaml`)
Real financial sources included:
- Economic Times Markets
- NASDAQ Original  
- Financial Times
- Money.com
- MarketWatch Headlines

### Models (`config/models.yaml`)
Configure your Ollama models:
```yaml
ollama:
  host: "localhost"
  port: 11434
  default_model: "your-finance-model"

specialized_models:
  triage: "your-triage-model"
  sentiment: "your-sentiment-model" 
  extraction: "your-extraction-model"
  analysis: "your-analysis-model"

thresholds:
  sentiment_confidence: 0.7
  quality_minimum: 0.6
  processing_timeout: 300
```

### Application (`config/app.yaml`)
Database, API, and system settings:
```yaml
database:
  host: "localhost"
  name: "financial_data_ingestion"
  user: "financeai"
  password: "your_password"

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]

seismograph:
  tremor_threshold: 0.8
  update_interval_seconds: 60
```

## 📈 Performance Features

### Fast & Smart Ingestion
- **10x Faster**: Concurrent RSS processing
- **Smart Deduplication**: Content hash-based
- **Priority Queuing**: Important news first
- **Resilient Processing**: Auto-retry with backoff

### AI Assembly Line
- **Specialized Models**: Each stage optimized
- **Parallel Processing**: Multiple articles simultaneously  
- **Quality Gates**: Validation at each stage
- **Performance Metrics**: Processing time tracking

### Seismograph Interface
- **Real-time Updates**: 1-minute intervals
- **Tremor Detection**: Configurable thresholds
- **Peak Analysis**: Intensity scoring
- **Frontend Ready**: Chart-friendly data format

## 🗄️ Database Schema

**5 Core Tables:**
- `rss_feeds`: Source configuration & monitoring
- `articles`: Raw content storage with deduplication  
- `analysis_results`: AI processing outputs
- `processing_queue`: Task management & retries
- `system_metrics`: Performance monitoring

**Advanced Features:**
- Full-text search indexes
- Time-series optimized queries
- Automatic data archiving
- Performance analytics

## 🚀 Deployment

### Development
```bash
python production_startup.py
```

### Production
```bash
# With Docker (optional)
docker-compose up -d

# With Gunicorn
gunicorn production_api:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Monitoring
- **Health Checks**: `/api/system/health`
- **Metrics**: Built-in Prometheus endpoints
- **Logs**: Structured logging to `logs/`
- **Database**: Performance query analytics

## 🧪 Example Usage

### Python Client
```python
import httpx

# Get real-time sentiment data
response = httpx.get("http://localhost:8000/api/seismograph/data?hours=24")
data = response.json()

# Ask a question
query_response = httpx.post("http://localhost:8000/api/query", json={
    "query": "What's the market sentiment for Apple?",
    "tickers": ["AAPL"]
})
```

### Frontend Integration
```javascript
// Real-time seismograph chart
const chartData = await fetch('/api/seismograph/data?hours=24')
  .then(r => r.json());

// Tremor detection with alerts  
const tremors = await fetch('/api/tremors?min_intensity=0.8')
  .then(r => r.json());

// Interactive AI queries
const response = await fetch('/api/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Analyze the latest Tesla news sentiment",
    tickers: ["TSLA"]
  })
});
```

## 📚 Documentation

- **[Production Deployment Guide](PRODUCTION_DEPLOYMENT_GUIDE.md)**: Complete setup instructions
- **[API Documentation](http://localhost:8000/docs)**: Interactive API explorer
- **[Configuration Reference](config/)**: All configuration options
- **[Database Schema](production_database_init.py)**: Table structure & indexes

## 🛠️ Technology Stack

**Backend:**
- **FastAPI**: Modern async Python web framework
- **PostgreSQL**: Production database with time-series optimization
- **Ollama**: Local LLM inference engine
- **Scrapy**: High-performance web scraping

**AI/ML:**
- **Multi-LLM Pipeline**: Specialized models for each task
- **Assembly Line Architecture**: 6-stage processing 
- **Real-time Analysis**: Continuous sentiment monitoring
- **Quality Assurance**: Automated output validation

**Data Processing:**
- **RSS Monitoring**: 15+ financial news sources
- **Smart Extraction**: Content deduplication & cleaning
- **Concurrent Processing**: Parallel article analysis
- **Time-series Storage**: Optimized for financial data

## 📊 System Metrics

The system tracks comprehensive metrics:
- **Throughput**: Articles processed per hour
- **Latency**: End-to-end processing time  
- **Quality**: AI output confidence scores
- **Reliability**: Success rates & error tracking
- **Performance**: Database query optimization

## 🔐 Security & Production

### Security Features
- **Configuration Management**: Centralized secrets
- **Database Security**: User isolation & permissions
- **API Security**: CORS configuration & rate limiting
- **Input Validation**: Comprehensive data sanitization

### Production Ready
- **Health Monitoring**: Comprehensive system checks
- **Graceful Shutdown**: Clean resource cleanup
- **Error Handling**: Robust error recovery
- **Logging**: Structured logging for monitoring

## 🆘 Troubleshooting

### Common Issues
1. **Database Connection**: Check PostgreSQL service & credentials
2. **Ollama Models**: Verify models are pulled and running
3. **RSS Feeds**: Check internet connectivity & feed validity
4. **Performance**: Monitor system resources & queue sizes

### Debug Commands
```bash
# Test database connection
python -c "from config_manager import get_config; print('DB Config:', get_config().database_config)"

# Check Ollama models
ollama list

# Validate configuration
python -c "from config_manager import validate_configuration, get_config; print(validate_configuration(get_config()))"

# View system logs
tail -f logs/financial_seismograph.log
```

## 🤝 Contributing

This is a production-ready financial intelligence system. Key areas for enhancement:
- Additional financial data sources
- More specialized AI models
- Advanced visualization features
- Mobile API endpoints
- Real-time alerting systems

## 📄 License

This project is part of the FinanceAI Framework - designed for hackathons and financial technology development.

---

## 🌟 Success Metrics

When fully operational, your Financial Seismograph will:
- ✅ Process **100+ articles per hour** automatically
- ✅ Detect market **sentiment tremors** in real-time  
- ✅ Provide **AI-powered insights** for investment decisions
- ✅ Offer **seismograph visualization** of market events
- ✅ Enable **natural language queries** about market sentiment

**Ready to detect the next market tremor?** 🌊📈