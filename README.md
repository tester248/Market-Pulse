# Market Pulse 🌊📈

> Real-time Financial Sentiment Analysis & AI-Powered Market Intelligence

**Market Pulse** is a sophisticated financial insights engine that monitors financial news in real-time, processes it through a multi-LLM AI assembly line, and provides seismograph-style visualization of market sentiment tremors. Built with OpenRouter integration for enterprise-grade AI capabilities.

## 🎯 What It Does

- **📰 Smart Data Ingestion**: Continuously monitors 15+ financial RSS feeds
- **🤖 AI Assembly Line**: Multi-stage processing pipeline with specialized LLMs via OpenRouter
- **🌊 Seismograph Interface**: Real-time sentiment visualization with tremor detection
- **🔍 Epicenter Analysis**: Deep-dive reports on significant market events
- **💬 Interactive Queries**: Natural language questions about market sentiment
- **📊 Live Headlines**: Aggregated news from configured RSS feeds
- **🎛️ Modern Frontend**: React/TypeScript dashboard with real-time charts

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- OpenRouter API key

### 1. Clone and Setup Backend

```bash
git clone <repository-url>
cd MarketPulseWorkspace

# Install Python dependencies
pip install -r requirements_fast_smart.txt

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
# Or on Windows:
set OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Start the API Server

```bash
# Option 1: Using the startup script (recommended)
python start_market_pulse.py

# Option 2: Direct uvicorn
python production_startup.py

# Option 3: Development mode
python run_market_pulse.py --reload
```

The API will be available at `http://localhost:8000`

### 3. Setup Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

The frontend dashboard will be available at `http://localhost:5173`

### 4. Test the API

```bash
# Run comprehensive API tests
python test_market_pulse_api.py

# Quick health check
curl http://localhost:8000/api/v1/health
```

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/api/v1/health` | System health status |
| GET | `/api/v1/pulse/timeline` | Seismograph data |
| GET | `/api/v1/insights/event/{id}` | Event details |
| POST | `/api/v1/insights/query` | AI-powered queries |
| POST | `/api/v1/insights/process-article` | Article processing |
| GET | `/api/v1/news/headlines` | Live RSS headlines |

### Example Usage

#### Get Seismograph Data
```bash
curl "http://localhost:8000/api/v1/pulse/timeline?hours=24&tickers=AAPL,MSFT&min_confidence=0.7"
```

#### Ask AI Questions
```bash
curl -X POST "http://localhost:8000/api/v1/insights/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do rising interest rates affect tech stocks?",
    "use_rag": true
  }'
```

#### Get Live Headlines
```bash
curl "http://localhost:8000/api/v1/news/headlines?limit_per_feed=5"
```

#### Process Financial Articles
```bash
curl -X POST "http://localhost:8000/api/v1/insights/process-article" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fed Announces Rate Decision",
    "content": "The Federal Reserve announced...",
    "source": "Financial Times",
    "url": "https://ft.com/article/123"
  }'
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RSS Feeds     │───▶│  Market Pulse    │───▶│   Frontend      │
│   (15+ sources) │    │   API Server     │    │   Dashboard     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  OpenRouter      │
                       │  Assembly Line   │
                       │  - Triage        │
                       │  - Sentiment     │
                       │  - Analysis      │
                       │  - Synthesis     │
                       └──────────────────┘
```

### Key Components

1. **OpenRouter Assembly Line** (`openrouter_assembly_line.py`)
   - Multi-stage LLM processing pipeline
   - Triage → Specialist Analysis → Synthesis
   - Uses multiple models for optimal results

2. **Market Pulse API** (`market_pulse_api.py`)
   - FastAPI server with CORS support
   - Real-time data endpoints
   - Background article processing

3. **RSS Feed Processor** (integrated in `openrouter_llm_manager.py`)
   - Monitors configured financial news feeds
   - Extracts and processes articles
   - Provides context for RAG queries

4. **Frontend Dashboard** (`frontend/`)
   - React/TypeScript with Vite
   - Real-time seismograph charts
   - Interactive AI chat interface

## 🔧 Configuration

### API Keys (`config/api_keys.yaml`)
```yaml
openrouter:
  api_key: "your_openrouter_api_key"
  base_url: "https://openrouter.ai/api/v1/chat/completions"
```

### RSS Feeds (`config/rss_feeds.yaml`)
```yaml
rss_feeds:
  - name: "Economic Times Markets"
    url: "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
  - name: "NASDAQ Original"
    url: "https://www.nasdaq.com/feed/rssoutbound"
  - name: "Financial Times"
    url: "https://www.ft.com/rss/home"
  # ... more feeds
```

### Models Configuration (`config/models.yaml`)
```yaml
models:
  triage: "meta-llama/llama-3.3-8b-instruct:free"
  sentiment: "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
  general: "meta-llama/llama-3.3-8b-instruct:free"
  # ... additional models
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance async web framework
- **OpenRouter**: Multi-LLM API integration
- **aiohttp**: Async HTTP client for RSS processing
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server

### Frontend
- **React 18**: Modern UI framework
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool
- **Tailwind CSS**: Utility-first styling

### AI/ML
- **OpenRouter**: Access to multiple LLMs
- **RAG (Retrieval Augmented Generation)**: Context-aware responses
- **Multi-model Pipeline**: Specialized models for different tasks

## 📈 Features

### Real-time Sentiment Analysis
- Continuous monitoring of financial news
- Sentiment scoring (-1.0 to 1.0)
- Confidence metrics and quality scores

### Seismograph Visualization
- Time-series sentiment data
- Market tremor detection
- Interactive charts with filtering

### AI-Powered Insights
- Natural language query interface
- Context-aware responses using current news
- Multi-stage analysis pipeline

### Live Data Feeds
- 15+ configured RSS sources
- Real-time headline aggregation
- Automatic article processing

## 🧪 Testing

### Run All Tests
```bash
python test_market_pulse_api.py
```

### Test Specific Endpoints
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Timeline data
curl "http://localhost:8000/api/v1/pulse/timeline?hours=12"

# Live headlines
curl http://localhost:8000/api/v1/news/headlines
```

### Frontend Testing
```bash
cd frontend
npm run test
npm run lint
```

## 🚀 Development

### Backend Development
```bash
# Install dependencies
pip install -r requirements_fast_smart.txt

# Run in development mode with auto-reload
python run_market_pulse.py --reload --host 0.0.0.0 --port 8000

# Run with custom API key
python run_market_pulse.py --api-key your_key_here
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check
```

### Code Quality
```bash
# Python linting
flake8 *.py

# TypeScript linting
cd frontend && npm run lint

# Fix auto-fixable issues
cd frontend && npm run lint:fix
```

## 📚 API Documentation

### Interactive Documentation
When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Query Examples

#### Market Sentiment Analysis
```json
{
  "query": "What's the current sentiment around Tesla stock?",
  "use_rag": true
}
```

#### Economic Impact Assessment
```json
{
  "query": "How might the latest Fed decision impact tech stocks?",
  "use_rag": true
}
```

#### Sector Analysis
```json
{
  "query": "Analyze the sentiment trends in the banking sector",
  "use_rag": false
}
```

## 🔐 Security & Production

### Environment Variables
```bash
# Required
export OPENROUTER_API_KEY="your_api_key"

# Optional
export MARKET_PULSE_HOST="0.0.0.0"
export MARKET_PULSE_PORT="8000"
export LOG_LEVEL="INFO"
```

### Production Deployment
```bash
# Using production startup script
python production_startup.py

# With custom configuration
python production_startup.py --host 0.0.0.0 --port 8000

# Using gunicorn (recommended for production)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker market_pulse_api:app
```

### Health Monitoring
```bash
# System health
curl http://localhost:8000/api/v1/health

# Model status
curl http://localhost:8000/api/v1/health | jq '.models'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python test_market_pulse_api.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Add tests for new features
- Update documentation as needed

## 🆘 Troubleshooting

### Common Issues

#### API Key Issues
```bash
# Check if API key is set
echo $OPENROUTER_API_KEY

# Test API key directly
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models
```

#### Server Won't Start
```bash
# Check port availability
netstat -an | grep 8000

# Try different port
python run_market_pulse.py --port 8001
```

#### Frontend Build Issues
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### RSS Feed Issues
- Check `config/rss_feeds.yaml` for valid URLs
- Test individual feeds manually
- Verify network connectivity

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_market_pulse.py

# Verbose API testing
python test_market_pulse_api.py --verbose
```

## 📄 License

This project is part of the FinanceAI Framework - designed for hackathons and financial technology development.

## 🌟 Success Metrics

When fully operational, Market Pulse will:
- ✅ Process **100+ articles per hour** automatically
- ✅ Detect market **sentiment tremors** in real-time
- ✅ Provide **AI-powered insights** for investment decisions
- ✅ Offer **seismograph visualization** of market events
- ✅ Enable **natural language queries** about market sentiment
- ✅ Aggregate **live headlines** from multiple sources

---

**Ready to detect the next market tremor?** 🌊📈

For detailed API documentation, see [API_ENDPOINTS.md](API_ENDPOINTS.md)  
For production deployment guide, see [README_PRODUCTION.md](README_PRODUCTION.md)