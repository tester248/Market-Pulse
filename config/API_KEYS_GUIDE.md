# API Keys Management - FinanceAI Framework

This directory contains all API key configurations and management utilities for the FinanceAI framework.

## 🔑 Quick Setup

### 1. Copy Environment Template
```bash
cp .env.template .env
```

### 2. Edit the .env file with your API keys
```bash
# Edit .env with your favorite editor
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 3. Verify Configuration
```bash
python config/verify_keys.py
```

## 📚 Where to Get API Keys

### 🤖 AI/LLM Providers

#### OpenAI (ChatGPT, GPT-4)
- **URL**: https://platform.openai.com/api-keys
- **Free Tier**: $5 credit (expires after 3 months)
- **Pricing**: $0.002-0.12 per 1K tokens
- **Best For**: General AI analysis, text generation
- **Setup Steps**:
  1. Create OpenAI account
  2. Go to API keys section
  3. Create new secret key
  4. Copy to OPENAI_API_KEY in .env

#### Anthropic (Claude)
- **URL**: https://console.anthropic.com/
- **Free Tier**: $5 credit for new users
- **Pricing**: $3-15 per million tokens
- **Best For**: Reasoning, analysis, long-form content
- **Setup Steps**:
  1. Create Anthropic account
  2. Generate API key in console
  3. Copy to ANTHROPIC_API_KEY in .env

#### Google AI (Gemini)
- **URL**: https://makersuite.google.com/app/apikey
- **Free Tier**: Generous free limits
- **Best For**: Alternative to OpenAI, long context
- **Setup Steps**:
  1. Sign in with Google account
  2. Create API key
  3. Copy to GOOGLE_AI_API_KEY in .env

### 📊 Financial Data Providers

#### Alpha Vantage (Free Stock Data)
- **URL**: https://www.alphavantage.co/support/#api-key
- **Free Tier**: 5 requests/minute, 500/day
- **Best For**: Basic stock data, getting started
- **Setup Steps**:
  1. Visit Alpha Vantage website
  2. Click "Get your free API key today"
  3. Fill out simple form
  4. Copy key to ALPHA_VANTAGE_API_KEY in .env

#### Polygon.io (Professional Market Data)
- **URL**: https://polygon.io/dashboard/api-keys
- **Free Tier**: 5 calls/minute
- **Pricing**: $99/month for unlimited
- **Best For**: Real-time market data, professional use
- **Setup Steps**:
  1. Create Polygon account
  2. Go to API Keys in dashboard
  3. Generate new key
  4. Copy to POLYGON_API_KEY in .env

#### Financial Modeling Prep
- **URL**: https://financialmodelingprep.com/developer/docs
- **Free Tier**: 250 requests/day
- **Best For**: Fundamental analysis, financial statements
- **Setup Steps**:
  1. Sign up for free account
  2. Get API key from dashboard
  3. Copy to FMP_API_KEY in .env

### ₿ Cryptocurrency Data

#### CoinGecko
- **URL**: https://www.coingecko.com/en/api/pricing
- **Free Tier**: 10,000 requests/month
- **Best For**: Comprehensive crypto data
- **Setup Steps**:
  1. Create CoinGecko account
  2. Subscribe to free plan
  3. Get API key from dashboard
  4. Copy to COINGECKO_API_KEY in .env

#### Binance API
- **URL**: https://www.binance.com/en/my/settings/api-management
- **Free**: With Binance account
- **Best For**: Real-time crypto trading data
- **Setup Steps**:
  1. Create Binance account
  2. Complete KYC verification
  3. Go to API Management
  4. Create API key (read-only for data)
  5. Copy key and secret to .env

### 🌐 Blockchain Data

#### Etherscan
- **URL**: https://etherscan.io/apis
- **Free Tier**: 5 requests/second
- **Best For**: Ethereum blockchain data
- **Setup Steps**:
  1. Create Etherscan account
  2. Go to API-KEYs section
  3. Add new API key
  4. Copy to ETHERSCAN_API_KEY in .env

#### Moralis
- **URL**: https://admin.moralis.io/
- **Free Tier**: 40,000 requests/month
- **Best For**: Multi-chain DeFi data
- **Setup Steps**:
  1. Create Moralis account
  2. Create new project
  3. Get API key from settings
  4. Copy to MORALIS_API_KEY in .env

## 🔒 Security Best Practices

### Environment Variables
```bash
# Load environment variables in your code
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

### Git Security
```bash
# Make sure .env is in .gitignore
echo ".env" >> .gitignore
echo "config/api_keys.yaml" >> .gitignore
```

### Production Deployment
```bash
# Use cloud secret management
# AWS Secrets Manager
# Google Secret Manager
# Azure Key Vault
```

## 🚀 Quick Start Examples

### OpenAI Integration
```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze this portfolio"}]
)
```

### Financial Data
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key}"
response = requests.get(url)
data = response.json()
```

### Crypto Data
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('COINGECKO_API_KEY')

headers = {'x-cg-demo-api-key': api_key}
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
response = requests.get(url, headers=headers)
```

## 📋 API Key Checklist

### For Hackathons (Minimum Required)
- [ ] OpenAI API Key (for AI analysis)
- [ ] Alpha Vantage API Key (for basic stock data)
- [ ] CoinGecko API Key (for crypto data)

### For Professional Development
- [ ] OpenAI + Anthropic (AI redundancy)
- [ ] Polygon.io (real-time financial data)
- [ ] Financial Modeling Prep (fundamental data)
- [ ] Binance API (crypto trading data)
- [ ] Etherscan (blockchain data)

### For Production Applications
- [ ] Multiple AI providers (OpenAI, Anthropic, Google)
- [ ] Premium financial data (Polygon, Bloomberg)
- [ ] Cloud services (AWS, GCP, Azure)
- [ ] Database connections
- [ ] Monitoring and alerting

## 🛠️ Troubleshooting

### Common Issues

#### "Invalid API Key"
1. Check key is copied correctly (no extra spaces)
2. Verify key is not expired
3. Check if service requires activation
4. Ensure sufficient credits/quota

#### "Rate Limit Exceeded"
1. Check your current usage
2. Upgrade to paid tier
3. Implement rate limiting in code
4. Use caching to reduce API calls

#### "Insufficient Permissions"
1. Check API key permissions
2. Verify account verification status
3. Review service-specific requirements

### Getting Help
1. Check service documentation
2. Look at FinanceAI examples
3. Review error messages carefully
4. Contact API provider support

## 💰 Cost Management

### Free Tier Limits
- **OpenAI**: $5 credit (3 months)
- **Alpha Vantage**: 500 requests/day
- **CoinGecko**: 10,000 requests/month
- **Etherscan**: 5 requests/second

### Cost Optimization Tips
1. Use caching to reduce API calls
2. Batch requests when possible
3. Choose appropriate AI models (GPT-3.5 vs GPT-4)
4. Monitor usage regularly
5. Set up billing alerts
6. Use mock providers for development

### Budget Planning
- **Basic Hackathon**: $0-50/month
- **Professional Development**: $100-500/month
- **Production Application**: $500-5000/month

---

**Remember**: Never commit API keys to version control! Always use environment variables or secure secret management systems.