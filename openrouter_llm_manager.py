"""
OpenRouter Multi-LLM Manager

Orchestrates multiple specialized finance models via OpenRouter with configuration management.
Provides model management, health monitoring, and intelligent routing.
Designed as a drop-in replacement for the Ollama Multi-LLM Manager.
"""

import asyncio
import aiohttp
import json
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use same ModelType enum as in the Ollama manager for compatibility
class ModelType(Enum):
    """Types of specialized finance models"""
    TRIAGE = "triage"           # Content classification and routing
    SENTIMENT = "sentiment"     # Sentiment analysis (FinBERT-style)
    EXTRACTION = "extraction"   # Data extraction and summarization
    GENERAL = "general"         # General-purpose finance model

@dataclass
class ModelConfig:
    """Configuration for a specialized model"""
    name: str                   # Descriptive name
    model_type: ModelType       # Type of specialist
    openrouter_model: str       # Model name on OpenRouter
    description: str            # What this model does
    max_context: int = 4096     # Maximum context length
    temperature: float = 0.1    # Temperature for inference
    max_concurrent: int = 3     # Max concurrent requests
    timeout_seconds: int = 30   # Request timeout 
    enabled: bool = True        # Whether model is active

@dataclass
class ModelResponse:
    """Response from a model"""
    model_name: str
    model_type: ModelType
    content: str
    metadata: Dict[str, Any]
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class ModelHealth:
    """Health status of a model"""
    model_name: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    success_count: int = 0
    uptime_percentage: float = 100.0

class RSSFeedProcessor:
    def __init__(self):
        self.rss_feeds = [
            "https://www.nasdaq.com/feed/nasdaq-original/rss.xml",
            "https://www.ft.com/rss/home", 
            "https://money.com/money/feed/",
            "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"
        ]
        self.articles = []
    
    def fetch_rss_content(self, max_articles_per_feed: int = 10) -> List[Dict]:
        """Fetch and parse RSS feeds with timeout protection"""
        import feedparser
        import time
        import requests
        from datetime import datetime
        import threading
        
        all_articles = []
        
        for feed_url in self.rss_feeds:
            try:
                # Use requests with timeout first, then feedparser
                logger.info(f"Fetching RSS feed: {feed_url}")
                response = requests.get(feed_url, timeout=8)
                feed = feedparser.parse(response.content)
                
                logger.info(f"Parsed {len(feed.entries)} entries from {feed_url}")
                source_name = feed.feed.get('title', feed_url.split('/')[2])
                
                for entry in feed.entries[:max_articles_per_feed]:
                    # Parse date if available
                    published = entry.get('published', entry.get('updated', None))
                    if published:
                        try:
                            pub_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
                        except:
                            pub_date = datetime.now()
                    else:
                        pub_date = datetime.now()
                    
                    # Extract content
                    content = self.extract_article_content(entry)
                    
                    # Create article object
                    article = {
                        'title': entry.get('title', 'No Title'),
                        'link': entry.get('link', ''),
                        'published': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'summary': entry.get('summary', '')[:300],
                        'content': content,
                        'source': source_name
                    }
                    
                    all_articles.append(article)
                    
            except (requests.RequestException, Exception) as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {str(e)}")
                continue
        
        self.articles = all_articles
        return all_articles
    
    def extract_article_content(self, entry) -> str:
        """Extract content from RSS entry"""
        content = ""
        
        # Try to get content from various fields
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value
        elif hasattr(entry, 'summary') and entry.summary:
            content = entry.summary
        elif hasattr(entry, 'description') and entry.description:
            content = entry.description
        
        # Clean HTML if present
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text().strip()
        
        return content[:1000]  # Limit content length
    
    def create_context_from_articles(self, query: str, max_articles: int = 15) -> str:
        """Create context from relevant articles for RAG"""
        if not self.articles:
            self.fetch_rss_content()
        
        # Simple relevance scoring based on keyword matching
        query_words = set(query.lower().split())
        
        scored_articles = []
        for article in self.articles:
            article_text = f"{article['title']} {article['summary']} {article['content']}".lower()
            
            # Calculate relevance score based on keyword matches
            word_matches = sum(1 for word in query_words if word in article_text)
            
            # Add recency bonus
            try:
                pub_time = datetime.strptime(article['published'], '%Y-%m-%d %H:%M:%S')
                hours_ago = (datetime.now() - pub_time).total_seconds() / 3600
                recency_score = max(0, 24 - hours_ago) / 24  # Higher score for newer articles
            except:
                recency_score = 0.5  # Default if date parsing fails
            
            relevance_score = word_matches + (recency_score * 0.5)
            
            scored_articles.append({
                **article,
                'relevance_score': relevance_score
            })
        
        # Sort by relevance and take top articles
        scored_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_articles = scored_articles[:max_articles]
        
        # Create context string
        context = "=== RECENT FINANCIAL NEWS CONTEXT ===\n\n"
        
        for i, article in enumerate(top_articles, 1):
            published = article.get('published', 'Unknown date')
            context += f"{i}. **{article['title']}**\n"
            context += f"   Source: {article['source']}\n"
            context += f"   Published: {published}\n"
            context += f"   Summary: {article['summary'][:300]}...\n"
            if article['content']:
                context += f"   Content: {article['content'][:400]}...\n"
            context += f"   Link: {article['link']}\n\n"
        
        context += "=== END OF NEWS CONTEXT ===\n\n"
        return context, len(top_articles)

class OpenRouterMultiLLMManager:
    """Production multi-LLM manager using OpenRouter API"""
    
    def __init__(self, api_key: str = None):
        # Use environment variable or fallback to provided key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.error("No OpenRouter API key found. Set OPENROUTER_API_KEY environment variable.")
            raise ValueError("OpenRouter API key is required")
        
        # Base URL for OpenRouter API
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Model mapping
        self.model_mapping = {
            ModelType.TRIAGE: "meta-llama/llama-3.3-8b-instruct:free",
            ModelType.SENTIMENT: "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            ModelType.EXTRACTION: "nousresearch/deephermes-3-llama-3-8b-preview:free",
            ModelType.GENERAL: "meta-llama/llama-3.3-8b-instruct:free"
        }
        
        # Model configurations
        self.models: Dict[str, ModelConfig] = {
            "triage": ModelConfig(
                name="Triage Specialist",
                model_type=ModelType.TRIAGE,
                openrouter_model=self.model_mapping[ModelType.TRIAGE],
                description="Classifies content and routes to appropriate specialists",
                temperature=0.1,
                timeout_seconds=20,
                max_concurrent=5
            ),
            "sentiment": ModelConfig(
                name="Sentiment Analyst",
                model_type=ModelType.SENTIMENT,
                openrouter_model=self.model_mapping[ModelType.SENTIMENT],
                description="Analyzes sentiment in financial content",
                temperature=0.2,
                timeout_seconds=25,
                max_concurrent=3
            ),
            "extraction": ModelConfig(
                name="Data Extractor",
                model_type=ModelType.EXTRACTION,
                openrouter_model=self.model_mapping[ModelType.EXTRACTION],
                description="Extracts structured data from financial text",
                temperature=0.1,
                timeout_seconds=30,
                max_concurrent=3
            ),
            "general": ModelConfig(
                name="General Finance Expert",
                model_type=ModelType.GENERAL,
                openrouter_model=self.model_mapping[ModelType.GENERAL],
                description="General financial analysis and insights",
                temperature=0.7,
                timeout_seconds=35,
                max_concurrent=2
            )
        }
        
        # Model health tracking
        self.model_health: Dict[str, ModelHealth] = {}
        
        # Concurrency control
        self.model_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Initialize semaphores and health status
        for model_name, model_config in self.models.items():
            self.model_semaphores[model_name] = asyncio.Semaphore(model_config.max_concurrent)
            self.model_health[model_name] = ModelHealth(
                model_name=model_name,
                is_healthy=True,
                last_check=datetime.now(),
                response_time_ms=0
            )
        
        # RSS processor for RAG
        self.rss_processor = RSSFeedProcessor()
        
        # HTTP session
        self.session = None

    async def ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close resources"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def query_model(self, model_name: str, messages: List[Dict], timeout: int = None) -> Dict[str, Any]:
        """Query a single model with error handling and timeout"""
        if model_name not in self.models:
            return {
                "model": model_name,
                "response": None,
                "success": False,
                "error": f"Unknown model: {model_name}",
                "timestamp": time.time()
            }
        
        model_config = self.models[model_name]
        openrouter_model = model_config.openrouter_model
        
        # Use model-specific timeout or default
        timeout = timeout or model_config.timeout_seconds
        
        # Use semaphore to limit concurrent requests
        async with self.model_semaphores[model_name]:
            start_time = time.time()
            session = await self.ensure_session()
            
            try:
                logger.info(f"Querying {model_name} ({openrouter_model})...")
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": openrouter_model,
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": model_config.temperature
                }
                
                async with session.post(
                    self.base_url, 
                    headers=headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    # Update health metrics
                    self.update_health_metrics(model_name, True, response_time_ms)
                    
                    if response.status == 200:
                        data = await response.json()
                        response_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        return {
                            "model": model_name,
                            "response": response_content,
                            "success": True,
                            "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                            "timestamp": time.time(),
                            "status_code": response.status,
                            "processing_time_ms": response_time_ms
                        }
                    else:
                        error_text = await response.text()
                        self.update_health_metrics(model_name, False, response_time_ms)
                        
                        return {
                            "model": model_name,
                            "response": None,
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "timestamp": time.time(),
                            "status_code": response.status,
                            "processing_time_ms": response_time_ms
                        }
                
            except asyncio.TimeoutError:
                response_time_ms = (time.time() - start_time) * 1000
                self.update_health_metrics(model_name, False, response_time_ms)
                
                return {
                    "model": model_name,
                    "response": None,
                    "success": False,
                    "error": "Request timeout",
                    "timestamp": time.time(),
                    "processing_time_ms": response_time_ms
                }
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                self.update_health_metrics(model_name, False, response_time_ms)
                
                return {
                    "model": model_name,
                    "response": None,
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time(),
                    "processing_time_ms": response_time_ms
                }
    
    def update_health_metrics(self, model_name: str, success: bool, response_time_ms: float):
        """Update health metrics for a model"""
        if model_name not in self.model_health:
            return
        
        health = self.model_health[model_name]
        health.last_check = datetime.now()
        health.response_time_ms = response_time_ms
        
        if success:
            health.success_count += 1
            # Set to healthy if 3 consecutive successes
            if health.error_count < 3:
                health.is_healthy = True
        else:
            health.error_count += 1
            # Set to unhealthy if 3 consecutive failures
            if health.error_count >= 3:
                health.is_healthy = False
        
        # Calculate uptime percentage
        total_requests = health.success_count + health.error_count
        if total_requests > 0:
            health.uptime_percentage = (health.success_count / total_requests) * 100
    
    async def run_specialist_model(self, model_type: ModelType, content: str, 
                                  system_prompt: str, metadata: Dict[str, Any] = None) -> ModelResponse:
        """Run a specialist model with appropriate system prompt"""
        model_name = model_type.value
        
        if not content:
            return ModelResponse(
                model_name=model_name,
                model_type=model_type,
                content="",
                metadata=metadata or {},
                processing_time_ms=0,
                success=False,
                error_message="Empty content provided"
            )
        
        # Create messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        # Query the model
        start_time = time.time()
        result = await self.query_model(model_name, messages)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Process result
        if result["success"]:
            # Extract confidence score if provided by the model
            confidence_score = None
            try:
                # Attempt to parse JSON if model returned it
                response_text = result["response"]
                if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                    response_json = json.loads(response_text)
                    if "confidence" in response_json:
                        confidence_score = float(response_json["confidence"])
            except:
                pass
            
            return ModelResponse(
                model_name=model_name,
                model_type=model_type,
                content=result["response"],
                metadata=metadata or {},
                processing_time_ms=processing_time_ms,
                success=True,
                confidence_score=confidence_score
            )
        else:
            return ModelResponse(
                model_name=model_name,
                model_type=model_type,
                content="",
                metadata=metadata or {},
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=result.get("error", "Unknown error")
            )
    
    async def run_triage(self, content: str, metadata: Dict[str, Any] = None) -> ModelResponse:
        """Run the triage model to classify and route content"""
        system_prompt = """You are a financial content classification specialist. Analyze the provided text and categorize it.
Return a JSON object with the following fields:
- category: The primary category (earnings_report, market_news, analysis, opinion, etc.)
- tickers: Array of ticker symbols mentioned (e.g. ["AAPL", "MSFT"])
- sentiment: Overall sentiment (bullish, bearish, neutral)
- confidence: Your confidence in this classification (0.0 to 1.0)
- key_points: Array of 1-3 main points from the content
- specialist_needed: Which specialist should handle this (sentiment, extraction, or general)

Respond ONLY with valid JSON. Include no explanations or additional text."""
        
        return await self.run_specialist_model(
            ModelType.TRIAGE, 
            content, 
            system_prompt, 
            metadata
        )
    
    async def run_sentiment_analysis(self, content: str, metadata: Dict[str, Any] = None) -> ModelResponse:
        """Run the sentiment analysis model on financial content"""
        system_prompt = """You are a financial sentiment analysis specialist. Analyze the provided financial text for sentiment.
Return a JSON object with the following fields:
- overall_sentiment: The dominant sentiment (bullish, bearish, neutral)
- sentiment_score: A numeric score from -1.0 (bearish) to 1.0 (bullish)
- confidence: Your confidence in this analysis (0.0 to 1.0)
- key_factors: Array of factors driving the sentiment
- outlook: Short-term market outlook based on this content
- tickers_sentiment: Object with ticker symbols as keys and sentiment scores as values

Respond ONLY with valid JSON. Include no explanations or additional text."""
        
        return await self.run_specialist_model(
            ModelType.SENTIMENT, 
            content, 
            system_prompt, 
            metadata
        )
    
    async def run_data_extraction(self, content: str, metadata: Dict[str, Any] = None) -> ModelResponse:
        """Run the data extraction model on financial content"""
        system_prompt = """You are a financial data extraction specialist. Extract structured data from the provided financial text.
Return a JSON object with the following fields:
- entities: Array of companies, people, and organizations mentioned
- metrics: Object containing financial metrics (revenue, EPS, etc.)
- dates: Important dates mentioned (earnings dates, report periods)
- comparisons: Year-over-year or quarter-over-quarter comparisons
- projections: Forward-looking statements and guidance
- summary: A concise factual summary of the key information

Respond ONLY with valid JSON. Include no explanations or additional text."""
        
        return await self.run_specialist_model(
            ModelType.EXTRACTION, 
            content, 
            system_prompt, 
            metadata
        )
    
    async def run_general_analysis(self, content: str, query: str = None, use_rag: bool = True, 
                                 metadata: Dict[str, Any] = None) -> ModelResponse:
        """Run the general finance model, with optional RAG"""
        start_time = time.time()
        system_prompt = """You are a financial analysis expert. Provide insightful analysis of the content provided.
Your response should be comprehensive, factual, and clearly structured.
Focus on implications for investors, market impact, and wider economic context.
If specific questions are asked, answer them directly and thoroughly."""

        # Prepare RAG context and enhanced content
        try:
            rag_context = ""
            articles_count = 0

            if query and use_rag:
                rag_context, articles_count = self.rss_processor.create_context_from_articles(query)

            if metadata is None:
                metadata = {}

            if rag_context:
                enhanced_content = f"{rag_context}\n\nUser Content: {content}\n\nAnalyze the above information and respond to: {query if query else 'Provide a comprehensive analysis.'}"
            else:
                enhanced_content = content

            # Record how many articles were used for RAG
            metadata["articles_processed"] = articles_count

            # Run the general model with a timeout
            try:
                async with asyncio.timeout(25):  # 25 second timeout for model response
                    response = await self.run_specialist_model(
                        ModelType.GENERAL,
                        enhanced_content,
                        system_prompt,
                        metadata
                    )
                    return response
            except asyncio.TimeoutError:
                processing_time = (time.time() - start_time) * 1000
                logger.error("General analysis timed out after 25 seconds")
                return ModelResponse(
                    model_name=ModelType.GENERAL.value,
                    model_type=ModelType.GENERAL,
                    content="",
                    metadata=metadata or {},
                    processing_time_ms=processing_time,
                    success=False,
                    error_message="Analysis timed out. Please try again."
                )
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error(f"General analysis failed: {str(e)}")
                return ModelResponse(
                    model_name=ModelType.GENERAL.value,
                    model_type=ModelType.GENERAL,
                    content="",
                    metadata=metadata or {},
                    processing_time_ms=processing_time,
                    success=False,
                    error_message=f"Analysis failed: {str(e)}"
                )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error preparing general analysis: {str(e)}")
            return ModelResponse(
                model_name=ModelType.GENERAL.value,
                model_type=ModelType.GENERAL,
                content="",
                metadata=metadata or {},
                processing_time_ms=processing_time,
                success=False,
                error_message=f"Error preparing analysis: {str(e)}"
            )
    
    async def get_model_health(self) -> Dict[str, Dict]:
        """Get health status for all models"""
        return {name: asdict(health) for name, health in self.model_health.items()}

# Compatibility function to provide same interface as OllamaMultiLLMManager
def get_openrouter_llm_manager(api_key: str = None):
    """Factory function to create and return a new manager instance"""
    return OpenRouterMultiLLMManager(api_key=api_key)