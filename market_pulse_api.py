"""
Market Pulse API - FastAPI Implementation using OpenRouter

This module provides the FastAPI endpoints for the Market Pulse financial insights engine.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import time

# Import our OpenRouter integration
from openrouter_assembly_line import OpenRouterAssemblyLine, get_openrouter_assembly_line

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class SeismographDataPoint(BaseModel):
    timestamp: datetime
    sentiment_score: float  # -1.0 to 1.0
    volume: int           # number of articles
    peak_intensity: float  # 0.0 to 1.0 (same as confidence)
    tickers: List[str]
    
    # Optional fields
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    market_impact: Optional[str] = None
    processing_time_ms: Optional[float] = None

class TremorDetail(BaseModel):
    id: str
    timestamp: datetime
    sentiment_score: float
    confidence: float
    impact_level: str
    title: str
    summary: str
    tickers: List[str]
    source: str
    quality_grade: str
    
class UserQuery(BaseModel):
    query: str
    use_rag: bool = True
    
class HealthStatus(BaseModel):
    status: str
    models: Dict[str, Any]
    uptime_seconds: float
    processed_articles: int
    processed_queries: int

# Global state for the API
class APIState:
    def __init__(self):
        self.assembly_line: Optional[OpenRouterAssemblyLine] = None
        self.start_time = time.time()
        self.processed_articles = 0
        self.processed_queries = 0
        self.recent_data_points: List[Dict[str, Any]] = []
        self.article_cache: Dict[str, Dict[str, Any]] = {}

# Initialize API state
api_state = APIState()

# Create FastAPI app
app = FastAPI(
    title="Market Pulse API",
    description="Financial insights engine for real-time market analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get assembly line
async def get_assembly_line() -> OpenRouterAssemblyLine:
    if api_state.assembly_line is None:
        # Get API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("No OPENROUTER_API_KEY found in environment. Using default key.")
        
        # Create assembly line
        api_state.assembly_line = get_openrouter_assembly_line(api_key)
    
    return api_state.assembly_line

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Market Pulse API",
        "version": "1.0.0",
        "description": "Financial insights engine powered by OpenRouter"
    }

@app.get("/api/v1/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint with detailed model status"""
    try:
        # Get current assembly line if exists
        assembly_line = api_state.assembly_line

        # Basic health data
        health_data = {
            "status": "healthy",
            "models": {},
            "uptime_seconds": time.time() - api_state.start_time,
            "processed_articles": api_state.processed_articles,
            "processed_queries": api_state.processed_queries
        }
        
        # Add model health if available; try to create assembly line if missing
        if not assembly_line:
            try:
                assembly_line = await get_assembly_line()
            except Exception as e:
                logger.warning(f"Failed to initialize assembly line during health check: {e}")

        if assembly_line:
            try:
                health_data["models"] = await assembly_line.get_health_status()
            except Exception as e:
                logger.warning(f"Could not get model health: {e}")
                health_data["status"] = "degraded"
                health_data["models"] = {"error": str(e)}
        else:
            # Assembly line still not available
            health_data["status"] = "degraded"
            health_data["models"] = {"warning": "OpenRouter assembly line not initialized"}
            
        return health_data

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "models": {},
            "uptime_seconds": time.time() - api_state.start_time,
            "processed_articles": api_state.processed_articles,
            "processed_queries": api_state.processed_queries,
            "error": str(e)
        }

@app.get("/api/v1/pulse/timeline")
async def get_seismograph_timeline(
    hours: int = Query(24, ge=1, le=168),
    tickers: Optional[str] = None,
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
    assembly_line: OpenRouterAssemblyLine = Depends(get_assembly_line)
):
    """
    Get time-series sentiment and event data for the Seismograph chart
    
    Args:
        hours: Number of hours of data to return
        tickers: Comma-separated list of ticker symbols to filter by
        min_confidence: Minimum confidence threshold for including events
    """
    # Simulate data points if we don't have enough real ones
    # In production, these would come from a database
    if len(api_state.recent_data_points) < 10:
        _generate_sample_data_points()
    
    # Filter by tickers if provided
    ticker_list = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    # Filter data points
    now = datetime.now()
    cutoff_time = now - timedelta(hours=hours)
    
    filtered_points = []
    for point in api_state.recent_data_points:
        point_time = point["timestamp"]
        if isinstance(point_time, str):
            point_time = datetime.fromisoformat(point_time.replace("Z", "+00:00"))
        
        if point_time < cutoff_time:
            continue
            
        if ticker_list and not any(ticker in point["tickers"] for ticker in ticker_list):
            continue
            
        if point.get("confidence", 0) < min_confidence:
            continue
            
        filtered_points.append(point)
    
    # Return the filtered points
    return {
        "data_points": filtered_points,
        "tickers": ticker_list,
        "hours": hours,
        "total_points": len(filtered_points)
    }

@app.get("/api/v1/insights/event/{event_id}")
async def get_event_details(
    event_id: str,
    assembly_line: OpenRouterAssemblyLine = Depends(get_assembly_line)
):
    """
    Get detailed insights for a specific event
    
    Args:
        event_id: The ID of the event to retrieve
    """
    # In production, this would fetch from a database
    if event_id in api_state.article_cache:
        return api_state.article_cache[event_id]
    
    # Return 404 if not found
    raise HTTPException(status_code=404, detail=f"Event {event_id} not found")


@app.get("/api/v1/news/headlines")
async def get_live_headlines(limit_per_feed: int = 5, assembly_line: OpenRouterAssemblyLine = Depends(get_assembly_line)):
    """
    Aggregate latest headlines from configured RSS feeds.

    Query parameters:
      - limit_per_feed: Number of articles to fetch per feed (default 5)
    """
    try:
        # Ensure assembly line (and its llm_manager) is initialized
        if assembly_line is None:
            assembly_line = await get_assembly_line()

        # Use the rss_processor attached to the llm manager
        rss_proc = None
        try:
            rss_proc = assembly_line.llm_manager.rss_processor
        except Exception:
            # Fallback: attempt to access directly if assembly line exposes it differently
            rss_proc = getattr(assembly_line, 'rss_processor', None)

        if rss_proc is None:
            raise HTTPException(status_code=503, detail="RSS processor not available")

        # Fetch and return headlines
        articles = rss_proc.fetch_rss_content(max_articles_per_feed=limit_per_feed)
        # Compact the articles to headlines only
        headlines = [
            {
                "title": a.get('title'),
                "link": a.get('link'),
                "published": a.get('published'),
                "source": a.get('source'),
                "summary": a.get('summary')
            }
            for a in articles
        ]

        return {"headlines": headlines, "count": len(headlines)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch live headlines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/insights/query")
async def process_user_query(
    query: UserQuery,
    assembly_line: OpenRouterAssemblyLine = Depends(get_assembly_line)
):
    """
    Process a user query through the OpenRouter pipeline
    
    Args:
        query: The user's financial question and options
    """
    try:
        # Process the query
        result = await assembly_line.process_user_query(
            query.query,
            use_rag=query.use_rag
        )
        
        # Increment processed queries counter
        api_state.processed_queries += 1
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/api/v1/insights/process-article")
async def process_article(
    article_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    assembly_line: OpenRouterAssemblyLine = Depends(get_assembly_line)
):
    """
    Process a financial article through the assembly line
    
    Args:
        article_data: Article content and metadata
    """
    # Generate a unique ID if not provided
    if "id" not in article_data:
        article_data["id"] = f"article-{int(time.time())}-{hash(article_data.get('title', ''))}"
    
    # Process in the background
    background_tasks.add_task(_process_article_background, article_data)
    
    return {
        "message": "Article processing started",
        "article_id": article_data["id"]
    }

async def _process_article_background(article_data: Dict[str, Any]):
    """Background task to process an article"""
    try:
        assembly_line = await get_assembly_line()
        
        # Process the article
        result = await assembly_line.process_article(article_data)
        
        if result.get("success", False):
            # Store in cache
            api_state.article_cache[result["article_id"]] = result
            
            # Create seismograph data point
            data_point = {
                "id": result["article_id"],
                "timestamp": datetime.now(),
                "sentiment_score": result.get("sentiment_score", 0.0),
                "volume": 1,
                "peak_intensity": result.get("specialist", {}).get("data", {}).get("confidence", 0.7),
                "tickers": result.get("tickers", []),
                "confidence": result.get("specialist", {}).get("data", {}).get("confidence", 0.7),
                "quality_score": 0.8,  # Default value
                "title": result.get("title", ""),
                "processing_time_ms": result.get("processing_times", {}).get("total_ms", 0)
            }
            
            # Add to recent data points
            api_state.recent_data_points.append(data_point)
            
            # Limit the size of recent data points
            if len(api_state.recent_data_points) > 1000:
                api_state.recent_data_points = api_state.recent_data_points[-1000:]
                
            # Increment processed articles counter
            api_state.processed_articles += 1
            
        else:
            logger.error(f"Article processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in background article processing: {str(e)}")

def _generate_sample_data_points():
    """Generate sample data points for demonstration"""
    import random
    
    now = datetime.now()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    sources = ["Bloomberg", "CNBC", "Reuters", "Financial Times", "Wall Street Journal"]
    
    for i in range(50):
        hours_ago = random.randint(0, 24)
        sentiment = random.uniform(-0.9, 0.9)
        confidence = random.uniform(0.6, 0.95)
        
        # More positive/negative sentiment should have higher confidence
        if abs(sentiment) > 0.5:
            confidence = min(0.95, confidence + 0.1)
        
        # Random tickers, 1-3 per event
        num_tickers = random.randint(1, 3)
        event_tickers = random.sample(tickers, num_tickers)
        
        # Create data point
        data_point = {
            "id": f"sample-{i}",
            "timestamp": now - timedelta(hours=hours_ago),
            "sentiment_score": sentiment,
            "volume": random.randint(1, 5),
            "peak_intensity": confidence,
            "tickers": event_tickers,
            "confidence": confidence,
            "quality_score": random.uniform(0.7, 0.95),
            "title": f"Sample Financial Event {i}",
            "market_impact": random.choice(["High", "Medium", "Low"]),
            "source": random.choice(sources),
            "processing_time_ms": random.uniform(1000, 5000)
        }
        
        api_state.recent_data_points.append(data_point)
        
        # Add to article cache for event details
        api_state.article_cache[f"sample-{i}"] = {
            "id": f"sample-{i}",
            "title": f"Sample Financial Event {i}",
            "source": random.choice(sources),
            "url": "https://example.com/article",
            "published": (now - timedelta(hours=hours_ago)).isoformat(),
            "content": "This is sample article content for demonstration purposes.",
            "sentiment_score": sentiment,
            "tickers": event_tickers,
            "analysis": "Sample analysis of the financial event.",
            "market_impact": random.choice(["High", "Medium", "Low"]),
            "confidence": confidence
        }

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    logger.info("Starting Market Pulse API with OpenRouter integration")
    
    # Generate sample data
    _generate_sample_data_points()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Market Pulse API")
    
    if api_state.assembly_line is not None:
        await api_state.assembly_line.close()

# Run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("market_pulse_api:app", host="0.0.0.0", port=8000, reload=True)