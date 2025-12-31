"""
Production API Endpoints for Financial Seismograph Interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncpg
from contextlib import asynccontextmanager

from config_manager import get_config
from fast_smart_integration import FastSmartIntegration
from assembly_line_orchestrator import AssemblyLineOrchestrator
from ollama_multi_llm_manager import OllamaMultiLLMManager

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SeismographDataPoint(BaseModel):
    timestamp: datetime
    sentiment_score: float  # -1.0 to 1.0
    volume: int           # number of articles
    peak_intensity: float  # 0.0 to 1.0 (same as confidence for now)
    tickers: List[str]
    
    # Optional fields for backward compatibility
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

class Article(BaseModel):
    id: int
    title: str
    content: str
    url: str
    published_at: datetime
    word_count: int
    rss_feed_name: Optional[str] = None
    is_scraped: Optional[bool] = None
    scraped_at: Optional[datetime] = None
    analysis_sentiment: Optional[float] = None
    analysis_confidence: Optional[float] = None

class EpicenterReport(BaseModel):
    tremor_id: str
    title: str
    executive_summary: str
    sentiment_analysis: Dict[str, Any]
    financial_metrics: List[Dict[str, Any]]
    market_impact: str
    key_insights: List[str]
    investment_implications: List[str]
    risk_factors: List[str]
    pipeline_trace: List[Dict[str, str]]
    quality_score: float
    processing_time_ms: float

class UserQuery(BaseModel):
    query: str
    context: Optional[str] = None
    tickers: Optional[List[str]] = None

class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    related_articles: List[str]
    processing_time_ms: float

# Global application state
class AppState:
    def __init__(self):
        self.integration: Optional[FastSmartIntegration] = None
        self.orchestrator: Optional[AssemblyLineOrchestrator] = None
        self.db_pool = None  # Will be either asyncpg.Pool or sqlite connection manager
        self.seismograph_data: List[SeismographDataPoint] = []
        self.config = get_config()

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Financial Seismograph API")
    
    try:
        # Initialize database connection
        db_config = app_state.config.database_config
        
        if db_config.type == "sqlite":
            # For SQLite, we'll use a simple connection approach
            import sqlite3
            import os
            
            # Ensure directory exists
            db_dir = os.path.dirname(db_config.path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Store the path for later use
            app_state.db_config = {'type': 'sqlite', 'path': db_config.path}
            app_state.db_pool = "sqlite_ready"  # Placeholder to indicate ready
            
        elif db_config.type == "postgresql":
            app_state.db_pool = await asyncpg.create_pool(
                host=db_config.host,
                port=db_config.port,
                database=db_config.name,
                user=db_config.user,
                password=db_config.password,
                min_size=db_config.pool['min_connections'],
                max_size=db_config.pool['max_connections']
            )
            app_state.db_config = {
                'type': 'postgresql',
                'host': db_config.host,
                'port': db_config.port,
                'database': db_config.name,
                'user': db_config.user,
                'password': db_config.password
            }
        else:
            raise ValueError(f"Unsupported database type: {db_config.type}")
        
        # Initialize AI integration
        app_state.integration = FastSmartIntegration(db_config=app_state.db_config)
        
        await app_state.integration.initialize()
        
        # Start background processing
        asyncio.create_task(background_processing())
        asyncio.create_task(update_seismograph_data())
        
        logger.info("✅ Financial Seismograph API ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Financial Seismograph API")
    if app_state.integration:
        await app_state.integration.cleanup()
    if app_state.db_pool and app_state.db_config['type'] == 'postgresql':
        await app_state.db_pool.close()

# Create FastAPI app
app = FastAPI(
    title="Financial Seismograph API",
    description="Real-time financial sentiment analysis and intelligence",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
api_config = get_config().api_config
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database connection or demo mode indicator
async def get_db():
    if app_state.db_config['type'] == 'sqlite':
        # For SQLite demo mode, we'll return a simple indicator
        yield "demo_mode"
    else:
        # PostgreSQL mode
        async with app_state.db_pool.acquire() as connection:
            yield connection

@app.get("/api/seismograph/data")
async def get_seismograph_data(
    hours: int = Query(48, ge=1, le=168),  # Default to 48 hours, up to 1 week
    tickers: Optional[str] = Query(None, description="Comma-separated ticker symbols")
) -> List[SeismographDataPoint]:
    """Get seismograph data for the chart"""
    
    try:
        if app_state.db_config['type'] == 'sqlite':
            # Use real data from SQLite database
            from datetime import datetime, timedelta
            import sqlite3
            
            # Calculate time range
            start_time = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(app_state.db_config['path'])
            cursor = conn.cursor()
            
            # Query real data from articles and analysis_results tables
            # Only include articles that have analysis results for meaningful data
            query = """
                SELECT 
                    datetime(a.published_at) as timestamp,
                    AVG(ar.sentiment_confidence) as sentiment_score,
                    COUNT(*) as volume,
                    MAX(ar.sentiment_confidence) as peak_intensity,
                    GROUP_CONCAT(DISTINCT a.title) as article_titles,
                    GROUP_CONCAT(DISTINCT ar.sentiment) as sentiments
                FROM articles a
                INNER JOIN analysis_results ar ON ar.article_id = a.id
                WHERE a.published_at >= ?
                GROUP BY strftime('%Y-%m-%d %H:%M', a.published_at)
                ORDER BY timestamp DESC
                LIMIT 100
            """
            
            cursor.execute(query, (start_time.isoformat(),))
            rows = cursor.fetchall()
            conn.close()
            
            data_points = []
            for row in rows:
                # Parse timestamp
                timestamp = datetime.fromisoformat(row[0]) if row[0] else datetime.now()
                
                # Extract potential tickers from titles (simple extraction)
                tickers = []
                if row[4]:  # article_titles
                    # Look for common ticker patterns in titles
                    import re
                    ticker_pattern = r'\b[A-Z]{2,5}\b'
                    potential_tickers = re.findall(ticker_pattern, row[4])
                    tickers = list(set([t for t in potential_tickers if len(t) <= 5]))[:5]  # Limit to 5 tickers
                
                # Get sentiment info
                sentiments = row[5].split(',') if row[5] else []
                avg_sentiment_score = float(row[1] or 0.0)
                
                data_points.append(SeismographDataPoint(
                    timestamp=timestamp,
                    sentiment_score=avg_sentiment_score,
                    volume=int(row[2] or 0),
                    peak_intensity=float(row[3] or 0.0),
                    tickers=tickers,
                    confidence=avg_sentiment_score
                ))
            
            return data_points
            
        else:
            # PostgreSQL mode - original database query
            async with app_state.db_pool.acquire() as db:
                # Build query based on parameters
                where_conditions = ["ar.created_at >= $1"]
                params = [datetime.now() - timedelta(hours=hours)]
                
                if tickers:
                    ticker_list = [t.strip().upper() for t in tickers.split(',')]
                    where_conditions.append("ar.tickers && $2")
                    params.append(ticker_list)
                
                query = f"""
                    SELECT 
                        DATE_TRUNC('minute', ar.created_at) as timestamp,
                        AVG(CASE 
                            WHEN ar.sentiment = 'bullish' THEN ar.sentiment_confidence
                            WHEN ar.sentiment = 'bearish' THEN -ar.sentiment_confidence
                            ELSE 0
                        END) as sentiment_score,
                        AVG(ar.sentiment_confidence) as confidence,
                        COUNT(*) as volume,
                        MAX(ar.sentiment_confidence) as peak_intensity,
                        ARRAY_AGG(DISTINCT unnest(ar.tickers)) FILTER (WHERE ar.tickers IS NOT NULL) as tickers
                    FROM analysis_results ar
                    JOIN articles a ON ar.article_id = a.id
                    WHERE {' AND '.join(where_conditions)}
                    GROUP BY DATE_TRUNC('minute', ar.created_at)
                    ORDER BY timestamp DESC
                    LIMIT 1440
                """
                
                rows = await db.fetch(query, *params)
                
                data_points = []
                for row in rows:
                    data_points.append(SeismographDataPoint(
                        timestamp=row['timestamp'],
                        sentiment_score=float(row['sentiment_score'] or 0),
                        confidence=float(row['confidence'] or 0),
                        volume=int(row['volume']),
                        peak_intensity=float(row['peak_intensity'] or 0),
                        tickers=row['tickers'] or []
                    ))
                
                return data_points
            
    except Exception as e:
        logger.error(f"❌ Failed to get seismograph data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve seismograph data")

@app.get("/api/tremors")
async def get_tremors(
    hours: int = Query(24, ge=1, le=168),
    min_intensity: float = Query(0.6, ge=0.0, le=1.0),
    tickers: Optional[str] = Query(None)
) -> List[TremorDetail]:
    """Get detected tremors (significant sentiment events)"""
    
    try:
        if app_state.db_config['type'] == 'sqlite':
            # Use real data from SQLite database
            import sqlite3
            import json
            
            # Calculate time range
            start_time = datetime.now() - timedelta(hours=hours)
            
            conn = sqlite3.connect(app_state.db_config['path'])
            cursor = conn.cursor()
            
            # Build the query with proper column names
            where_conditions = [
                "ar.created_at >= ?",
                "ar.sentiment_confidence >= ?",
                "(ar.sentiment = 'bullish' OR ar.sentiment = 'bearish')"
            ]
            params = [start_time.isoformat(), min_intensity]
            
            if tickers:
                ticker_list = [t.strip().upper() for t in tickers.split(',')]
                # For SQLite, we'll do a simple text search in the tickers JSON
                ticker_conditions = []
                for ticker in ticker_list:
                    ticker_conditions.append("ar.tickers LIKE ?")
                    params.append(f'%{ticker}%')
                
                if ticker_conditions:
                    where_conditions.append(f"({' OR '.join(ticker_conditions)})")
            
            query = f"""
                SELECT 
                    ar.id,
                    ar.created_at as timestamp,
                    CASE 
                        WHEN ar.sentiment = 'bullish' THEN ar.sentiment_confidence
                        WHEN ar.sentiment = 'bearish' THEN -ar.sentiment_confidence
                        ELSE 0
                    END as sentiment_score,
                    ar.sentiment_confidence as confidence,
                    ar.market_impact as impact_level,
                    a.title,
                    ar.executive_summary as summary,
                    ar.tickers,
                    a.url as source,
                    ar.quality_grade
                FROM analysis_results ar
                JOIN articles a ON ar.article_id = a.id
                WHERE {' AND '.join(where_conditions)}
                ORDER BY ar.sentiment_confidence DESC, ar.created_at DESC
                LIMIT 50
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            tremors = []
            for row in rows:
                # Parse tickers JSON safely
                try:
                    tickers_data = json.loads(row[7]) if row[7] else []
                except (json.JSONDecodeError, TypeError):
                    tickers_data = []
                
                tremors.append(TremorDetail(
                    id=str(row[0]),
                    timestamp=datetime.fromisoformat(row[1]),
                    sentiment_score=float(row[2]),
                    confidence=float(row[3]),
                    impact_level=row[4] or 'NEUTRAL',
                    title=row[5],
                    summary=row[6] or '',
                    tickers=tickers_data,
                    source=row[8],
                    quality_grade=row[9] or 'unknown'
                ))
            
            conn.close()
            return tremors
            
        else:
            # PostgreSQL mode
            async with app_state.db_pool.acquire() as db:
                where_conditions = [
                    "ar.created_at >= $1",
                    "ar.sentiment_confidence >= $2",
                    "(ar.sentiment = 'bullish' OR ar.sentiment = 'bearish')"
                ]
                params = [datetime.now() - timedelta(hours=hours), min_intensity]
                
                if tickers:
                    ticker_list = [t.strip().upper() for t in tickers.split(',')]
                    where_conditions.append("ar.tickers && $3")
                    params.append(ticker_list)
            
            query = f"""
                SELECT 
                    ar.id,
                    ar.created_at as timestamp,
                    CASE 
                        WHEN ar.sentiment = 'bullish' THEN ar.sentiment_confidence
                        WHEN ar.sentiment = 'bearish' THEN -ar.sentiment_confidence
                        ELSE 0
                    END as sentiment_score,
                    ar.sentiment_confidence as confidence,
                    ar.market_impact as impact_level,
                    a.title,
                    ar.executive_summary as summary,
                    ar.tickers,
                    a.url as source,
                    ar.quality_grade
                FROM analysis_results ar
                JOIN articles a ON ar.article_id = a.id
                WHERE {' AND '.join(where_conditions)}
                ORDER BY ar.sentiment_confidence DESC, ar.created_at DESC
                LIMIT 50
            """
            
            rows = await db.fetch(query, *params)
            
            tremors = []
            for row in rows:
                tremors.append(TremorDetail(
                    id=str(row['id']),
                    timestamp=row['timestamp'],
                    sentiment_score=float(row['sentiment_score']),
                    confidence=float(row['confidence']),
                    impact_level=row['impact_level'],
                    title=row['title'],
                    summary=row['summary'] or '',
                    tickers=row['tickers'] or [],
                    source=row['source'],
                    quality_grade=row['quality_grade']
                ))
            
            return tremors
            
    except Exception as e:
        logger.error(f"❌ Failed to get tremors: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tremors")

@app.get("/api/epicenter/{tremor_id}")
async def get_epicenter_report(tremor_id: str) -> EpicenterReport:
    """Get detailed epicenter report for a specific tremor"""
    
    try:
        if app_state.db_config['type'] == 'sqlite':
            # Use real data from SQLite database
            import sqlite3
            import json
            
            conn = sqlite3.connect(app_state.db_config['path'])
            cursor = conn.cursor()
            
            # Get analysis result details
            query = """
                SELECT 
                    ar.*,
                    a.title,
                    a.content,
                    a.url as source
                FROM analysis_results ar
                JOIN articles a ON ar.article_id = a.id
                WHERE ar.id = ?
            """
            
            cursor.execute(query, (int(tremor_id),))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                raise HTTPException(status_code=404, detail="Tremor not found")
            
            # Map the row to column names (based on our schema analysis)
            # analysis_results columns: id, article_id, content_type, sentiment, sentiment_confidence, 
            # market_impact, quality_grade, quality_score, executive_summary, key_insights, 
            # tickers, processing_time_ms, model_versions, created_at
            # Plus joined article data: title, content, source
            
            result_data = {
                'id': row[0],
                'article_id': row[1], 
                'content_type': row[2],
                'sentiment': row[3],
                'sentiment_confidence': row[4],
                'market_impact': row[5],
                'quality_grade': row[6],
                'quality_score': row[7],
                'executive_summary': row[8],
                'key_insights': row[9],
                'tickers': row[10],
                'processing_time_ms': row[11],
                'model_versions': row[12],
                'created_at': row[13],
                'title': row[14],
                'content': row[15],
                'source': row[16]
            }
            
            # Generate comprehensive report using real data
            if result_data['quality_grade'] in ['excellent', 'good']:
                # Create pipeline trace from model_versions
                pipeline_trace = []
                try:
                    model_versions = json.loads(result_data['model_versions']) if result_data['model_versions'] else {}
                    for stage, model in model_versions.items():
                        pipeline_trace.append({
                            "stage": stage.title(),
                            "model": model,
                            "result": "completed"
                        })
                except (json.JSONDecodeError, TypeError):
                    pipeline_trace = [
                        {"stage": "Analysis", "model": "SQLite-LLM", "result": result_data['quality_grade']}
                    ]
                
                # Parse financial metrics from key_insights
                financial_metrics = []
                key_insights_list = []
                if result_data['key_insights']:
                    try:
                        insights_data = json.loads(result_data['key_insights'])
                        if isinstance(insights_data, list):
                            key_insights_list = insights_data
                            for insight in insights_data[:5]:  # Top 5 insights as metrics
                                financial_metrics.append({
                                    "metric": "Key Insight",
                                    "value": str(insight),
                                    "confidence": "high"
                                })
                    except (json.JSONDecodeError, TypeError):
                        # If not valid JSON, treat as plain text
                        key_insights_list = [result_data['key_insights']]
                        financial_metrics.append({
                            "metric": "Analysis Insight",
                            "value": result_data['key_insights'],
                            "confidence": "medium"
                        })
                
                # Parse tickers
                tickers_list = []
                if result_data['tickers']:
                    try:
                        tickers_list = json.loads(result_data['tickers'])
                    except (json.JSONDecodeError, TypeError):
                        tickers_list = []
                
                return EpicenterReport(
                    tremor_id=tremor_id,
                    title=result_data['title'],
                    executive_summary=result_data['executive_summary'] or 'Analysis available for this financial event.',
                    sentiment_analysis={
                        "primary_sentiment": result_data['sentiment'] or 'neutral',
                        "confidence": result_data['sentiment_confidence'] or 0.5,
                        "market_impact": result_data['market_impact'] or 'NEUTRAL'
                    },
                    financial_metrics=financial_metrics,
                    market_impact=result_data['market_impact'] or 'NEUTRAL',
                    key_insights=key_insights_list,
                    investment_implications=["Review detailed analysis", "Monitor related developments"],
                    risk_factors=["Market volatility", "Information accuracy"],
                    pipeline_trace=pipeline_trace,
                    quality_score=result_data['quality_score'] or 0.5,
                    processing_time_ms=result_data['processing_time_ms'] or 0.0
                )
            else:
                # For lower quality grades, return a basic report
                return EpicenterReport(
                    tremor_id=tremor_id,
                    title=result_data['title'],
                    executive_summary="Limited analysis available for this financial event.",
                    sentiment_analysis={
                        "primary_sentiment": result_data['sentiment'] or 'neutral',
                        "confidence": result_data['sentiment_confidence'] or 0.3,
                        "market_impact": "NEUTRAL"
                    },
                    financial_metrics=[],
                    market_impact="NEUTRAL",
                    key_insights=[],
                    investment_implications=["Insufficient data for recommendations"],
                    risk_factors=["Low analysis quality", "Limited data"],
                    pipeline_trace=[{"stage": "Analysis", "model": "Basic", "result": result_data['quality_grade']}],
                    quality_score=result_data['quality_score'] or 0.3,
                    processing_time_ms=result_data['processing_time_ms'] or 0.0
                )
            
        else:
            # PostgreSQL mode
            async with app_state.db_pool.acquire() as db:
                # Get analysis result details
                query = """
                    SELECT 
                        ar.*,
                        a.title,
                        a.content,
                        a.url as source
                    FROM analysis_results ar
                    JOIN articles a ON ar.article_id = a.id
                    WHERE ar.id = $1
                """
                
                row = await db.fetchrow(query, int(tremor_id))
                if not row:
                    raise HTTPException(status_code=404, detail="Tremor not found")
            
            # Generate comprehensive report using AI if needed
            if row['quality_grade'] in ['excellent', 'good']:
                # Create pipeline trace
                pipeline_trace = [
                    {"stage": "Triage", "model": "Adapt-Finance-Llama", "result": row['content_type']},
                    {"stage": "Sentiment", "model": "FinBERT", "result": f"{row['sentiment']} ({row['sentiment_confidence']:.2f})"},
                    {"stage": "Extraction", "model": "FinGPT", "result": f"{row['extracted_metrics_count']} metrics"},
                    {"stage": "Analysis", "model": "Assembly Line", "result": row['quality_grade']}
                ]
                
                # Parse financial metrics (stored as JSON)
                financial_metrics = []
                if row.get('key_insights'):
                    try:
                        insights_data = json.loads(row['key_insights'])
                        if isinstance(insights_data, list):
                            for insight in insights_data[:5]:  # Top 5 insights
                                financial_metrics.append({
                                    "metric": "Key Insight",
                                    "value": insight,
                                    "confidence": "high"
                                })
                    except json.JSONDecodeError:
                        pass
                
                return EpicenterReport(
                    tremor_id=tremor_id,
                    title=row['title'],
                    executive_summary=row['executive_summary'] or '',
                    sentiment_analysis={
                        "primary_sentiment": row['sentiment'],
                        "confidence": row['sentiment_confidence'],
                        "market_impact": row['market_impact']
                    },
                    financial_metrics=financial_metrics,
                    market_impact=row['market_impact'],
                    key_insights=json.loads(row['key_insights']) if row.get('key_insights') else [],
                    investment_implications=[],  # Would be populated by AI analysis
                    risk_factors=[],  # Would be populated by AI analysis
                    pipeline_trace=pipeline_trace,
                    quality_score=row['quality_score'],
                    processing_time_ms=row['processing_time_ms']
                )
            else:
                raise HTTPException(status_code=422, detail="Low quality analysis data")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get epicenter report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate epicenter report")

@app.post("/api/query")
async def process_user_query(query: UserQuery, background_tasks: BackgroundTasks) -> QueryResponse:
    """Process user query through complete AI pipeline"""
    
    try:
        start_time = datetime.now()
        
        # Process query through assembly line
        if not app_state.integration or not app_state.integration.orchestrator:
            raise HTTPException(status_code=503, detail="AI system not ready")
        
        result = await app_state.integration.orchestrator.process_content(
            title=f"User Query: {query.query[:50]}...",
            content=f"User Query: {query.query}\nContext: {query.context or 'None'}",
            source="user_query",
            metadata={
                "query_type": "user_interactive",
                "tickers": query.tickers or []
            }
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Generate response based on analysis
        response_text = ""
        confidence = 0.5
        sources = []
        related_articles = []
        
        if result.success and result.aggregated_insights:
            insights = result.aggregated_insights
            response_text = insights.executive_summary
            confidence = result.overall_quality_score
            
            # Get related articles from database
            if query.tickers:
                background_tasks.add_task(find_related_articles, query.tickers)
        
        if not response_text:
            response_text = "I apologize, but I couldn't generate a comprehensive response to your query at this time."
            confidence = 0.2
        
        return QueryResponse(
            response=response_text,
            confidence=confidence,
            sources=sources,
            related_articles=related_articles,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to process user query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")

@app.get("/api/articles", response_model=List[Article])
async def get_articles(
    limit: int = Query(default=50, ge=1, le=1000, description="Number of articles to return"),
    offset: int = Query(default=0, ge=0, description="Number of articles to skip"),
    rss_feed: Optional[str] = Query(default=None, description="Filter by RSS feed name")
):
    """Get articles from the database"""
    try:
        import sqlite3
        
        conn = sqlite3.connect(app_state.db_config['path'])
        cursor = conn.cursor()
        
        # Build query with optional RSS feed filter
        base_query = """
            SELECT 
                a.id,
                a.title,
                a.content,
                a.url,
                a.published_at,
                a.word_count,
                rf.name as rss_feed_name,
                COALESCE(ar.sentiment_confidence, 0.0) as sentiment_score,
                COALESCE(ar.sentiment_confidence, 0.0) as confidence
            FROM articles a
            LEFT JOIN rss_feeds rf ON a.rss_feed_id = rf.id
            LEFT JOIN analysis_results ar ON a.id = ar.article_id
        """
        
        params = []
        if rss_feed:
            base_query += " WHERE rf.name LIKE ?"
            params.append(f"%{rss_feed}%")
        
        base_query += " ORDER BY a.published_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        
        articles = []
        for row in rows:
            articles.append(Article(
                id=row[0],
                title=row[1] or "Untitled",
                content=row[2] or "",
                url=row[3] or "",
                published_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                word_count=row[5] or 0,
                rss_feed_name=row[6],
                analysis_sentiment=row[7],
                analysis_confidence=row[8]
            ))
        
        conn.close()
        
        logger.info(f"✅ Retrieved {len(articles)} articles (limit={limit}, offset={offset})")
        return articles
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve articles: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve articles")

@app.get("/api/articles/scraping/status")
async def get_scraping_status():
    """Get overall scraping status and statistics"""
    try:
        db_config = app_state.db_config
        
        if db_config['type'] == 'sqlite':
            conn = sqlite3.connect(db_config['path'])
            cursor = conn.cursor()
            
            # Get scraping statistics
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM articles WHERE is_scraped = TRUE")
            scraped_articles = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM articles WHERE is_scraped = FALSE OR is_scraped IS NULL")
            pending_scraping = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE scraped_at > datetime('now', '-1 hour')
            """)
            scraped_last_hour = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT AVG(word_count) FROM articles WHERE is_scraped = TRUE
            """)
            avg_scraped_word_count = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT AVG(word_count) FROM articles WHERE is_scraped = FALSE OR is_scraped IS NULL
            """)
            avg_rss_word_count = cursor.fetchone()[0] or 0
            
            conn.close()
            
            scraping_rate = (scraped_articles / total_articles * 100) if total_articles > 0 else 0
            
            return {
                "total_articles": total_articles,
                "scraped_articles": scraped_articles,
                "pending_scraping": pending_scraping,
                "scraping_rate_percent": round(scraping_rate, 2),
                "scraped_last_hour": scraped_last_hour,
                "content_enhancement": {
                    "avg_scraped_word_count": round(avg_scraped_word_count, 1),
                    "avg_rss_word_count": round(avg_rss_word_count, 1),
                    "content_improvement_ratio": round(avg_scraped_word_count / avg_rss_word_count, 2) if avg_rss_word_count > 0 else 0
                },
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to get scraping status: {e}")
        return {"error": "Failed to retrieve scraping status"}

@app.post("/api/articles/scraping/trigger")
async def trigger_scraping(article_ids: List[int] = None, force_rescrape: bool = False):
    """Manually trigger scraping for specific articles or all pending articles"""
    try:
        db_config = app_state.db_config
        
        if db_config['type'] == 'sqlite':
            conn = sqlite3.connect(db_config['path'])
            cursor = conn.cursor()
            
            # Build query based on parameters
            if article_ids:
                placeholders = ','.join(['?'] * len(article_ids))
                if force_rescrape:
                    query = f"SELECT id, url, title FROM articles WHERE id IN ({placeholders})"
                    params = article_ids
                else:
                    query = f"SELECT id, url, title FROM articles WHERE id IN ({placeholders}) AND (is_scraped = FALSE OR is_scraped IS NULL)"
                    params = article_ids
            else:
                if force_rescrape:
                    query = "SELECT id, url, title FROM articles"
                    params = []
                else:
                    query = "SELECT id, url, title FROM articles WHERE is_scraped = FALSE OR is_scraped IS NULL"
                    params = []
            
            cursor.execute(query, params)
            articles_to_scrape = cursor.fetchall()
            conn.close()
            
            # Trigger scraping for each article
            scraping_tasks = []
            for article_id, url, title in articles_to_scrape:
                # Import the scraping function from production_startup
                # For now, we'll create a simplified version here
                task = asyncio.create_task(_scrape_article_content_api(article_id, url, title, db_config))
                scraping_tasks.append(task)
            
            return {
                "message": f"Triggered scraping for {len(articles_to_scrape)} articles",
                "articles_queued": len(articles_to_scrape),
                "force_rescrape": force_rescrape,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to trigger scraping: {e}")
        return {"error": "Failed to trigger scraping"}

async def _scrape_article_content_api(article_id: int, url: str, title: str, db_config: dict):
    """API version of article scraping function"""
    try:
        # Import scraping libraries
        try:
            import trafilatura
            import aiohttp
            import re
            from readability import Document
        except ImportError:
            logger.warning("⚠️ Scraping libraries not available")
            return
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Fetch the webpage
            async with session.get(url) as response:
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
            extracted_text = re.sub(r'<[^>]+>', '', extracted_text)
        
        if extracted_text and len(extracted_text) > 100:
            # Update database with scraped content
            conn = sqlite3.connect(db_config['path'])
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
            
            logger.info(f"🕷️ API scraped content for: {title[:50]}... ({len(extracted_text)} chars)")
        else:
            logger.warning(f"⚠️ Failed to extract meaningful content from {url}")
            
    except Exception as e:
        logger.error(f"❌ Failed to scrape article {url}: {e}")

@app.get("/api/system/health")
async def system_health():
    """Get system health status"""
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "database": "unknown",
                "ai_integration": "unknown",
                "ollama": "unknown"
            },
            "metrics": {
                "active_connections": 0,
                "processing_queue": 0,
                "total_processed": 0
            }
        }
        
        # Check database
        try:
            if app_state.db_config['type'] == 'sqlite':
                # For SQLite, just check if we can create a connection
                import sqlite3
                conn = sqlite3.connect(app_state.db_config['path'])
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                conn.close()
                health_status["components"]["database"] = "healthy"
            else:
                # PostgreSQL
                async with app_state.db_pool.acquire() as db:
                    await db.fetchval("SELECT 1")
                health_status["components"]["database"] = "healthy"
        except:
            health_status["components"]["database"] = "unhealthy"
            health_status["status"] = "degraded"
        
        # Check AI integration
        if app_state.integration:
            health_status["components"]["ai_integration"] = "healthy"
            stats = await app_state.integration.get_integration_stats()
            health_status["metrics"]["total_processed"] = stats.get("articles_processed", 0)
        else:
            health_status["components"]["ai_integration"] = "unhealthy"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(),
            "error": str(e)
        }

@app.get("/api/system/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    
    try:
        stats = {}
        
        if app_state.integration:
            stats = await app_state.integration.get_integration_stats()
        
        # Add seismograph metrics
        stats["seismograph"] = {
            "data_points": len(app_state.seismograph_data),
            "last_update": datetime.now(),
            "active_feeds": len([f for f in app_state.config.rss_feeds if f.enabled])
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

# Background tasks
async def background_processing():
    """Background processing for continuous article analysis"""
    
    if not app_state.integration:
        logger.warning("⚠️ Integration not initialized for background processing")
        return
    
    try:
        # Start continuous processing with configuration settings
        processing_settings = app_state.config.processing_settings
        
        await app_state.integration.start_continuous_processing(
            poll_interval_seconds=30,
            max_articles_per_batch=processing_settings['batch_size']
        )
        
    except Exception as e:
        logger.error(f"❌ Background processing failed: {e}")

async def update_seismograph_data():
    """Update seismograph data periodically"""
    
    seismo_config = app_state.config.seismograph_config
    
    while True:
        try:
            # This would be replaced with actual seismograph data calculation
            # For now, just maintain the data structure
            if len(app_state.seismograph_data) > 1440:  # Keep last 24 hours (1 minute intervals)
                app_state.seismograph_data = app_state.seismograph_data[-1440:]
            
            await asyncio.sleep(seismo_config.update_interval_seconds)
            
        except Exception as e:
            logger.error(f"❌ Failed to update seismograph data: {e}")
            await asyncio.sleep(60)  # Wait longer on error

async def find_related_articles(tickers: List[str]) -> List[str]:
    """Background task to find related articles"""
    # Implementation would query database for related articles
    pass

if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    api_config = config.api_config
    
    uvicorn.run(
        "production_api:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug,
        log_level="info"
    )