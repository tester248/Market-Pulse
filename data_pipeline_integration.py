"""
Data Pipeline Integration

Connects the Fast & Smart data ingestion layer to Finance-LLM-13B analysis pipeline
and Virtual Trading Sandbox for real-time financial insights and trading signals.
"""

import asyncio
import asyncpg
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re

# Import existing components
try:
    from finance_llm_provider import FinanceLLMProvider
    from financial_insights_assistant import FinancialInsightsAssistant
    FINANCE_LLM_AVAILABLE = True
except ImportError:
    FINANCE_LLM_AVAILABLE = False
    print("⚠️ Finance LLM components not found. Check finance_llm_provider.py")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of financial analysis"""
    article_id: int
    url: str
    analysis_type: str
    
    # Market sentiment
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str   # bullish, bearish, neutral
    confidence: float      # 0.0 to 1.0
    
    # Key insights
    key_points: List[str]
    market_impact: str     # high, medium, low
    affected_sectors: List[str]
    mentioned_tickers: List[str]
    
    # Trading signals
    trading_signals: List[Dict[str, Any]]
    risk_assessment: str
    time_horizon: str      # short-term, medium-term, long-term
    
    # Analysis metadata
    analysis_time: datetime
    processing_time_ms: float
    model_used: str
    
    def __post_init__(self):
        if not self.analysis_time:
            self.analysis_time = datetime.now()

@dataclass
class TradingSignal:
    """Trading signal generated from analysis"""
    ticker: str
    signal_type: str      # buy, sell, hold, watch
    strength: float       # 0.0 to 1.0
    price_target: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str
    reasoning: str
    confidence: float
    risk_level: str       # low, medium, high
    
    # Source information
    source_article_id: int
    generated_at: datetime
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now()

class FinancialDataPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, db_config: Dict[str, str], llm_config: Dict[str, Any]):
        self.db_config = db_config
        self.llm_config = llm_config
        
        # Database connection
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # LLM components
        self.llm_provider: Optional[FinanceLLMProvider] = None
        self.insights_assistant: Optional[FinancialInsightsAssistant] = None
        
        # Pipeline configuration
        self.batch_size = 20
        self.analysis_timeout = 300  # 5 minutes per article
        self.max_concurrent_analyses = 5
        
        # Performance tracking
        self.stats = {
            'articles_analyzed': 0,
            'signals_generated': 0,
            'avg_analysis_time_ms': 0.0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': time.time()
        }
        
        # Analysis prompts
        self.analysis_prompts = self._load_analysis_prompts()

    def _load_analysis_prompts(self) -> Dict[str, str]:
        """Load analysis prompts for different content types"""
        return {
            'sentiment_analysis': """
            Analyze the financial sentiment of this article. Consider:
            1. Overall market sentiment (bullish/bearish/neutral)
            2. Impact on specific sectors or companies
            3. Short-term vs long-term implications
            4. Risk factors mentioned
            
            Provide:
            - Sentiment score (-1.0 to 1.0)
            - Key market-moving points
            - Affected sectors/tickers
            - Trading implications
            
            Article: {content}
            """,
            
            'trading_signals': """
            Based on this financial article, generate specific trading signals:
            1. Identify actionable investment opportunities
            2. Assess risk/reward ratios
            3. Suggest entry/exit points
            4. Consider time horizons
            
            For each signal provide:
            - Ticker symbol
            - Signal type (buy/sell/hold)
            - Strength (0-10)
            - Reasoning
            - Risk level
            
            Article: {content}
            """,
            
            'market_impact': """
            Evaluate the potential market impact of this news:
            1. Scope of impact (company/sector/market-wide)
            2. Timeline of effects (immediate/short/long-term)
            3. Magnitude of impact (high/medium/low)
            4. Related market movements
            
            Article: {content}
            """
        }

    async def initialize(self):
        """Initialize pipeline components"""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=15,
                command_timeout=60
            )
            
            # Initialize LLM components if available
            if FINANCE_LLM_AVAILABLE:
                self.llm_provider = FinanceLLMProvider(
                    model_path=self.llm_config.get('model_path', 'models/finance-llm-13b.Q5_K_S.gguf'),
                    context_length=self.llm_config.get('context_length', 4096),
                    temperature=self.llm_config.get('temperature', 0.3)
                )
                
                self.insights_assistant = FinancialInsightsAssistant(
                    llm_provider=self.llm_provider
                )
                
                logger.info("✅ Finance LLM components initialized")
            else:
                logger.warning("⚠️ Finance LLM not available - using mock analysis")
            
            logger.info("✅ Financial Data Pipeline initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()

    async def get_pending_articles(self, limit: int = None) -> List[Dict]:
        """Get articles pending analysis"""
        limit = limit or self.batch_size
        
        try:
            async with self.db_pool.acquire() as conn:
                articles = await conn.fetch("""
                    SELECT id, url, title, main_text, source_name, 
                           categories, tags, word_count, publish_date
                    FROM articles 
                    WHERE processing_status = 'completed' 
                    AND main_text IS NOT NULL
                    AND analysis_status = 'pending'
                    ORDER BY priority DESC, publish_date DESC
                    LIMIT $1
                """, limit)
                
                return [dict(article) for article in articles]
                
        except Exception as e:
            logger.error(f"❌ Error getting pending articles: {e}")
            return []

    async def analyze_article_sentiment(self, article: Dict) -> AnalysisResult:
        """Analyze sentiment and market impact of an article"""
        start_time = time.time()
        
        try:
            content = f"Title: {article['title']}\n\nContent: {article['main_text']}"
            
            # Use Finance LLM if available
            if self.llm_provider:
                prompt = self.analysis_prompts['sentiment_analysis'].format(content=content[:2000])
                response = await self._query_llm_async(prompt)
                analysis = self._parse_sentiment_response(response)
            else:
                # Mock analysis for demo
                analysis = self._mock_sentiment_analysis(article)
            
            processing_time = (time.time() - start_time) * 1000
            
            result = AnalysisResult(
                article_id=article['id'],
                url=article['url'],
                analysis_type='sentiment',
                sentiment_score=analysis['sentiment_score'],
                sentiment_label=analysis['sentiment_label'],
                confidence=analysis['confidence'],
                key_points=analysis['key_points'],
                market_impact=analysis['market_impact'],
                affected_sectors=analysis['affected_sectors'],
                mentioned_tickers=analysis['mentioned_tickers'],
                trading_signals=analysis.get('trading_signals', []),
                risk_assessment=analysis['risk_assessment'],
                time_horizon=analysis['time_horizon'],
                processing_time_ms=processing_time,
                model_used='finance-llm-13b' if self.llm_provider else 'mock'
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing article {article['id']}: {e}")
            raise

    async def _query_llm_async(self, prompt: str) -> str:
        """Query LLM asynchronously"""
        if not self.insights_assistant:
            return "Mock LLM response"
            
        try:
            # Run LLM query in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                self.insights_assistant.generate_insights,
                prompt
            )
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM query failed: {e}")
            return "Error in LLM analysis"

    def _parse_sentiment_response(self, response: str) -> Dict:
        """Parse LLM response for sentiment analysis"""
        # This would parse the structured LLM response
        # For now, using a simplified mock parser
        
        # Extract sentiment score
        sentiment_score = 0.0
        if 'bullish' in response.lower():
            sentiment_score = 0.6
        elif 'bearish' in response.lower():
            sentiment_score = -0.6
        elif 'very bullish' in response.lower():
            sentiment_score = 0.8
        elif 'very bearish' in response.lower():
            sentiment_score = -0.8
        
        # Extract tickers (simplified)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', response)
        tickers = [t for t in tickers if len(t) <= 4 and t.isalpha()][:5]
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': 'bullish' if sentiment_score > 0.2 else 'bearish' if sentiment_score < -0.2 else 'neutral',
            'confidence': 0.8,
            'key_points': self._extract_key_points(response),
            'market_impact': 'medium',
            'affected_sectors': ['technology', 'finance'],
            'mentioned_tickers': tickers,
            'risk_assessment': 'medium',
            'time_horizon': 'short-term'
        }

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from analysis text"""
        # Simple extraction of sentences containing key terms
        sentences = text.split('.')
        key_terms = ['revenue', 'earnings', 'growth', 'market', 'stock', 'price', 'analyst', 'target']
        
        key_points = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in key_terms):
                key_points.append(sentence.strip())
                if len(key_points) >= 5:
                    break
        
        return key_points or ["Analysis pending"]

    def _mock_sentiment_analysis(self, article: Dict) -> Dict:
        """Mock sentiment analysis for testing"""
        # Analyze title and content for sentiment keywords
        text = (article.get('title', '') + ' ' + article.get('main_text', '')).lower()
        
        positive_words = ['growth', 'profit', 'revenue', 'beat', 'strong', 'upgrade', 'buy']
        negative_words = ['loss', 'decline', 'weak', 'miss', 'downgrade', 'sell', 'risk']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        # Calculate sentiment
        if pos_count > neg_count:
            sentiment_score = min(0.8, 0.1 * (pos_count - neg_count))
            sentiment_label = 'bullish'
        elif neg_count > pos_count:
            sentiment_score = max(-0.8, -0.1 * (neg_count - pos_count))
            sentiment_label = 'bearish'
        else:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
        
        # Extract tickers from tags/content
        tickers = []
        if article.get('tags'):
            tickers = [tag.split(':')[1] for tag in article['tags'] if tag.startswith('tickers:')]
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': 0.7,
            'key_points': [f"Market sentiment: {sentiment_label}", f"Confidence: 70%"],
            'market_impact': 'medium' if abs(sentiment_score) > 0.5 else 'low',
            'affected_sectors': article.get('categories', ['general']),
            'mentioned_tickers': tickers[:5],
            'risk_assessment': 'high' if abs(sentiment_score) > 0.7 else 'medium',
            'time_horizon': 'short-term'
        }

    async def generate_trading_signals(self, analysis: AnalysisResult) -> List[TradingSignal]:
        """Generate trading signals from analysis"""
        signals = []
        
        try:
            for ticker in analysis.mentioned_tickers:
                if not ticker or len(ticker) > 5:
                    continue
                    
                # Determine signal type based on sentiment
                if analysis.sentiment_score > 0.5:
                    signal_type = 'buy'
                    strength = min(0.9, analysis.sentiment_score + 0.2)
                elif analysis.sentiment_score < -0.5:
                    signal_type = 'sell'
                    strength = min(0.9, abs(analysis.sentiment_score) + 0.2)
                elif abs(analysis.sentiment_score) < 0.2:
                    signal_type = 'hold'
                    strength = 0.5
                else:
                    signal_type = 'watch'
                    strength = 0.6
                
                # Create trading signal
                signal = TradingSignal(
                    ticker=ticker,
                    signal_type=signal_type,
                    strength=strength,
                    price_target=None,  # Would be calculated based on analysis
                    stop_loss=None,
                    time_horizon=analysis.time_horizon,
                    reasoning=f"Based on {analysis.sentiment_label} sentiment ({analysis.sentiment_score:.2f})",
                    confidence=analysis.confidence,
                    risk_level=analysis.risk_assessment,
                    source_article_id=analysis.article_id
                )
                
                signals.append(signal)
                
                # Limit signals per article
                if len(signals) >= 5:
                    break
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating trading signals: {e}")
            return []

    async def store_analysis_results(self, analysis: AnalysisResult, signals: List[TradingSignal]):
        """Store analysis results and trading signals"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store analysis result
                await conn.execute("""
                    INSERT INTO analysis_results 
                    (article_id, analysis_type, sentiment_score, sentiment_label,
                     confidence, key_points, market_impact, affected_sectors,
                     mentioned_tickers, risk_assessment, time_horizon,
                     processing_time_ms, model_used, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
                """,
                analysis.article_id, analysis.analysis_type, analysis.sentiment_score,
                analysis.sentiment_label, analysis.confidence, analysis.key_points,
                analysis.market_impact, analysis.affected_sectors, 
                analysis.mentioned_tickers, analysis.risk_assessment,
                analysis.time_horizon, analysis.processing_time_ms, analysis.model_used
                )
                
                # Store trading signals
                for signal in signals:
                    await conn.execute("""
                        INSERT INTO trading_signals
                        (ticker, signal_type, strength, time_horizon, reasoning,
                         confidence, risk_level, source_article_id, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    """,
                    signal.ticker, signal.signal_type, signal.strength,
                    signal.time_horizon, signal.reasoning, signal.confidence,
                    signal.risk_level, signal.source_article_id
                    )
                
                # Update article analysis status
                await conn.execute("""
                    UPDATE articles 
                    SET analysis_status = 'completed',
                        analyzed_at = NOW()
                    WHERE id = $1
                """, analysis.article_id)
                
                logger.debug(f"💾 Stored analysis for article {analysis.article_id} with {len(signals)} signals")
                
        except Exception as e:
            logger.error(f"❌ Error storing analysis results: {e}")

    async def process_analysis_batch(self) -> int:
        """Process a batch of articles for analysis"""
        articles = await self.get_pending_articles()
        
        if not articles:
            return 0
            
        logger.info(f"🧠 Analyzing {len(articles)} articles")
        
        successful = 0
        
        # Process articles with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_analyses)
        
        async def process_single_article(article):
            nonlocal successful
            async with semaphore:
                try:
                    # Analyze article
                    analysis = await self.analyze_article_sentiment(article)
                    
                    # Generate trading signals
                    signals = await self.generate_trading_signals(analysis)
                    
                    # Store results
                    await self.store_analysis_results(analysis, signals)
                    
                    successful += 1
                    self.stats['successful_analyses'] += 1
                    self.stats['signals_generated'] += len(signals)
                    
                    logger.debug(f"✅ Analyzed article {article['id']} - generated {len(signals)} signals")
                    
                except Exception as e:
                    self.stats['failed_analyses'] += 1
                    logger.error(f"❌ Error analyzing article {article['id']}: {e}")
        
        # Process all articles concurrently
        tasks = [process_single_article(article) for article in articles]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats['articles_analyzed'] += len(articles)
        
        logger.info(f"✅ Analysis batch complete: {successful}/{len(articles)} successful")
        return successful

    async def run_analysis_worker(self):
        """Run continuous analysis worker"""
        logger.info("🚀 Starting financial analysis worker")
        
        while True:
            try:
                processed = await self.process_analysis_batch()
                
                if processed == 0:
                    # No articles to process, wait a bit
                    await asyncio.sleep(30)
                else:
                    # Short pause between batches
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"❌ Analysis worker error: {e}")
                await asyncio.sleep(60)  # Back off on errors

    async def get_recent_signals(self, limit: int = 20) -> List[Dict]:
        """Get recent trading signals"""
        try:
            async with self.db_pool.acquire() as conn:
                signals = await conn.fetch("""
                    SELECT ts.*, a.title, a.url, a.source_name
                    FROM trading_signals ts
                    JOIN articles a ON ts.source_article_id = a.id
                    ORDER BY ts.created_at DESC
                    LIMIT $1
                """, limit)
                
                return [dict(signal) for signal in signals]
                
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []

    async def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        runtime = time.time() - self.stats['start_time']
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get database stats
                stats_query = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) FILTER (WHERE analysis_status = 'pending') as pending_articles,
                        COUNT(*) FILTER (WHERE analysis_status = 'completed') as completed_articles,
                        (SELECT COUNT(*) FROM trading_signals) as total_signals,
                        (SELECT COUNT(*) FROM trading_signals WHERE created_at > NOW() - INTERVAL '24 hours') as signals_24h
                    FROM articles
                    WHERE processing_status = 'completed'
                """)
                
                pipeline_stats = {
                    'runtime_seconds': runtime,
                    'articles_analyzed': self.stats['articles_analyzed'],
                    'successful_analyses': self.stats['successful_analyses'],
                    'failed_analyses': self.stats['failed_analyses'],
                    'signals_generated': self.stats['signals_generated'],
                    'success_rate': self.stats['successful_analyses'] / max(self.stats['articles_analyzed'], 1),
                    'throughput_articles_per_hour': self.stats['articles_analyzed'] / max(runtime / 3600, 1),
                    'pending_articles': stats_query['pending_articles'],
                    'completed_articles': stats_query['completed_articles'],
                    'total_signals': stats_query['total_signals'],
                    'signals_24h': stats_query['signals_24h'],
                    'finance_llm_available': FINANCE_LLM_AVAILABLE
                }
                
                return pipeline_stats
                
        except Exception as e:
            logger.error(f"❌ Error getting pipeline stats: {e}")
            return {'error': str(e)}

# Demo function
async def demo_pipeline():
    """Demonstrate pipeline functionality"""
    print("🔄 Financial Data Pipeline Demo")
    print("=" * 40)
    
    # Configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'financial_data_ingestion',
        'user': 'postgres',
        'password': 'your_password'
    }
    
    llm_config = {
        'model_path': 'models/finance-llm-13b.Q5_K_S.gguf',
        'context_length': 4096,
        'temperature': 0.3
    }
    
    try:
        pipeline = FinancialDataPipeline(db_config, llm_config)
        await pipeline.initialize()
        
        # Get pipeline stats
        stats = await pipeline.get_pipeline_stats()
        print(f"📊 Pipeline Status:")
        print(f"   Finance LLM: {'✅ Available' if stats['finance_llm_available'] else '❌ Not Available'}")
        print(f"   Pending articles: {stats.get('pending_articles', 0)}")
        print(f"   Completed analyses: {stats.get('completed_articles', 0)}")
        print(f"   Total signals: {stats.get('total_signals', 0)}")
        
        # Get recent signals
        signals = await pipeline.get_recent_signals(5)
        if signals:
            print(f"\n🚀 Recent Trading Signals:")
            for signal in signals[:3]:
                print(f"   {signal['ticker']}: {signal['signal_type'].upper()} ({signal['strength']:.2f})")
                print(f"      {signal['reasoning']}")
        
        await pipeline.cleanup()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_pipeline())