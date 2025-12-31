"""
Smart Text Extraction Engine

Intelligent text extraction using trafilatura library for clean content parsing.
Removes ads, navigation, boilerplate and extracts pure article text with metadata.
"""

import asyncio
import asyncpg
import logging
import time
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from urllib.parse import urlparse
import json

# Import trafilatura and related libraries
try:
    import trafilatura
    from trafilatura import extract, extract_metadata
    from trafilatura.settings import use_config
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("⚠️ trafilatura not installed. Run: pip install trafilatura")

# Import readability as fallback
try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False

# Import newspaper3k as another fallback
try:
    from newspaper import Article as NewspaperArticle
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    """Container for extracted article content and metadata"""
    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    main_text: Optional[str] = None
    summary: Optional[str] = None
    language: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    extraction_method: str = 'unknown'
    confidence_score: float = 0.0
    
    # Additional metadata
    tags: List[str] = None
    categories: List[str] = None
    source_domain: str = ''
    content_hash: str = ''
    
    # Quality metrics
    text_density: float = 0.0  # Text to HTML ratio
    paragraph_count: int = 0
    sentence_count: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.categories is None:
            self.categories = []
        if self.main_text:
            self.word_count = len(self.main_text.split())
            self.char_count = len(self.main_text)
            self.paragraph_count = len([p for p in self.main_text.split('\n\n') if p.strip()])
            self.sentence_count = len(re.findall(r'[.!?]+', self.main_text))
            self.content_hash = hashlib.sha256(self.main_text.encode()).hexdigest()[:16]
        
        # Extract domain
        try:
            self.source_domain = urlparse(self.url).netloc.lower()
        except:
            self.source_domain = 'unknown'

class SmartTextExtractor:
    """Smart text extraction engine with multiple fallback methods"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # Performance tracking
        self.stats = {
            'articles_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'avg_processing_time_ms': 0.0,
            'extraction_methods': {},
            'start_time': time.time()
        }
        
        # Quality thresholds
        self.min_word_count = 50
        self.min_char_count = 300
        self.min_text_density = 0.1
        
        # Initialize trafilatura config if available
        if TRAFILATURA_AVAILABLE:
            self.trafilatura_config = use_config()
            # Optimize for financial content
            self.trafilatura_config.set('DEFAULT', 'MIN_EXTRACTED_SIZE', '200')
            self.trafilatura_config.set('DEFAULT', 'MIN_OUTPUT_SIZE', '100')
            self.trafilatura_config.set('DEFAULT', 'MAX_OUTPUT_SIZE', '100000')

    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=15,
                command_timeout=60
            )
            logger.info("✅ Smart Text Extractor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize extractor: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()

    def extract_with_trafilatura(self, html_content: str, url: str) -> Optional[ExtractedContent]:
        """Extract content using trafilatura (primary method)"""
        if not TRAFILATURA_AVAILABLE:
            return None
            
        try:
            start_time = time.time()
            
            # Extract main content
            main_text = extract(
                html_content,
                config=self.trafilatura_config,
                include_comments=False,
                include_tables=True,  # Financial data often in tables
                include_formatting=False,
                url=url
            )
            
            # Extract metadata
            metadata = extract_metadata(html_content, fast=True)
            
            processing_time = (time.time() - start_time) * 1000
            
            if not main_text or len(main_text) < self.min_char_count:
                return None
                
            # Calculate text density
            text_density = len(main_text) / len(html_content) if html_content else 0
            
            # Build extracted content
            extracted = ExtractedContent(
                url=url,
                main_text=main_text,
                title=metadata.title if metadata else None,
                author=metadata.author if metadata else None,
                publish_date=self._parse_date(metadata.date if metadata else None),
                language=metadata.language if metadata else None,
                extraction_method='trafilatura',
                confidence_score=0.9 if text_density > 0.15 else 0.7,
                text_density=text_density
            )
            
            # Auto-generate summary (first 200 words)
            if main_text:
                words = main_text.split()
                if len(words) > 30:
                    extracted.summary = ' '.join(words[:30]) + '...'
            
            logger.debug(f"✅ Trafilatura extraction: {extracted.word_count} words in {processing_time:.1f}ms")
            return extracted
            
        except Exception as e:
            logger.warning(f"⚠️ Trafilatura extraction failed: {e}")
            return None

    def extract_with_readability(self, html_content: str, url: str) -> Optional[ExtractedContent]:
        """Extract content using readability (fallback method)"""
        if not READABILITY_AVAILABLE:
            return None
            
        try:
            doc = Document(html_content)
            main_text = doc.summary()
            
            # Clean up HTML tags
            import re
            main_text = re.sub(r'<[^>]+>', '', main_text)
            main_text = re.sub(r'\s+', ' ', main_text).strip()
            
            if len(main_text) < self.min_char_count:
                return None
                
            extracted = ExtractedContent(
                url=url,
                main_text=main_text,
                title=doc.title(),
                extraction_method='readability',
                confidence_score=0.6,
                text_density=len(main_text) / len(html_content) if html_content else 0
            )
            
            return extracted
            
        except Exception as e:
            logger.warning(f"⚠️ Readability extraction failed: {e}")
            return None

    def extract_with_newspaper(self, html_content: str, url: str) -> Optional[ExtractedContent]:
        """Extract content using newspaper3k (second fallback)"""
        if not NEWSPAPER_AVAILABLE:
            return None
            
        try:
            article = NewspaperArticle(url)
            article.set_html(html_content)
            article.parse()
            
            if not article.text or len(article.text) < self.min_char_count:
                return None
                
            extracted = ExtractedContent(
                url=url,
                main_text=article.text,
                title=article.title,
                author=', '.join(article.authors) if article.authors else None,
                publish_date=article.publish_date,
                summary=article.summary if hasattr(article, 'summary') else None,
                extraction_method='newspaper3k',
                confidence_score=0.5,
                text_density=len(article.text) / len(html_content) if html_content else 0
            )
            
            return extracted
            
        except Exception as e:
            logger.warning(f"⚠️ Newspaper3k extraction failed: {e}")
            return None

    def extract_with_regex(self, html_content: str, url: str) -> Optional[ExtractedContent]:
        """Basic regex extraction (last resort)"""
        try:
            # Remove script and style elements
            clean_html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', clean_html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else None
            
            # Remove all HTML tags
            text = re.sub(r'<[^>]+>', ' ', clean_html)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) < self.min_char_count:
                return None
                
            extracted = ExtractedContent(
                url=url,
                main_text=text,
                title=title,
                extraction_method='regex',
                confidence_score=0.2,
                text_density=len(text) / len(html_content) if html_content else 0
            )
            
            return extracted
            
        except Exception as e:
            logger.warning(f"⚠️ Regex extraction failed: {e}")
            return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
            
        try:
            # Try common date formats
            formats = [
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S%z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%d %b %Y',
                '%B %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
                    
            return None
            
        except Exception:
            return None

    def enhance_financial_content(self, extracted: ExtractedContent) -> ExtractedContent:
        """Enhance extracted content with financial-specific processing"""
        if not extracted.main_text:
            return extracted
            
        text = extracted.main_text
        
        # Extract financial entities
        financial_patterns = {
            'tickers': r'\b[A-Z]{1,5}\b(?=\s|$|[^\w])',
            'currencies': r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|CAD|AUD)',
            'percentages': r'\d+(?:\.\d+)?%',
            'market_caps': r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|T)',
        }
        
        financial_entities = {}
        for entity_type, pattern in financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                financial_entities[entity_type] = list(set(matches))
        
        # Categorize content
        categories = []
        financial_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'quarterly', 'eps'],
            'market_news': ['market', 'trading', 'index', 'dow', 'nasdaq', 's&p'],
            'crypto': ['bitcoin', 'ethereum', 'cryptocurrency', 'blockchain'],
            'fed_policy': ['federal reserve', 'fed', 'interest rate', 'monetary policy'],
            'merger': ['merger', 'acquisition', 'deal', 'buyout'],
            'ipo': ['ipo', 'initial public offering', 'public offering'],
            'analyst': ['analyst', 'rating', 'upgrade', 'downgrade', 'price target']
        }
        
        text_lower = text.lower()
        for category, keywords in financial_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        # Update extracted content
        extracted.categories = categories
        
        # Add financial entities as tags
        for entity_type, entities in financial_entities.items():
            extracted.tags.extend([f"{entity_type}:{entity}" for entity in entities[:5]])  # Limit tags
        
        return extracted

    async def extract_text(self, html_content: str, url: str) -> Optional[ExtractedContent]:
        """Extract text using multiple methods with fallbacks"""
        start_time = time.time()
        
        # Try extraction methods in order of preference
        extraction_methods = [
            self.extract_with_trafilatura,
            self.extract_with_readability,
            self.extract_with_newspaper,
            self.extract_with_regex
        ]
        
        extracted = None
        for method in extraction_methods:
            try:
                extracted = method(html_content, url)
                if extracted and self._is_quality_content(extracted):
                    # Enhance with financial-specific processing
                    extracted = self.enhance_financial_content(extracted)
                    break
            except Exception as e:
                logger.warning(f"⚠️ Extraction method {method.__name__} failed: {e}")
                continue
        
        # Update stats
        processing_time = (time.time() - start_time) * 1000
        self.stats['articles_processed'] += 1
        
        if extracted:
            self.stats['successful_extractions'] += 1
            method_name = extracted.extraction_method
            self.stats['extraction_methods'][method_name] = self.stats['extraction_methods'].get(method_name, 0) + 1
            logger.debug(f"✅ Text extraction successful using {method_name} ({processing_time:.1f}ms)")
        else:
            self.stats['failed_extractions'] += 1
            logger.warning(f"❌ All extraction methods failed for {url}")
        
        # Update average processing time
        self.stats['avg_processing_time_ms'] = (
            (self.stats['avg_processing_time_ms'] * (self.stats['articles_processed'] - 1) + processing_time) /
            self.stats['articles_processed']
        )
        
        return extracted

    def _is_quality_content(self, extracted: ExtractedContent) -> bool:
        """Check if extracted content meets quality thresholds"""
        if not extracted.main_text:
            return False
            
        return (
            extracted.word_count >= self.min_word_count and
            extracted.char_count >= self.min_char_count and
            extracted.text_density >= self.min_text_density
        )

    async def get_pending_extraction_jobs(self, limit: int = 100) -> List[Dict]:
        """Get articles pending text extraction"""
        try:
            async with self.db_pool.acquire() as conn:
                articles = await conn.fetch("""
                    SELECT id, url, content, source_name, scraped_at
                    FROM articles 
                    WHERE processing_status = 'pending' 
                    AND content IS NOT NULL
                    ORDER BY priority DESC, scraped_at ASC
                    LIMIT $1
                """, limit)
                
                return [dict(article) for article in articles]
                
        except Exception as e:
            logger.error(f"❌ Error getting pending extraction jobs: {e}")
            return []

    async def update_extracted_content(self, article_id: int, extracted: ExtractedContent):
        """Update article with extracted content"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE articles SET
                        title = COALESCE($2, title),
                        author = $3,
                        publish_date = $4,
                        main_text = $5,
                        summary = $6,
                        language = $7,
                        word_count = $8,
                        categories = $9,
                        tags = $10,
                        processing_status = 'completed',
                        processed_at = NOW(),
                        extraction_method = $11,
                        confidence_score = $12
                    WHERE id = $1
                """,
                article_id,
                extracted.title,
                extracted.author,
                extracted.publish_date,
                extracted.main_text,
                extracted.summary,
                extracted.language,
                extracted.word_count,
                extracted.categories,
                extracted.tags,
                extracted.extraction_method,
                extracted.confidence_score
                )
                
                logger.debug(f"💾 Updated article {article_id} with extracted content")
                
        except Exception as e:
            logger.error(f"❌ Error updating extracted content: {e}")

    async def mark_extraction_failed(self, article_id: int, error_message: str):
        """Mark article extraction as failed"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE articles SET
                        processing_status = 'failed',
                        processed_at = NOW(),
                        error_message = $2
                    WHERE id = $1
                """, article_id, error_message)
                
        except Exception as e:
            logger.error(f"❌ Error marking extraction as failed: {e}")

    async def process_extraction_batch(self, batch_size: int = 50):
        """Process a batch of articles for text extraction"""
        articles = await self.get_pending_extraction_jobs(batch_size)
        
        if not articles:
            return 0
            
        logger.info(f"📝 Processing {len(articles)} articles for text extraction")
        
        successful = 0
        for article in articles:
            try:
                extracted = await self.extract_text(article['content'], article['url'])
                
                if extracted:
                    await self.update_extracted_content(article['id'], extracted)
                    successful += 1
                else:
                    await self.mark_extraction_failed(article['id'], "Failed to extract quality content")
                    
            except Exception as e:
                await self.mark_extraction_failed(article['id'], str(e))
                logger.error(f"❌ Error processing article {article['id']}: {e}")
        
        logger.info(f"✅ Extraction batch complete: {successful}/{len(articles)} successful")
        return successful

    async def run_extraction_worker(self):
        """Run continuous extraction worker"""
        logger.info("🚀 Starting text extraction worker")
        
        while True:
            try:
                processed = await self.process_extraction_batch()
                
                if processed == 0:
                    # No articles to process, wait a bit
                    await asyncio.sleep(10)
                else:
                    # Short pause between batches
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Extraction worker error: {e}")
                await asyncio.sleep(30)  # Back off on errors

    def get_extraction_stats(self) -> Dict:
        """Get extraction statistics"""
        runtime = time.time() - self.stats['start_time']
        
        return {
            'runtime_seconds': runtime,
            'articles_processed': self.stats['articles_processed'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'success_rate': self.stats['successful_extractions'] / max(self.stats['articles_processed'], 1),
            'avg_processing_time_ms': self.stats['avg_processing_time_ms'],
            'throughput_articles_per_second': self.stats['articles_processed'] / max(runtime, 1),
            'extraction_methods': dict(self.stats['extraction_methods']),
            'trafilatura_available': TRAFILATURA_AVAILABLE,
            'readability_available': READABILITY_AVAILABLE,
            'newspaper_available': NEWSPAPER_AVAILABLE
        }

# Demo function
async def demo_text_extraction():
    """Demonstrate text extraction functionality"""
    print("📝 Smart Text Extraction Demo")
    print("=" * 40)
    
    if not TRAFILATURA_AVAILABLE:
        print("⚠️ Installing trafilatura for optimal performance...")
        print("   Run: pip install trafilatura")
    
    # Test HTML content
    test_html = """
    <html>
    <head><title>Market Update: Tech Stocks Surge</title></head>
    <body>
        <div class="nav">Navigation</div>
        <div class="content">
            <h1>Tech Stocks Rally as AI Optimism Grows</h1>
            <p class="byline">By Financial Reporter | March 15, 2024</p>
            <p>Technology stocks surged today as investors showed renewed optimism about artificial intelligence developments. The NASDAQ composite index rose 2.3%, with major tech companies leading the gains.</p>
            <p>Apple (AAPL) gained 3.1% to $185.50, while Microsoft (MSFT) climbed 2.8% to $420.25. The rally comes after several positive analyst reports highlighting the potential for AI integration across various sectors.</p>
            <p>"We're seeing a fundamental shift in how investors view AI-enabled companies," said market analyst John Smith. "The revenue potential is enormous."</p>
        </div>
        <div class="ads">Advertisement</div>
    </body>
    </html>
    """
    
    # Database config (placeholder)
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'financial_data_ingestion',
        'user': 'postgres',
        'password': 'your_password'
    }
    
    try:
        extractor = SmartTextExtractor(db_config)
        
        # Test extraction
        extracted = await extractor.extract_text(test_html, 'https://example.com/tech-rally')
        
        if extracted:
            print("✅ Extraction successful!")
            print(f"   Title: {extracted.title}")
            print(f"   Method: {extracted.extraction_method}")
            print(f"   Word count: {extracted.word_count}")
            print(f"   Categories: {extracted.categories}")
            print(f"   Tags: {extracted.tags[:5]}")  # First 5 tags
            print(f"   Confidence: {extracted.confidence_score:.2f}")
            print(f"   Text preview: {extracted.main_text[:200]}...")
        else:
            print("❌ Extraction failed")
        
        # Show stats
        stats = extractor.get_extraction_stats()
        print(f"\n📊 Extraction capabilities:")
        print(f"   Trafilatura: {'✅' if stats['trafilatura_available'] else '❌'}")
        print(f"   Readability: {'✅' if stats['readability_available'] else '❌'}")
        print(f"   Newspaper3k: {'✅' if stats['newspaper_available'] else '❌'}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_text_extraction())