"""
Financial Web Scraping Pipeline with Finance-LLM-13B Integration
Scrapes financial data and processes it through finance-llm-13b.Q5_K_S.gguf for structured output
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import requests
from bs4 import BeautifulSoup

# Import our finance LLM components
from financial_insights_assistant import FinancialInsightsAssistant
from finance_llm_provider import FinanceLLMProvider


@dataclass
class ScrapedFinancialData:
    """Structure for scraped financial data"""
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    data_type: str  # 'news', 'earnings', 'analysis', 'market_data'
    raw_html: Optional[str] = None


@dataclass
class StructuredFinancialOutput:
    """Structure for LLM-processed financial data"""
    original_data: ScrapedFinancialData
    sentiment: str
    confidence: float
    key_metrics: Dict[str, Any]
    summary_points: List[str]
    entities: List[str]  # Companies, people, financial instruments
    financial_figures: Dict[str, str]  # Revenue, earnings, etc.
    market_impact: str
    investment_thesis: str
    risk_factors: List[str]
    processing_timestamp: datetime


class FinancialWebScraper:
    """Main web scraping orchestrator for financial data"""
    
    def __init__(self):
        self.scraped_data: List[ScrapedFinancialData] = []
        self.finance_assistant = FinancialInsightsAssistant()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the scraper"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('financial_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_financial_news_sources(self) -> List[Dict[str, str]]:
        """Get list of financial news sources to scrape"""
        return [
            {
                'name': 'Yahoo Finance News',
                'base_url': 'https://finance.yahoo.com/news/',
                'type': 'news'
            },
            {
                'name': 'MarketWatch',
                'base_url': 'https://www.marketwatch.com/latest-news',
                'type': 'news'
            },
            {
                'name': 'Reuters Business',
                'base_url': 'https://www.reuters.com/business/',
                'type': 'news'
            },
            {
                'name': 'SEC EDGAR (Earnings)',
                'base_url': 'https://www.sec.gov/edgar/search/',
                'type': 'earnings'
            }
        ]
    
    async def scrape_simple_financial_data(self, urls: List[str]) -> List[ScrapedFinancialData]:
        """Simple scraping method for testing (without Scrapy for now)"""
        scraped_items = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for url in urls:
            try:
                self.logger.info(f"Scraping: {url}")
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = self._extract_title(soup, url)
                
                # Extract main content
                content = self._extract_content(soup, url)
                
                if content and len(content) > 100:  # Only process substantial content
                    scraped_item = ScrapedFinancialData(
                        title=title,
                        content=content,
                        url=url,
                        source=self._get_source_name(url),
                        timestamp=datetime.now(),
                        data_type='news',
                        raw_html=str(soup)[:1000]  # Store first 1000 chars of HTML
                    )
                    
                    scraped_items.append(scraped_item)
                    self.logger.info(f"Successfully scraped: {title[:50]}...")
                else:
                    self.logger.warning(f"Insufficient content from: {url}")
                    
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {str(e)}")
                continue
        
        return scraped_items
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract title from webpage"""
        # Try multiple title selectors
        title_selectors = [
            'h1',
            'title',
            '.headline',
            '.article-title',
            '[data-test="headline"]'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                if title and len(title) > 10:
                    return title
        
        return f"Content from {self._get_source_name(url)}"
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract main content from webpage"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Try multiple content selectors
        content_selectors = [
            '.article-body',
            '.content',
            '.story-content',
            '.entry-content',
            'article',
            '.post-content',
            '[data-test="article-content"]'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text().strip()
                if content and len(content) > 200:
                    return self._clean_content(content)
        
        # Fallback: get all paragraph text
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        return self._clean_content(content)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            'Subscribe to continue reading',
            'Sign up for free',
            'Already a subscriber?',
            'This article is for subscribers only'
        ]
        
        for phrase in unwanted_phrases:
            content = content.replace(phrase, '')
        
        return content.strip()
    
    def _get_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        source_mapping = {
            'finance.yahoo.com': 'Yahoo Finance',
            'www.marketwatch.com': 'MarketWatch',
            'www.reuters.com': 'Reuters',
            'www.bloomberg.com': 'Bloomberg',
            'www.cnbc.com': 'CNBC',
            'www.wsj.com': 'Wall Street Journal'
        }
        
        return source_mapping.get(domain, domain)
    
    async def process_with_finance_llm(self, scraped_data: ScrapedFinancialData) -> StructuredFinancialOutput:
        """Process scraped data through finance-llm-13b for structured output"""
        
        self.logger.info(f"Processing with Finance LLM: {scraped_data.title[:50]}...")
        
        try:
            # Create comprehensive prompt for structured data extraction
            extraction_prompt = f"""
Analyze the following financial content and extract structured information:

TITLE: {scraped_data.title}
SOURCE: {scraped_data.source}
CONTENT: {scraped_data.content[:2000]}...

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "sentiment": "Positive|Negative|Neutral",
    "confidence": 0.85,
    "key_metrics": {{
        "revenue": "value or null",
        "earnings": "value or null", 
        "eps": "value or null",
        "growth_rate": "value or null"
    }},
    "summary_points": ["key point 1", "key point 2", "key point 3"],
    "entities": ["Company Name", "CEO Name", "Financial Instrument"],
    "financial_figures": {{
        "figure_name": "figure_value_with_currency"
    }},
    "market_impact": "Brief description of market impact",
    "investment_thesis": "Investment implications in 1-2 sentences",
    "risk_factors": ["risk 1", "risk 2", "risk 3"]
}}

Focus on extracting concrete financial data, company performance metrics, and actionable insights.
"""
            
            # Get response from finance LLM
            response = await self.finance_assistant._generate_response(extraction_prompt)
            
            # Parse JSON response
            structured_data = await self.finance_assistant._parse_json_response(response)
            
            # Create structured output
            output = StructuredFinancialOutput(
                original_data=scraped_data,
                sentiment=structured_data.get('sentiment', 'Neutral'),
                confidence=float(structured_data.get('confidence', 0.7)),
                key_metrics=structured_data.get('key_metrics', {}),
                summary_points=structured_data.get('summary_points', []),
                entities=structured_data.get('entities', []),
                financial_figures=structured_data.get('financial_figures', {}),
                market_impact=structured_data.get('market_impact', ''),
                investment_thesis=structured_data.get('investment_thesis', ''),
                risk_factors=structured_data.get('risk_factors', []),
                processing_timestamp=datetime.now()
            )
            
            self.logger.info(f"Successfully processed: {output.sentiment} sentiment with {output.confidence:.1%} confidence")
            return output
            
        except Exception as e:
            self.logger.error(f"Error processing with Finance LLM: {str(e)}")
            
            # Return fallback structure
            return StructuredFinancialOutput(
                original_data=scraped_data,
                sentiment='Neutral',
                confidence=0.0,
                key_metrics={},
                summary_points=[f"Error processing: {str(e)}"],
                entities=[],
                financial_figures={},
                market_impact='Unable to determine',
                investment_thesis='Analysis failed',
                risk_factors=['Processing error'],
                processing_timestamp=datetime.now()
            )
    
    async def run_pipeline(self, urls: List[str]) -> List[StructuredFinancialOutput]:
        """Run the complete scraping and analysis pipeline"""
        
        self.logger.info(f"Starting financial scraping pipeline for {len(urls)} URLs")
        
        # Step 1: Scrape financial data
        scraped_data = await self.scrape_simple_financial_data(urls)
        self.logger.info(f"Scraped {len(scraped_data)} financial items")
        
        if not scraped_data:
            self.logger.warning("No data scraped, ending pipeline")
            return []
        
        # Step 2: Process through Finance LLM
        structured_outputs = []
        
        for item in scraped_data:
            try:
                structured_output = await self.process_with_finance_llm(item)
                structured_outputs.append(structured_output)
                
                # Add small delay to avoid overwhelming the LLM
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing item: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(structured_outputs)} items through Finance LLM")
        return structured_outputs
    
    def save_results(self, structured_outputs: List[StructuredFinancialOutput], filename: str = None):
        """Save structured outputs to JSON and CSV"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_analysis_{timestamp}"
        
        # Save as JSON
        json_data = []
        for output in structured_outputs:
            json_item = {
                'title': output.original_data.title,
                'source': output.original_data.source,
                'url': output.original_data.url,
                'sentiment': output.sentiment,
                'confidence': output.confidence,
                'key_metrics': output.key_metrics,
                'summary_points': output.summary_points,
                'entities': output.entities,
                'financial_figures': output.financial_figures,
                'market_impact': output.market_impact,
                'investment_thesis': output.investment_thesis,
                'risk_factors': output.risk_factors,
                'scrape_time': output.original_data.timestamp.isoformat(),
                'processing_time': output.processing_timestamp.isoformat()
            }
            json_data.append(json_item)
        
        # Save JSON
        json_file = f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy analysis
        csv_file = f"{filename}.csv"
        df = pd.DataFrame(json_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"Results saved to {json_file} and {csv_file}")
        
        return json_file, csv_file


# Demo financial URLs for testing
DEMO_FINANCIAL_URLS = [
    # Add some financial news URLs here - these would be actual URLs in production
    # For demo purposes, we'll use placeholders
    "https://finance.yahoo.com/news/",
    "https://www.marketwatch.com/story/",
    "https://www.reuters.com/business/"
]


async def demo_scraping_pipeline():
    """Demo the financial scraping and LLM analysis pipeline"""
    
    print("🕷️ Financial Web Scraping + Finance-LLM-13B Pipeline Demo")
    print("=" * 60)
    
    # Initialize scraper
    scraper = FinancialWebScraper()
    
    # Test with some sample URLs (you can replace with actual financial news URLs)
    test_urls = [
        # We'll use actual accessible URLs for testing
        "https://finance.yahoo.com/",
        "https://www.marketwatch.com/",
    ]
    
    print(f"📊 Testing with {len(test_urls)} URLs")
    print("🔍 Scraping financial data...")
    
    try:
        # Run the pipeline
        structured_outputs = await scraper.run_pipeline(test_urls)
        
        if structured_outputs:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📈 Processed {len(structured_outputs)} financial items")
            
            # Display results
            for i, output in enumerate(structured_outputs, 1):
                print(f"\n--- Analysis {i} ---")
                print(f"📰 Title: {output.original_data.title[:50]}...")
                print(f"🏢 Source: {output.original_data.source}")
                print(f"💭 Sentiment: {output.sentiment} ({output.confidence:.1%} confidence)")
                print(f"📊 Key Metrics: {output.key_metrics}")
                print(f"💡 Investment Thesis: {output.investment_thesis}")
                print(f"⚠️ Risk Factors: {len(output.risk_factors)} identified")
            
            # Save results
            json_file, csv_file = scraper.save_results(structured_outputs)
            print(f"\n💾 Results saved to:")
            print(f"   📄 {json_file}")
            print(f"   📊 {csv_file}")
            
        else:
            print("❌ No data processed")
            
    except Exception as e:
        print(f"💥 Pipeline failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(demo_scraping_pipeline())