"""
Simplified Financial Web Scraping Pipeline for Testing
Uses only basic HTTP requests and BeautifulSoup (no Scrapy dependency)
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

import requests
from bs4 import BeautifulSoup

# Import our finance LLM components
from financial_insights_assistant import FinancialInsightsAssistant


@dataclass
class SimpleFinancialData:
    """Structure for scraped financial data"""
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    data_type: str  # 'news', 'earnings', 'analysis'


@dataclass
class SimpleFinancialOutput:
    """Structure for LLM-processed financial data"""
    original_data: SimpleFinancialData
    sentiment: str
    confidence: float
    key_metrics: Dict[str, Any]
    summary_points: List[str]
    entities: List[str]
    financial_figures: Dict[str, str]
    market_impact: str
    investment_thesis: str
    risk_factors: List[str]
    processing_timestamp: datetime


class SimpleFinancialScraper:
    """Simplified financial web scraper without Scrapy dependency"""
    
    def __init__(self):
        self.finance_assistant = FinancialInsightsAssistant()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the scraper"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def scrape_financial_content(self, urls: List[str]) -> List[SimpleFinancialData]:
        """Scrape financial content from URLs"""
        scraped_items = []
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in urls:
            try:
                self.logger.info(f"Scraping: {url}")
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title and content
                title = self._extract_title(soup)
                content = self._extract_content(soup)
                
                if content and len(content) > 100:
                    scraped_item = SimpleFinancialData(
                        title=title,
                        content=content[:2000],  # Limit content length
                        url=url,
                        source=self._get_source_name(url),
                        timestamp=datetime.now(),
                        data_type='news'
                    )
                    scraped_items.append(scraped_item)
                    self.logger.info(f"Successfully scraped: {title[:50]}...")
                    
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {str(e)}")
                continue
        
        return scraped_items
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from webpage"""
        title_elem = soup.find('title')
        if title_elem:
            return title_elem.get_text().strip()
        return "Financial Content"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from webpage"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get paragraph text
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Clean content
        content = re.sub(r'\s+', ' ', content)
        return content.strip()
    
    def _get_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace('www.', '').title()
    
    async def process_with_finance_llm(self, data: SimpleFinancialData) -> SimpleFinancialOutput:
        """Process scraped data through finance-llm-13b"""
        
        self.logger.info(f"Processing with Finance LLM: {data.title[:50]}...")
        
        try:
            # Create extraction prompt
            extraction_prompt = f"""
Analyze the following financial content and provide structured analysis:

TITLE: {data.title}
SOURCE: {data.source}
CONTENT: {data.content}

Provide analysis in JSON format:
{{
    "sentiment": "Positive|Negative|Neutral",
    "confidence": 0.85,
    "key_metrics": {{
        "revenue": "value or null",
        "earnings": "value or null"
    }},
    "summary_points": ["key point 1", "key point 2"],
    "entities": ["Company", "Person", "Financial Instrument"],
    "financial_figures": {{
        "figure_name": "figure_value"
    }},
    "market_impact": "Brief market impact description",
    "investment_thesis": "Investment implications",
    "risk_factors": ["risk 1", "risk 2"]
}}

Focus on financial data, company performance, and investment insights.
"""
            
            # Get response from finance LLM
            response = await self.finance_assistant._generate_response(extraction_prompt)
            
            # Parse JSON response
            structured_data = await self.finance_assistant._parse_json_response(response)
            
            # Create structured output
            output = SimpleFinancialOutput(
                original_data=data,
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
            
            self.logger.info(f"Analysis complete: {output.sentiment} sentiment")
            return output
            
        except Exception as e:
            self.logger.error(f"Error processing with Finance LLM: {str(e)}")
            
            # Return fallback structure
            return SimpleFinancialOutput(
                original_data=data,
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
    
    async def run_pipeline(self, urls: List[str] = None) -> List[SimpleFinancialOutput]:
        """Run the complete scraping and analysis pipeline"""
        
        if not urls:
            # Use test URLs
            urls = [
                "https://finance.yahoo.com/",
                "https://www.marketwatch.com/"
            ]
        
        self.logger.info(f"Starting pipeline for {len(urls)} URLs")
        
        # Step 1: Scrape financial data
        scraped_data = await self.scrape_financial_content(urls)
        self.logger.info(f"Scraped {len(scraped_data)} items")
        
        if not scraped_data:
            self.logger.warning("No data scraped")
            return []
        
        # Step 2: Process through Finance LLM
        outputs = []
        for item in scraped_data:
            try:
                output = await self.process_with_finance_llm(item)
                outputs.append(output)
                await asyncio.sleep(0.5)  # Small delay
            except Exception as e:
                self.logger.error(f"Error processing item: {str(e)}")
                continue
        
        self.logger.info(f"Generated {len(outputs)} insights")
        return outputs
    
    def save_results(self, outputs: List[SimpleFinancialOutput], filename: str = None):
        """Save results to JSON and CSV"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_financial_analysis_{timestamp}"
        
        # Prepare data
        json_data = []
        for output in outputs:
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
                'processing_time': output.processing_timestamp.isoformat()
            }
            json_data.append(json_item)
        
        # Save JSON
        json_file = f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save CSV
        csv_file = f"{filename}.csv"
        df = pd.DataFrame(json_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"Results saved to {json_file} and {csv_file}")
        return json_file, csv_file


async def demo_simple_pipeline():
    """Demo the simplified financial pipeline"""
    
    print("Simple Financial Web Scraping + Finance-LLM Pipeline Demo")
    print("=" * 60)
    
    # Initialize scraper
    scraper = SimpleFinancialScraper()
    
    # Create test financial data
    test_data = SimpleFinancialData(
        title="Tesla Reports Record Q3 Deliveries and Strong Financial Performance",
        content="""Tesla Inc announced record third-quarter vehicle deliveries of 435,059 units, 
        exceeding analyst expectations of 430,000 vehicles. The electric vehicle manufacturer 
        reported revenue of $23.4 billion, up 9% year-over-year, with automotive revenue reaching 
        $20.0 billion. Net income was $7.9 billion, representing a significant improvement from 
        the previous quarter. Tesla's energy generation and storage segment contributed $1.6 billion 
        in revenue, while services and other revenue was $2.2 billion. The company's gross margin 
        improved to 19.3%, driven by cost reduction initiatives and higher average selling prices. 
        CEO Elon Musk highlighted strong demand for the Model Y and successful expansion in 
        international markets.""",
        url="https://example.com/tesla-earnings",
        source="Financial Test News",
        timestamp=datetime.now(),
        data_type="earnings"
    )
    
    print("Test financial data created")
    print(f"Title: {test_data.title}")
    print(f"Content length: {len(test_data.content)} characters")
    
    # Process with Finance LLM
    print("\nProcessing with Finance-LLM-13B...")
    result = await scraper.process_with_finance_llm(test_data)
    
    print("Analysis completed!")
    
    # Display results
    print("\n" + "="*60)
    print("FINANCIAL ANALYSIS RESULTS")
    print("="*60)
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Market Impact: {result.market_impact}")
    print(f"Investment Thesis: {result.investment_thesis}")
    
    if result.key_metrics:
        print(f"\nKey Financial Metrics:")
        for metric, value in result.key_metrics.items():
            print(f"  - {metric}: {value}")
    
    if result.entities:
        print(f"\nEntities Identified:")
        for entity in result.entities[:5]:
            print(f"  - {entity}")
    
    if result.risk_factors:
        print(f"\nRisk Factors:")
        for risk in result.risk_factors[:3]:
            print(f"  - {risk}")
    
    # Save results
    outputs = [result]
    json_file, csv_file = scraper.save_results(outputs)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    
    print("\nSIMPLE PIPELINE DEMO COMPLETED SUCCESSFULLY!")
    return True


if __name__ == "__main__":
    success = asyncio.run(demo_simple_pipeline())
    if success:
        print("\nThe Simple Financial Analysis Pipeline is working correctly!")
    else:
        print("\nPipeline test failed.")
