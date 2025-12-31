import asyncio
import aiohttp
import os
import json
import time
import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from datetime import datetime, timedelta

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
        """Fetch and parse RSS feeds"""
        all_articles = []
        
        for feed_url in self.rss_feeds:
            try:
                print(f"📡 Fetching RSS feed: {feed_url}")
                
                # Set a timeout and user agent for better compatibility
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                # Fetch the RSS feed
                response = requests.get(feed_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse the RSS feed
                feed = feedparser.parse(response.content)
                
                print(f"   Found {len(feed.entries)} articles")
                
                # Extract articles
                for i, entry in enumerate(feed.entries[:max_articles_per_feed]):
                    article = {
                        'title': entry.get('title', 'No title'),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': feed_url,
                        'content': self.extract_article_content(entry)
                    }
                    all_articles.append(article)
                    
            except Exception as e:
                print(f"   ❌ Error fetching {feed_url}: {str(e)}")
                continue
        
        self.articles = all_articles
        return all_articles
    
    def extract_article_content(self, entry) -> str:
        """Extract content from RSS entry"""
        content = ""
        
        # Try to get content from various fields
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value if isinstance(entry.content, list) else entry.content
        elif hasattr(entry, 'summary') and entry.summary:
            content = entry.summary
        elif hasattr(entry, 'description') and entry.description:
            content = entry.description
        
        # Clean HTML if present
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            content = soup.get_text(strip=True)
        
        return content[:1000]  # Limit content length
    
    def create_context_from_articles(self, query: str, max_articles: int = 15) -> str:
        """Create context from relevant articles for RAG"""
        if not self.articles:
            return "No recent financial news available."
        
        # Simple relevance scoring based on keyword matching
        query_words = set(query.lower().split())
        
        scored_articles = []
        for article in self.articles:
            # Score based on title and content relevance
            article_text = f"{article['title']} {article['summary']} {article['content']}".lower()
            article_words = set(article_text.split())
            
            # Calculate relevance score
            common_words = query_words.intersection(article_words)
            relevance_score = len(common_words) / max(len(query_words), 1)
            
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
        return context

class MultiModelProcessor:
    def __init__(self, api_key: str = None):
        # Use environment variable or fallback to provided key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "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")
        
        # List of currently available free models on OpenRouter (updated list)
        self.free_models = [
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "meta-llama/llama-3.3-8b-instruct:free",
            "nousresearch/deephermes-3-llama-3-8b-preview:free",
        ]
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rss_processor = RSSFeedProcessor()
    
    async def query_model(self, session: aiohttp.ClientSession, model: str, messages: List[Dict], timeout: int = 30) -> Dict[str, Any]:
        """Query a single model with error handling and timeout"""
        try:
            print(f"🤖 Querying {model}...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            async with session.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    response_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    return {
                        "model": model,
                        "response": response_content,
                        "success": True,
                        "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                        "timestamp": time.time(),
                        "status_code": response.status
                    }
                else:
                    error_text = await response.text()
                    return {
                        "model": model,
                        "response": None,
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "timestamp": time.time(),
                        "status_code": response.status
                    }
            
        except asyncio.TimeoutError:
            return {
                "model": model,
                "response": None,
                "success": False,
                "error": "Request timeout",
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "model": model,
                "response": None,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def sequential_pipeline_processing(self, original_prompt: str, use_rag: bool = True) -> Dict[str, Any]:
        """Process prompt through a sequential 3-model pipeline"""
        
        pipeline_results = {
            "original_prompt": original_prompt,
            "step1_refined_prompt": "",
            "step2_rag_response": "",
            "step3_structured_output": "",
            "pipeline_success": False,
            "errors": [],
            "articles_processed": 0
        }
        
        # Step 1: Prompt Fine-tuning Model
        print("🔧 Step 1: Fine-tuning prompt with first model...")
        step1_result = await self.step1_prompt_refinement(original_prompt)
        
        if not step1_result["success"]:
            pipeline_results["errors"].append(f"Step 1 failed: {step1_result['error']}")
            return pipeline_results
        
        refined_prompt = step1_result["refined_prompt"]
        pipeline_results["step1_refined_prompt"] = refined_prompt
        print(f"✅ Refined prompt: {refined_prompt[:100]}...")
        
        # Step 2: RAG-enhanced Response Generation
        print("\n📚 Step 2: Generating response with RAG context...")
        step2_result = await self.step2_rag_response_generation(refined_prompt, use_rag)
        
        if not step2_result["success"]:
            pipeline_results["errors"].append(f"Step 2 failed: {step2_result['error']}")
            return pipeline_results
        
        rag_response = step2_result["response"]
        pipeline_results["step2_rag_response"] = rag_response
        pipeline_results["articles_processed"] = step2_result.get("articles_count", 0)
        print(f"✅ Generated response: {rag_response[:100]}...")
        
        # Step 3: Output Structuring and Formatting
        print("\n📄 Step 3: Structuring and formatting output...")
        step3_result = await self.step3_output_structuring(rag_response, original_prompt)
        
        if not step3_result["success"]:
            pipeline_results["errors"].append(f"Step 3 failed: {step3_result['error']}")
            return pipeline_results
        
        structured_output = step3_result["structured_response"]
        pipeline_results["step3_structured_output"] = structured_output
        pipeline_results["pipeline_success"] = True
        print("✅ Pipeline completed successfully!")
        
        return pipeline_results
    
    async def step1_prompt_refinement(self, original_prompt: str) -> Dict[str, Any]:
        """Step 1: Use first model to refine and enhance the prompt"""
        try:
            refinement_system_prompt = """You are a prompt engineering expert. Your task is to take a user's question and refine it to be more specific, comprehensive, and effective for financial analysis.

Guidelines for prompt refinement:
1. Make the prompt more specific and detailed
2. Add context that would help generate better financial insights
3. Include relevant financial terminology if appropriate
4. Ensure the refined prompt will elicit comprehensive analysis
5. Keep the core intent of the original question

Return ONLY the refined prompt, nothing else."""

            messages = [
                {"role": "system", "content": refinement_system_prompt},
                {"role": "user", "content": f"Please refine this financial question to make it more comprehensive and specific: '{original_prompt}'"}
            ]
            
            async with aiohttp.ClientSession() as session:
                result = await self.query_model(session, self.free_models[0], messages, timeout=30)
            
            if result["success"]:
                return {
                    "success": True,
                    "refined_prompt": result["response"].strip(),
                    "model_used": result["model"],
                    "tokens_used": result.get("tokens_used", 0)
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "refined_prompt": original_prompt  # Fallback to original
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "refined_prompt": original_prompt  # Fallback to original
            }
    
    async def step2_rag_response_generation(self, refined_prompt: str, use_rag: bool = True) -> Dict[str, Any]:
        """Step 2: Use second model to generate response with RAG context"""
        try:
            # Fetch RSS content if RAG is enabled
            rag_context = ""
            articles_count = 0
            
            if use_rag:
                print("   🔍 Fetching financial news for context...")
                self.rss_processor.fetch_rss_content(max_articles_per_feed=8)
                rag_context = self.rss_processor.create_context_from_articles(refined_prompt, max_articles=12)
                articles_count = len(self.rss_processor.articles)
                print(f"   📰 Found {articles_count} relevant articles for context")
            
            # Create enhanced prompt with RAG context
            if rag_context and use_rag:
                enhanced_prompt = f"""{rag_context}

Based on the above recent financial news context, please provide a comprehensive analysis for the following question:

{refined_prompt}

Please reference specific news items from the context when relevant and provide detailed insights that incorporate current market information, trends, and data."""
            else:
                enhanced_prompt = f"""Please provide a comprehensive financial analysis for the following question:

{refined_prompt}

Provide detailed insights, current market context, and actionable information."""
            
            messages = [{"role": "user", "content": enhanced_prompt}]
            
            async with aiohttp.ClientSession() as session:
                result = await self.query_model(session, self.free_models[1], messages, timeout=45)
            
            if result["success"]:
                return {
                    "success": True,
                    "response": result["response"],
                    "model_used": result["model"],
                    "tokens_used": result.get("tokens_used", 0),
                    "articles_count": articles_count,
                    "rag_used": use_rag
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def step3_output_structuring(self, raw_response: str, original_prompt: str) -> Dict[str, Any]:
        """Step 3: Use third model to structure and format the output"""
        try:
            structuring_system_prompt = """You are an expert financial report formatter. Your task is to take a raw financial analysis response and structure it into a well-organized, professional format.

Structure the output with the following sections:
1. **Executive Summary** - Key takeaways (2-3 sentences)
2. **Detailed Analysis** - Main content organized with clear headings
3. **Key Insights** - Bullet points of important findings
4. **Market Context** - Current market conditions and relevance
5. **Actionable Recommendations** - Specific next steps or considerations
6. **Risk Factors** - Important risks to consider
7. **Data Sources** - If news sources were referenced

Make the output professional, well-formatted with markdown, and easy to read. Preserve all the important information from the original response while making it more structured and professional."""

            formatting_prompt = f"""Original Question: {original_prompt}

Raw Financial Analysis Response:
{raw_response}

Please restructure this response into a professional, well-organized financial report format as specified in the system instructions."""

            messages = [
                {"role": "system", "content": structuring_system_prompt},
                {"role": "user", "content": formatting_prompt}
            ]
            
            async with aiohttp.ClientSession() as session:
                result = await self.query_model(session, self.free_models[2], messages, timeout=45)
            
            if result["success"]:
                return {
                    "success": True,
                    "structured_response": result["response"],
                    "model_used": result["model"],
                    "tokens_used": result.get("tokens_used", 0)
                }
            else:
                return {
                    "success": False,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def aggregate_responses(self, responses: List[Dict]) -> Dict[str, Any]:
        """Aggregate and analyze responses from multiple models"""
        if not responses:
            return {"error": "No successful responses to aggregate"}
        
        # Extract response texts
        response_texts = [r["response"] for r in responses if r["response"]]
        
        if not response_texts:
            return {"error": "No valid response content to aggregate"}
        
        # Simple aggregation metrics
        avg_length = sum(len(text) for text in response_texts) / len(response_texts)
        
        # Score responses based on length and content diversity
        scored_responses = []
        for response in responses:
            if response["response"]:
                score = self.score_response(response["response"], response_texts)
                scored_responses.append({
                    **response,
                    "quality_score": score
                })
        
        # Sort by quality score
        scored_responses.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return {
            "best_response": scored_responses[0] if scored_responses else None,
            "all_responses": scored_responses,
            "summary": {
                "total_responses": len(response_texts),
                "average_length": avg_length,
                "response_lengths": [len(text) for text in response_texts]
            },
            "consensus_themes": self.extract_common_themes(response_texts)
        }
    
    def score_response(self, response: str, all_responses: List[str]) -> float:
        """Score a response based on various quality metrics"""
        if not response:
            return 0.0
            
        # Basic scoring factors
        length_score = min(len(response) / 500, 1.0)  # Normalize to 0-1, prefer longer responses
        
        # Diversity score (how different from others)
        diversity_score = 0
        for other_response in all_responses:
            if other_response != response:
                # Simple diversity measure based on unique words
                words1 = set(response.lower().split())
                words2 = set(other_response.lower().split())
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                diversity_score += (1 - overlap)
        
        diversity_score = diversity_score / max(len(all_responses) - 1, 1)
        
        # Combine scores
        final_score = (length_score * 0.4) + (diversity_score * 0.6)
        return final_score
    
    def extract_common_themes(self, responses: List[str]) -> List[str]:
        """Extract common themes across all responses"""
        # Simple keyword extraction for common themes
        all_words = []
        for response in responses:
            words = response.lower().split()
            all_words.extend([word.strip('.,!?;:"()[]') for word in words if len(word) > 3])
        
        # Count word frequency
        word_count = {}
        for word in all_words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Get most common themes (words appearing in multiple responses)
        threshold = max(2, len(responses) // 2)
        common_themes = [word for word, count in word_count.items() 
                        if count >= threshold and len(word) > 4]
        
        return sorted(common_themes, key=lambda x: word_count[x], reverse=True)[:10]
    
    def generate_consensus_answer_with_sources(self, responses: List[Dict], articles_count: int = 0) -> str:
        """Generate a consensus answer from multiple model responses with source attribution"""
        if not responses:
            return "No responses available to generate consensus."
        
        # Get the best response as the base
        aggregated = self.aggregate_responses(responses)
        best_response = aggregated.get("best_response")
        
        if not best_response:
            return "Unable to determine best response."
        
        consensus_themes = aggregated.get("consensus_themes", [])
        
        # Create a refined answer based on the best response and common themes
        refined_answer = f"""**📊 Multi-Model Financial Analysis (Based on {len(responses)} AI models + {articles_count} news articles):**

{best_response['response']}

**🔍 Key Themes Identified:** {', '.join(consensus_themes[:5]) if consensus_themes else 'Various perspectives provided'}

**📈 Analysis Confidence:** {len(responses)} models processed the query with {aggregated['summary']['average_length']:.0f} average characters per response.

**📰 News Context:** This analysis incorporates {articles_count} recent financial news articles for current market insights.
"""
        
        return refined_answer

async def main():
    # Initialize the multi-model processor
    processor = MultiModelProcessor()
    
    print("🤖 Financial AI Sequential Pipeline System")
    print("==========================================")
    print("Pipeline Flow:")
    print("1. 🔧 Prompt Refinement Model - Enhances your question")
    print("2. 📚 RAG Response Model - Generates answer with news context") 
    print("3. 📄 Structuring Model - Formats professional output")
    print()
    print("RSS Feeds: NASDAQ, Financial Times, Money.com, MarketWatch")
    print()
    
    # Get user input
    prompt = input("💬 Enter your financial question (or press Enter for default): ").strip()
    if not prompt:
        prompt = "What are the current trends in the stock market and what should investors be aware of?"
    
    use_rag = input("📰 Use current financial news context? (Y/n): ").strip().lower()
    use_rag = use_rag != 'n'
    
    print(f"\n📝 Original Query: '{prompt}'")
    print(f"🔄 Starting sequential pipeline processing...\n")
    
    # Process through sequential pipeline
    pipeline_results = await processor.sequential_pipeline_processing(prompt, use_rag=use_rag)
    
    print(f"\n" + "="*60)
    print("📊 PIPELINE RESULTS")
    print("="*60)
    
    if pipeline_results["pipeline_success"]:
        print("✅ Pipeline Status: SUCCESS")
        print(f"📰 Articles Processed: {pipeline_results['articles_processed']}")
        
        print(f"\n🔧 STEP 1 - Refined Prompt:")
        print(f"'{pipeline_results['step1_refined_prompt']}'")
        
        print(f"\n📚 STEP 2 - RAG Response Generated Successfully")
        print(f"Response Length: {len(pipeline_results['step2_rag_response'])} characters")
        
        print(f"\n📄 STEP 3 - FINAL STRUCTURED OUTPUT:")
        print("="*60)
        print(pipeline_results['step3_structured_output'])
        print("="*60)
        
        # Show pipeline details
        print(f"\n� PIPELINE DETAILS:")
        print(f"Models Used: {len(processor.free_models)} models in sequence")
        for i, model in enumerate(processor.free_models, 1):
            step_name = ["Prompt Refinement", "RAG Response", "Output Structuring"][i-1]
            print(f"  Step {i} ({step_name}): {model}")
        
        # Show news sources if RAG was used
        if use_rag and processor.rss_processor.articles:
            print(f"\n📰 NEWS SOURCES USED ({len(processor.rss_processor.articles)} articles):")
            for i, article in enumerate(processor.rss_processor.articles[:5], 1):
                print(f"{i}. {article['title'][:80]}...")
                print(f"   Published: {article['published']}")
                print(f"   Source: {article['source'].split('/')[-1]}")
                print()
    
    else:
        print("❌ Pipeline Status: FAILED")
        print("Errors encountered:")
        for error in pipeline_results["errors"]:
            print(f"  - {error}")
        
        print("\nThis might be due to:")
        print("- Network connectivity issues")
        print("- Invalid API key")
        print("- Rate limiting")
        print("- Model availability issues")
        
        # Show partial results if available
        if pipeline_results["step1_refined_prompt"]:
            print(f"\n✅ Step 1 completed - Refined Prompt:")
            print(f"'{pipeline_results['step1_refined_prompt']}'")
        
        if pipeline_results["step2_rag_response"]:
            print(f"\n✅ Step 2 completed - RAG Response generated")
            print(f"Response length: {len(pipeline_results['step2_rag_response'])} characters")
    
    print(f"\n🎯 Pipeline Processing Complete!")

if __name__ == "__main__":
    # Install required packages if not available
    required_packages = ['aiohttp', 'feedparser', 'beautifulsoup4', 'requests']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            import subprocess
            subprocess.check_call(["pip", "install", package])
    
    asyncio.run(main())