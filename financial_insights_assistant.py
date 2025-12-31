"""
Financial Insights Assistant - Core Engine
Uses finance-llm-13b.Q5_K_S.gguf for financial sentiment analysis, summarization, and Q&A
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from finance_llm_provider import FinanceLLMProvider, create_finance_llm_provider


class SentimentType(str, Enum):
    """Sentiment classification"""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: SentimentType
    confidence: float
    reasoning: str
    key_factors: List[str]


@dataclass
class SummaryResult:
    """Summarization result"""
    bullet_points: List[str]
    key_metrics: Dict[str, str]
    main_themes: List[str]
    impact_assessment: str


@dataclass
class QAResult:
    """Q&A result"""
    answer: str
    confidence: float
    sources_referenced: List[str]
    follow_up_questions: List[str]


class FinancialInsightsAssistant:
    """
    Financial Insights Assistant using Ollama for:
    1. Sentiment Analysis
    2. Financial Summarization  
    3. Q&A on Financial Topics
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.provider = create_finance_llm_provider(model_path)
        self.conversation_history = []
    
    def validate_connection(self) -> bool:
        """Check if Finance LLM is available and loaded"""
        return self.provider.validate_connection()
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """Async wrapper for the sync provider"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.provider.generate_response, prompt, **kwargs)
    
    async def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Async wrapper for JSON parsing"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.provider.parse_json_response, response_text)
    
    async def analyze_sentiment(self, text: str, context: str = "general") -> SentimentResult:
        """
        Analyze sentiment of financial text/news
        
        Args:
            text: Financial text to analyze
            context: Context type (earnings, market_news, analyst_report, etc.)
        
        Returns:
            SentimentResult with classification and reasoning
        """
        prompt = f"""Analyze the sentiment of this financial text and classify it as Positive, Negative, or Neutral.

Financial Text:
{text}

Context: {context}

Please provide your analysis in the following JSON format:
{{
    "sentiment": "Positive|Negative|Neutral",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this sentiment was chosen",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Focus on:
- Financial performance indicators (revenue, profit, growth)
- Market impact and investor sentiment
- Business outlook and future prospects
- Risk factors and challenges

JSON Response:"""

        try:
            response = await self._generate_response(prompt)
            
            # Parse JSON response using the provider's parser
            result_data = await self._parse_json_response(response)
            
            return SentimentResult(
                sentiment=SentimentType(result_data.get("sentiment", "Neutral")),
                confidence=float(result_data.get("confidence", 0.5)),
                reasoning=result_data.get("reasoning", ""),
                key_factors=result_data.get("key_factors", [])
            )
                
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            
            # Fallback: try to determine sentiment from keywords
            text_lower = text.lower()
            if any(word in text_lower for word in ['growth', 'increase', 'profit', 'positive', 'strong', 'record']):
                sentiment = SentimentType.POSITIVE
            elif any(word in text_lower for word in ['decline', 'decrease', 'loss', 'negative', 'weak', 'poor']):
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.NEUTRAL
                
            return SentimentResult(
                sentiment=sentiment,
                confidence=0.5,
                reasoning=f"Fallback analysis due to error: {str(e)}",
                key_factors=[]
            )
    
    async def summarize_financial_content(self, text: str, content_type: str = "report") -> SummaryResult:
        """
        Summarize financial reports/news into bullet points
        
        Args:
            text: Financial content to summarize
            content_type: Type of content (earnings_report, news_article, analyst_report)
        
        Returns:
            SummaryResult with bullet points and key metrics
        """
        prompt = f"""Summarize this financial content into clear, actionable bullet points with key metrics.

Content Type: {content_type}
Text:
{text}

Please provide a structured summary in JSON format:
{{
    "bullet_points": [
        "Revenue ↑ 8% to $2.5B",
        "Net profit ↓ 3% to $450M", 
        "Key strategic initiative launched"
    ],
    "key_metrics": {{
        "Revenue": "+8%",
        "Profit": "-3%",
        "Guidance": "Raised"
    }},
    "main_themes": ["Growth", "Profitability", "Strategy"],
    "impact_assessment": "Overall positive with strong revenue growth offsetting margin pressure"
}}

Focus on:
- Quantifiable metrics with percentage changes
- Financial performance indicators
- Strategic developments and initiatives
- Market position and competitive dynamics
- Forward-looking guidance and outlook

Use symbols: ↑ for increases, ↓ for decreases, → for flat/unchanged

JSON Response:"""

        try:
            response = await self._generate_response(prompt)
            
            # Parse JSON response using the provider's parser
            result_data = await self._parse_json_response(response)
            
            return SummaryResult(
                bullet_points=result_data.get("bullet_points", []),
                key_metrics=result_data.get("key_metrics", {}),
                main_themes=result_data.get("main_themes", []),
                impact_assessment=result_data.get("impact_assessment", "")
            )
                
        except Exception as e:
            print(f"Error in summarization: {e}")
            
            # Fallback: create basic summary
            lines = text.split('\n')[:5]  # Take first 5 lines
            bullet_points = [f"• {line.strip()}" for line in lines if line.strip()]
            
            return SummaryResult(
                bullet_points=bullet_points,
                key_metrics={},
                main_themes=["summary"],
                impact_assessment=f"Basic summary due to error: {str(e)}"
            )
    
    async def answer_financial_question(self, question: str, context: str = "") -> QAResult:
        """
        Answer financial questions using available context
        
        Args:
            question: Financial question to answer
            context: Additional context/data to help answer
        
        Returns:
            QAResult with answer and supporting information
        """
        # Add conversation history for context
        conversation_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            conversation_context = "Previous conversation:\n" + "\n".join([
                f"Q: {item['question']}\nA: {item['answer'][:200]}..."
                for item in recent_history
            ]) + "\n\n"
        
        prompt = f"""{conversation_context}Answer this financial question clearly and accurately.

Question: {question}

Additional Context:
{context}

Please provide your response in JSON format:
{{
    "answer": "Detailed answer to the question",
    "confidence": 0.85,
    "sources_referenced": ["context_source1", "general_knowledge"],
    "follow_up_questions": ["Related question 1", "Related question 2"]
}}

Guidelines:
- Provide specific, actionable answers
- Include relevant financial metrics when possible
- Cite sources when using provided context
- Suggest logical follow-up questions
- If uncertain, state limitations clearly

JSON Response:"""

        try:
            response = await self._generate_response(prompt)
            
            # Parse JSON response using the provider's parser
            result_data = await self._parse_json_response(response)
            
            result = QAResult(
                answer=result_data.get("answer", ""),
                confidence=float(result_data.get("confidence", 0.7)),
                sources_referenced=result_data.get("sources_referenced", []),
                follow_up_questions=result_data.get("follow_up_questions", [])
            )
            
            # Add to conversation history
            self.conversation_history.append({
                "question": question,
                "answer": result.answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 conversations
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return result
            
        except Exception as e:
            print(f"Error in Q&A: {e}")
            return QAResult(
                answer=f"Sorry, I encountered an error: {str(e)}",
                confidence=0.0,
                sources_referenced=[],
                follow_up_questions=[]
            )
    
    async def comprehensive_analysis(self, text: str, content_type: str = "mixed") -> Dict[str, Any]:
        """
        Perform comprehensive analysis combining all three capabilities
        
        Args:
            text: Financial content to analyze
            content_type: Type of content being analyzed
        
        Returns:
            Combined results from sentiment, summary, and insights
        """
        results = {}
        
        try:
            # Run all analyses in parallel
            sentiment_task = self.analyze_sentiment(text, content_type)
            summary_task = self.summarize_financial_content(text, content_type)
            
            sentiment_result, summary_result = await asyncio.gather(
                sentiment_task, summary_task
            )
            
            results["sentiment"] = sentiment_result
            results["summary"] = summary_result
            
            # Generate insights question
            insights_question = f"What are the key insights and implications from this {content_type}?"
            insights_result = await self.answer_financial_question(
                insights_question, 
                f"Content: {text[:1000]}..."  # First 1000 chars as context
            )
            results["insights"] = insights_result
            
            results["analysis_timestamp"] = datetime.now().isoformat()
            results["status"] = "success"
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def _extract_sentiment_fallback(self, text: str) -> SentimentType:
        """Fallback sentiment extraction from text"""
        text_lower = text.lower()
        
        positive_words = ["positive", "good", "strong", "growth", "increase", "bullish", "optimistic"]
        negative_words = ["negative", "poor", "weak", "decline", "decrease", "bearish", "pessimistic"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return SentimentType.POSITIVE
        elif negative_count > positive_count:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def _extract_summary_fallback(self, text: str) -> SummaryResult:
        """Fallback summary extraction"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        bullet_points = [line for line in lines if line.startswith('-') or line.startswith('•')][:5]
        
        return SummaryResult(
            bullet_points=bullet_points if bullet_points else lines[:3],
            key_metrics={},
            main_themes=["Analysis"],
            impact_assessment="Summary extracted from response"
        )
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Example usage and testing
async def demo_financial_insights():
    """Demo the Financial Insights Assistant"""
    assistant = FinancialInsightsAssistant()
    
    print("🧠 Financial Insights Assistant Demo")
    print("=" * 50)
    
    # Check connection
    if not assistant.validate_connection():
        print("❌ Ollama not available. Please start with: ollama serve")
        return
    
    print("✅ Connected to Ollama")
    
    # Sample financial content
    sample_earnings = """
    Apple Inc. reported quarterly revenue of $94.8 billion, up 8% year-over-year, 
    beating analyst expectations of $92.1 billion. iPhone sales grew 12% to $51.3 billion, 
    while Services revenue increased 6% to $22.3 billion. However, iPad sales declined 
    3% to $7.2 billion due to supply chain constraints. The company raised its full-year 
    guidance and announced a $25 billion share buyback program. CEO Tim Cook noted strong 
    demand in emerging markets and highlighted the success of the iPhone 15 Pro models.
    """
    
    print("\n📊 Sample Analysis: Apple Earnings Report")
    print("-" * 40)
    
    # Comprehensive analysis
    results = await assistant.comprehensive_analysis(sample_earnings, "earnings_report")
    
    if results["status"] == "success":
        # Display sentiment
        sentiment = results["sentiment"]
        print(f"\n💭 Sentiment: {sentiment.sentiment.value}")
        print(f"   Confidence: {sentiment.confidence:.1%}")
        print(f"   Reasoning: {sentiment.reasoning}")
        
        # Display summary
        summary = results["summary"]
        print(f"\n📋 Summary:")
        for point in summary.bullet_points:
            print(f"   • {point}")
        
        if summary.key_metrics:
            print(f"\n📈 Key Metrics:")
            for metric, value in summary.key_metrics.items():
                print(f"   {metric}: {value}")
        
        # Display insights
        insights = results["insights"]
        print(f"\n🎯 Key Insights:")
        print(f"   {insights.answer}")
        
        if insights.follow_up_questions:
            print(f"\n❓ Follow-up Questions:")
            for i, question in enumerate(insights.follow_up_questions, 1):
                print(f"   {i}. {question}")
    
    # Interactive Q&A demo
    print(f"\n🤔 Q&A Demo:")
    questions = [
        "What drove Apple's revenue growth this quarter?",
        "Should investors be concerned about iPad sales decline?",
        "How does this compare to previous quarters?"
    ]
    
    for question in questions:
        print(f"\n❓ {question}")
        answer = await assistant.answer_financial_question(question, sample_earnings)
        print(f"💡 {answer.answer}")


if __name__ == "__main__":
    asyncio.run(demo_financial_insights())