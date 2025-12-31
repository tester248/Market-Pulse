"""
OpenRouter Integration for Market Pulse

This module provides the integration layer between the OpenRouter LLM Manager
and the existing Market Pulse production API.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import OpenRouter LLM Manager
from openrouter_llm_manager import OpenRouterMultiLLMManager, ModelType

# Configure logging
logger = logging.getLogger(__name__)

class OpenRouterAssemblyLine:
    """
    Assembly Line implementation using OpenRouter LLM Manager
    Acts as a drop-in replacement for the original Ollama-based Assembly Line
    """
    
    def __init__(self, api_key: str = None):
        self.llm_manager = OpenRouterMultiLLMManager(api_key=api_key)
        
    async def close(self):
        """Clean up resources"""
        await self.llm_manager.close()
        
    async def process_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a financial article through the assembly line
        
        Args:
            article_data: Dict containing article content and metadata
        
        Returns:
            Dict with processed results including insights
        """
        content = article_data.get("content", "")
        title = article_data.get("title", "")
        source = article_data.get("source", "")
        url = article_data.get("url", "")
        
        # Create metadata for tracking
        metadata = {
            "title": title,
            "source": source,
            "url": url,
            "process_id": article_data.get("id", str(hash(title + content))),
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Triage the content
        logger.info(f"Triage stage: Classifying content for {title}")
        triage_result = await self.llm_manager.run_triage(
            f"Title: {title}\n\nContent: {content}", 
            metadata=metadata
        )
        
        if not triage_result.success:
            logger.error(f"Triage failed: {triage_result.error_message}")
            return {
                "success": False,
                "error": f"Triage failed: {triage_result.error_message}",
                "article_id": metadata["process_id"],
                "stages_completed": ["triage"]
            }
            
        # Parse triage result
        try:
            triage_data = json.loads(triage_result.content)
            specialist_needed = triage_data.get("specialist_needed", "general")
        except:
            # Default to sentiment if parsing fails
            specialist_needed = "sentiment"
            triage_data = {"category": "unknown", "tickers": [], "sentiment": "neutral"}
        
        # Step 2: Route to appropriate specialist
        if specialist_needed == "sentiment":
            # Run sentiment analysis
            logger.info(f"Sentiment analysis stage for {title}")
            specialist_result = await self.llm_manager.run_sentiment_analysis(
                f"Title: {title}\n\nContent: {content}",
                metadata=metadata
            )
            specialist_type = "sentiment"
        
        elif specialist_needed == "extraction":
            # Run data extraction
            logger.info(f"Data extraction stage for {title}")
            specialist_result = await self.llm_manager.run_data_extraction(
                f"Title: {title}\n\nContent: {content}",
                metadata=metadata
            )
            specialist_type = "extraction"
        
        else:
            # Run general analysis
            logger.info(f"General analysis stage for {title}")
            specialist_result = await self.llm_manager.run_general_analysis(
                f"Title: {title}\n\nContent: {content}",
                metadata=metadata
            )
            specialist_type = "general"
        
        if not specialist_result.success:
            logger.error(f"{specialist_type} analysis failed: {specialist_result.error_message}")
            return {
                "success": False,
                "error": f"{specialist_type} analysis failed: {specialist_result.error_message}",
                "article_id": metadata["process_id"],
                "stages_completed": ["triage", specialist_type + "_failed"],
                "triage_result": triage_data
            }
        
        # Step 3: Final synthesis with the general model
        synthesis_prompt = f"""
Title: {title}
Source: {source}

Content summary: {content[:500]}...

Triage classification: {triage_result.content}

Specialist analysis ({specialist_type}): {specialist_result.content}

Provide a comprehensive synthesis of this financial information, highlighting:
1. Key insights and implications
2. Market impact assessment
3. Actionable takeaways for investors
"""
        
        logger.info(f"Synthesis stage for {title}")
        synthesis_result = await self.llm_manager.run_general_analysis(
            synthesis_prompt,
            metadata=metadata
        )
        
        # Prepare final result
        if synthesis_result.success:
            # Try to parse specialist JSON
            try:
                specialist_data = json.loads(specialist_result.content)
            except:
                specialist_data = {"error": "Could not parse specialist output"}
            
            # Extract sentiment score
            sentiment_score = 0.0
            if specialist_type == "sentiment":
                try:
                    sentiment_score = float(specialist_data.get("sentiment_score", 0.0))
                except:
                    pass
            
            return {
                "success": True,
                "article_id": metadata["process_id"],
                "title": title,
                "source": source,
                "url": url,
                "triage": triage_data,
                "specialist": {
                    "type": specialist_type,
                    "data": specialist_data
                },
                "sentiment_score": sentiment_score,
                "synthesis": synthesis_result.content,
                "tickers": triage_data.get("tickers", []),
                "processing_times": {
                    "triage_ms": triage_result.processing_time_ms,
                    "specialist_ms": specialist_result.processing_time_ms,
                    "synthesis_ms": synthesis_result.processing_time_ms,
                    "total_ms": triage_result.processing_time_ms + 
                              specialist_result.processing_time_ms +
                              synthesis_result.processing_time_ms
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": f"Synthesis failed: {synthesis_result.error_message}",
                "article_id": metadata["process_id"],
                "stages_completed": ["triage", specialist_type, "synthesis_failed"],
                "triage_result": triage_data
            }
    
    async def process_user_query(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the OpenRouter pipeline
        
        Args:
            query: The user's financial question
            use_rag: Whether to use RAG with recent financial news
            
        Returns:
            Dict with processed results
        """
        logger.info(f"Processing user query: {query}")
        
        # Step 1: Refine the query
        refine_prompt = f"""
Please refine this financial question to make it more comprehensive and specific: '{query}'

When refining the question:
1. Make it more specific and detailed
2. Add context that would help generate better financial insights
3. Include relevant financial terminology if appropriate
4. Ensure the refined prompt will elicit comprehensive analysis
5. Keep the core intent of the original question
"""
        
        refined_result = await self.llm_manager.run_general_analysis(
            refine_prompt,
            metadata={"original_query": query}
        )
        
        if not refined_result.success:
            logger.error(f"Query refinement failed: {refined_result.error_message}")
            # Fall back to original query
            refined_query = query
        else:
            refined_query = refined_result.content
        
        # Step 2: Generate response with RAG
        logger.info(f"Generating response with RAG={use_rag}")
        try:
            # Add 30 second timeout for RAG response
            async with asyncio.timeout(30):
                response_result = await self.llm_manager.run_general_analysis(
                    f"Answer this financial question with detailed analysis: {refined_query}",
                    query=refined_query,
                    use_rag=use_rag,
                    metadata={"original_query": query, "refined_query": refined_query}
                )
        except asyncio.TimeoutError:
            logger.error("Response generation timed out after 30 seconds")
            return {
                "success": False,
                "error": "Response generation timed out. Please try again.",
                "original_query": query,
                "refined_query": refined_query,
                "stages_completed": ["refine", "response_timeout"],
                "processing_times": {
                    "refine_ms": refined_result.processing_time_ms,
                    "response_ms": 30000,  # 30 seconds timeout
                    "total_ms": refined_result.processing_time_ms + 30000
                }
            }
        except Exception as e:
            logger.error(f"Response generation failed with error: {str(e)}")
            return {
                "success": False,
                "error": f"Response generation failed: {str(e)}",
                "original_query": query,
                "refined_query": refined_query,
                "stages_completed": ["refine", "response_error"],
                "processing_times": {
                    "refine_ms": refined_result.processing_time_ms,
                    "response_ms": 0,
                    "total_ms": refined_result.processing_time_ms
                }
            }
        
        if not response_result.success:
            logger.error(f"Response generation failed: {response_result.error_message}")
            return {
                "success": False,
                "error": response_result.error_message,
                "original_query": query,
                "refined_query": refined_query,
                "stages_completed": ["refine", "response_failed"]
            }
        
        # Step 3: Structure the output
        structure_prompt = f"""
Original query: {query}

Refined query: {refined_query}

Raw response: {response_result.content}

Convert the above response into a well-structured format with the following sections:
1. Summary (2-3 sentences of key points)
2. Detailed Analysis (main body of the response)
3. Market Implications (what this means for investors)
4. Sources Used (mention any specific sources referenced)

Format the response professionally and ensure it directly answers the original query.
"""
        
        structure_result = await self.llm_manager.run_general_analysis(
            structure_prompt,
            metadata={
                "original_query": query,
                "refined_query": refined_query,
                "articles_processed": response_result.metadata.get("articles_processed", 0)
            }
        )
        
        # Prepare final result
        return {
            "success": structure_result.success,
            "original_query": query,
            "refined_query": refined_query,
            "response": structure_result.content if structure_result.success else response_result.content,
            "articles_processed": response_result.metadata.get("articles_processed", 0),
            "processing_times": {
                "refine_ms": refined_result.processing_time_ms,
                "response_ms": response_result.processing_time_ms,
                "structure_ms": structure_result.processing_time_ms if structure_result.success else 0,
                "total_ms": refined_result.processing_time_ms + 
                          response_result.processing_time_ms +
                          (structure_result.processing_time_ms if structure_result.success else 0)
            },
            "error": structure_result.error_message if not structure_result.success else None,
            "timestamp": datetime.now().isoformat()
        }
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all models"""
        return await self.llm_manager.get_model_health()

# Factory function to create and return a new assembly line
def get_openrouter_assembly_line(api_key: str = None):
    """Create and return a new OpenRouter Assembly Line instance"""
    return OpenRouterAssemblyLine(api_key=api_key)