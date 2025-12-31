"""
Production Ollama Multi-LLM Manager

Orchestrates multiple specialized finance models via Ollama with configuration management.
Provides model management, health monitoring, load balancing, and intelligent routing.
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from config_manager import get_config
from enum import Enum
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of specialized finance models"""
    TRIAGE = "triage"           # Content classification and routing
    SENTIMENT = "sentiment"     # Sentiment analysis (FinBERT-style)
    EXTRACTION = "extraction"   # Data extraction and summarization
    GENERAL = "general"         # General-purpose finance model

@dataclass
class ModelConfig:
    """Configuration for a specialized model"""
    name: str                   # Model name in Ollama
    model_type: ModelType       # Type of specialist
    ollama_model: str          # Actual model name for Ollama API
    description: str           # What this model does
    max_context: int = 4096    # Maximum context length
    temperature: float = 0.1   # Temperature for inference
    max_concurrent: int = 3    # Max concurrent requests
    timeout_seconds: int = 300  # Request timeout (5 minutes for financial models)
    enabled: bool = True       # Whether model is active

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

class OllamaMultiLLMManager:
    """Production multi-LLM manager using configuration"""
    
    def __init__(self):
        # Load configuration
        self.config = get_config()
        self.ollama_config = self.config.ollama_config
        self.model_configs = self.config.model_configs
        
        # Set Ollama base URL from configuration
        self.ollama_base_url = self.ollama_config.base_url
        
        # HTTP session for Ollama API
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Model mapping from types to actual names
        self.model_mapping = {
            ModelType.TRIAGE: self.config.get_model_name('triage'),
            ModelType.SENTIMENT: self.config.get_model_name('sentiment'),
            ModelType.EXTRACTION: self.config.get_model_name('extraction'),
            ModelType.GENERAL: self.config.get_model_name('general')
        }
        
        # Model configurations
        self.models: Dict[str, ModelConfig] = {}
        self.model_health: Dict[str, ModelHealth] = {}
        self.model_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0.0,
            'models_loaded': 0,
            'start_time': time.time()
        }
        
        # Load balancing
        self.request_counts: Dict[str, int] = {}
        
        # Health monitoring
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_interval = 300  # seconds between health checks (5 minutes)
        self.last_request_time = None  # Track when last request was made
        self.idle_threshold = 600  # Skip health checks if idle for 10 minutes
        self.running = False

    async def initialize(self):
        """Initialize the LLM manager"""
        logger.info("🚀 Initializing Ollama Multi-LLM Manager")
        
        try:
            # Create HTTP session with configured timeout
            connector = aiohttp.TCPConnector(
                limit=50,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.ollama_config.timeout_seconds)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            # Load default model configurations
            await self._load_default_models()
            
            # Check Ollama availability
            await self._check_ollama_health()
            
            # Initialize model health tracking
            await self._initialize_model_health()
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitor_loop())
            self.running = True
            
            logger.info("✅ Ollama Multi-LLM Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Manager: {e}")
            raise

    async def _load_default_models(self):
        """Load model configurations from config"""
        for model_type, model_config in self.model_configs.items():
            ollama_model_config = ModelConfig(
                name=f"{model_type}_specialist",
                model_type=ModelType(model_type),
                ollama_model=model_config.name,  # Use actual model name from config
                description=f"{model_type.title()} specialist for financial analysis",
                max_context=model_config.context_window,
                temperature=model_config.temperature,
                max_concurrent=3  # Could be made configurable
            )
            await self.register_model(ollama_model_config)
        
        logger.info(f"📋 Loaded {len(self.model_configs)} model configurations from config")

    async def register_model(self, model_config: ModelConfig):
        """Register a new specialized model"""
        self.models[model_config.name] = model_config
        self.model_semaphores[model_config.name] = asyncio.Semaphore(model_config.max_concurrent)
        self.request_counts[model_config.name] = 0
        
        logger.info(f"📝 Registered model: {model_config.name} ({model_config.model_type.value})")

    async def _check_ollama_health(self):
        """Check if Ollama is running and accessible"""
        try:
            async with self.session.get(f"{self.ollama_base_url}/api/version") as response:
                if response.status == 200:
                    version_info = await response.json()
                    logger.info(f"✅ Ollama is running: {version_info}")
                    return True
                else:
                    raise Exception(f"Ollama returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Ollama health check failed: {e}")
            raise Exception(f"Ollama is not accessible at {self.ollama_base_url}")

    async def _initialize_model_health(self):
        """Initialize health tracking for all models"""
        for model_name, model_config in self.models.items():
            health = await self._check_model_health(model_name)
            self.model_health[model_name] = health

    async def _check_model_health(self, model_name: str) -> ModelHealth:
        """Check health of a specific model"""
        model_config = self.models[model_name]
        start_time = time.time()
        
        try:
            # Send a simple test query
            test_prompt = "Test"
            response = await self._call_ollama_model(
                model_config.ollama_model, 
                test_prompt,
                max_tokens=10,
                timeout=10
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response:
                return ModelHealth(
                    model_name=model_name,
                    is_healthy=True,
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    success_count=1
                )
            else:
                return ModelHealth(
                    model_name=model_name,
                    is_healthy=False,
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    error_count=1
                )
                
        except Exception as e:
            logger.warning(f"⚠️ Health check failed for {model_name}: {e}")
            return ModelHealth(
                model_name=model_name,
                is_healthy=False,
                last_check=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_count=1
            )

    async def _call_ollama_model(self, 
                                model_name: str, 
                                prompt: str,
                                max_tokens: int = 1024,
                                temperature: float = 0.1,
                                timeout: int = 30) -> Optional[str]:
        """Make a direct call to Ollama model"""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            async with self.session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error calling Ollama model {model_name}: {e}")
            return None

    async def _unload_model(self, model_name: str) -> bool:
        """Unload a model from VRAM"""
        try:
            # Tell Ollama to unload the model by setting keep_alive to 0
            payload = {
                "model": model_name,
                "keep_alive": 0
            }
            
            async with self.session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    logger.debug(f"🗑️ Unloaded model {model_name}")
                    return True
                else:
                    logger.warning(f"Failed to unload model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.warning(f"Error unloading model {model_name}: {e}")
            return False

    async def query_specialist(self, 
                             model_type: ModelType, 
                             prompt: str,
                             preferred_model: Optional[str] = None,
                             **kwargs) -> ModelResponse:
        """Query a specialist model by type"""
        
        # Find appropriate model
        target_model = None
        if preferred_model and preferred_model in self.models:
            target_model = preferred_model
        else:
            # Find first healthy model of the requested type
            for model_name, model_config in self.models.items():
                if (model_config.model_type == model_type and 
                    model_config.enabled and
                    self.model_health.get(model_name, {}).is_healthy):
                    target_model = model_name
                    break
        
        if not target_model:
            return ModelResponse(
                model_name="none",
                model_type=model_type,
                content="",
                metadata={},
                processing_time_ms=0,
                success=False,
                error_message=f"No healthy {model_type.value} model available"
            )
        
        return await self.query_model(target_model, prompt, **kwargs)

    async def query_model(self, 
                         model_name: str, 
                         prompt: str,
                         **kwargs) -> ModelResponse:
        """Query a specific model"""
        
        # Track request time to determine system activity
        self.last_request_time = time.time()
        
        if model_name not in self.models:
            return ModelResponse(
                model_name=model_name,
                model_type=ModelType.GENERAL,
                content="",
                metadata={},
                processing_time_ms=0,
                success=False,
                error_message=f"Model {model_name} not found"
            )
        
        model_config = self.models[model_name]
        start_time = time.time()
        
        # Use semaphore for concurrency control
        async with self.model_semaphores[model_name]:
            try:
                # Prepare parameters
                temperature = kwargs.get('temperature', model_config.temperature)
                max_tokens = kwargs.get('max_tokens', 1024)
                timeout = kwargs.get('timeout', model_config.timeout_seconds)
                
                # Call the model
                response_text = await self._call_ollama_model(
                    model_config.ollama_model,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                if response_text:
                    # Update stats
                    self.stats['successful_requests'] += 1
                    self.request_counts[model_name] += 1
                    
                    # Update health
                    if model_name in self.model_health:
                        self.model_health[model_name].success_count += 1
                    
                    return ModelResponse(
                        model_name=model_name,
                        model_type=model_config.model_type,
                        content=response_text,
                        metadata={
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "ollama_model": model_config.ollama_model
                        },
                        processing_time_ms=processing_time,
                        success=True
                    )
                else:
                    raise Exception("Empty response from model")
                    
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                self.stats['failed_requests'] += 1
                
                # Update health
                if model_name in self.model_health:
                    self.model_health[model_name].error_count += 1
                
                logger.error(f"❌ Query failed for {model_name}: {e}")
                
                return ModelResponse(
                    model_name=model_name,
                    model_type=model_config.model_type,
                    content="",
                    metadata={},
                    processing_time_ms=processing_time,
                    success=False,
                    error_message=str(e)
                )
            finally:
                self.stats['total_requests'] += 1
                
                # Unload model from VRAM if VRAM management is enabled
                vram_management = getattr(self.ollama_config, 'vram_management', False)
                if vram_management:
                    await self._unload_model(model_config.ollama_model)
                    await asyncio.sleep(2)  # Brief pause after unloading

    async def _health_monitor_loop(self):
        """Continuous health monitoring of all models"""
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Skip health checks if system has been idle for too long
                current_time = time.time()
                if (self.last_request_time is not None and 
                    current_time - self.last_request_time > self.idle_threshold):
                    logger.info("💤 Skipping health check - system idle")
                    continue
                
                logger.info("🏥 Running model health checks...")
                
                for model_name in self.models:
                    health = await self._check_model_health(model_name)
                    self.model_health[model_name] = health
                    
                    if not health.is_healthy:
                        logger.warning(f"⚠️ Model {model_name} is unhealthy")
                
                # Log overall stats
                healthy_models = sum(1 for h in self.model_health.values() if h.is_healthy)
                total_models = len(self.models)
                
                logger.info(f"📊 Health check complete: {healthy_models}/{total_models} models healthy")
                
            except Exception as e:
                logger.error(f"❌ Health monitoring error: {e}")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with health status"""
        result = []
        
        for model_name, model_config in self.models.items():
            health = self.model_health.get(model_name)
            
            result.append({
                "name": model_name,
                "type": model_config.model_type.value,
                "ollama_model": model_config.ollama_model,
                "description": model_config.description,
                "enabled": model_config.enabled,
                "healthy": health.is_healthy if health else False,
                "response_time_ms": health.response_time_ms if health else 0,
                "request_count": self.request_counts.get(model_name, 0)
            })
        
        return result

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        runtime = time.time() - self.stats['start_time']
        
        return {
            "runtime_seconds": runtime,
            "total_requests": self.stats['total_requests'],
            "successful_requests": self.stats['successful_requests'],
            "failed_requests": self.stats['failed_requests'],
            "success_rate": self.stats['successful_requests'] / max(self.stats['total_requests'], 1),
            "requests_per_second": self.stats['total_requests'] / max(runtime, 1),
            "models_registered": len(self.models),
            "healthy_models": sum(1 for h in self.model_health.values() if h.is_healthy),
            "model_health": {name: h.is_healthy for name, h in self.model_health.items()},
            "request_distribution": dict(self.request_counts),
            "ollama_url": self.ollama_base_url
        }

    async def cleanup(self):
        """Clean up resources"""
        logger.info("🛑 Shutting down Ollama Multi-LLM Manager")
        
        self.running = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("✅ LLM Manager shutdown complete")

# Utility functions for easy usage
async def create_llm_manager(ollama_url: Optional[str] = None) -> OllamaMultiLLMManager:
    """Create and initialize LLM manager using configuration"""
    if ollama_url is None:
        config = get_config()
        ollama_config = config.ollama_config
        ollama_url = ollama_config.base_url
    
    manager = OllamaMultiLLMManager()
    await manager.initialize()
    return manager

# Demo function
async def demo_multi_llm_manager():
    """Demonstrate the Multi-LLM Manager"""
    print("🚀 Ollama Multi-LLM Manager Demo")
    print("=" * 40)
    
    try:
        # Initialize manager
        manager = await create_llm_manager()
        
        # Show available models
        models = await manager.get_available_models()
        print("📋 Available Models:")
        for model in models:
            status = "✅ Healthy" if model['healthy'] else "❌ Unhealthy"
            print(f"   {model['name']} ({model['type']}): {status}")
        
        # Test triage model
        print("\n🔍 Testing Triage Model:")
        triage_response = await manager.query_specialist(
            ModelType.TRIAGE,
            "Apple reported strong quarterly earnings with revenue up 15%"
        )
        
        if triage_response.success:
            print(f"   ✅ Response: {triage_response.content[:100]}...")
            print(f"   ⏱️ Time: {triage_response.processing_time_ms:.1f}ms")
        else:
            print(f"   ❌ Error: {triage_response.error_message}")
        
        # Get stats
        stats = await manager.get_stats()
        print(f"\n📊 Stats:")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Healthy models: {stats['healthy_models']}/{stats['models_registered']}")
        
        await manager.cleanup()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_multi_llm_manager())