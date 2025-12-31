"""
GGUF Model Provider for Financial Insights Assistant
Replaces Ollama with finance-llm-13b.Q5_K_S.gguf local model
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")


@dataclass
class GGUFConfig:
    """Configuration for GGUF model"""
    model_path: str
    n_ctx: int = 2048  # Match model's training context to avoid overflow warnings
    n_threads: int = 4
    n_gpu_layers: int = 0  # Set to > 0 if you have GPU
    verbose: bool = False
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9
    repeat_penalty: float = 1.1


class FinanceLLMProvider:
    """
    Finance-specific LLM provider using finance-llm-13b.Q5_K_S.gguf
    Replaces Ollama with local GGUF model loading
    """
    
    def __init__(self, config: Optional[GGUFConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
        
        # Default configuration
        if config is None:
            config = GGUFConfig(
                model_path=self._find_model_path(),
                n_ctx=2048,  # Match model's training context
                n_threads=4,
                temperature=0.3,
                max_tokens=1000
            )
        
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _find_model_path(self) -> str:
        """Find the finance-llm-13b.Q5_K_S.gguf file"""
        possible_paths = [
            # Current directory
            "finance-llm-13b.Q5_K_S.gguf",
            "./finance-llm-13b.Q5_K_S.gguf",
            
            # Common model directories
            "models/finance-llm-13b.Q5_K_S.gguf",
            "../models/finance-llm-13b.Q5_K_S.gguf",
            "../../models/finance-llm-13b.Q5_K_S.gguf",
            
            # Downloads folder
            f"{Path.home()}/Downloads/finance-llm-13b.Q5_K_S.gguf",
            
            # Desktop
            f"{Path.home()}/Desktop/finance-llm-13b.Q5_K_S.gguf",
            
            # E drive locations
            "E:/finance-llm-13b.Q5_K_S.gguf",
            "E:/models/finance-llm-13b.Q5_K_S.gguf",
            "E:/Downloads/finance-llm-13b.Q5_K_S.gguf",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                self.logger.info(f"Found model at: {path}")
                return str(Path(path).absolute())
        
        # If not found, return a default path with helpful error
        default_path = "./finance-llm-13b.Q5_K_S.gguf"
        self.logger.warning(f"Model not found. Please place finance-llm-13b.Q5_K_S.gguf at: {default_path}")
        return default_path
    
    def _initialize_model(self):
        """Initialize the GGUF model"""
        try:
            self.logger.info(f"Loading model from: {self.config.model_path}")
            
            if not Path(self.config.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
            
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=self.config.verbose
            )
            
            self.logger.info("✅ Finance LLM model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def validate_connection(self) -> bool:
        """Validate that the model is loaded and working"""
        try:
            if self.model is None:
                return False
            
            # Test with a simple prompt
            test_response = self.model(
                "What is finance?",
                max_tokens=50,
                temperature=0.1,
                echo=False
            )
            
            return bool(test_response and test_response.get('choices'))
            
        except Exception as e:
            self.logger.error(f"Connection validation failed: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the finance LLM"""
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            
            # Merge kwargs with default config
            generation_params = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
                "echo": False,
                "stop": kwargs.get("stop", ["Human:", "Assistant:", "\n\n"])
            }
            
            self.logger.debug(f"Generating response with params: {generation_params}")
            
            response = self.model(**generation_params)
            
            if response and response.get('choices'):
                text = response['choices'][0]['text'].strip()
                self.logger.debug(f"Generated response: {text[:100]}...")
                return text
            else:
                raise RuntimeError("No response generated")
                
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            raise
    
    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from model response with fallback handling"""
        try:
            # Try to find JSON in the response
            json_pattern = r'```json\n(.*?)\n```'
            json_match = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_match:
                json_text = json_match.group(1)
            else:
                # Look for JSON-like structure
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                else:
                    json_text = response_text
            
            # Clean up common issues
            json_text = json_text.strip()
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
            
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {str(e)}")
            
            # Fallback: try to extract key-value pairs
            fallback_result = {}
            
            # Extract common patterns
            patterns = {
                'sentiment': r'"sentiment":\s*"([^"]+)"',
                'confidence': r'"confidence":\s*([0-9.]+)',
                'reasoning': r'"reasoning":\s*"([^"]+)"',
                'answer': r'"answer":\s*"([^"]+)"',
                'summary': r'"summary":\s*"([^"]+)"'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if key == 'confidence':
                        try:
                            fallback_result[key] = float(value)
                        except ValueError:
                            fallback_result[key] = 0.7
                    else:
                        fallback_result[key] = value
            
            if fallback_result:
                self.logger.info("Used fallback parsing")
                return fallback_result
            
            raise ValueError(f"Unable to parse response: {response_text[:200]}...")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "finance-llm-13b.Q5_K_S",
            "model_path": self.config.model_path,
            "model_type": "GGUF",
            "context_length": self.config.n_ctx,
            "quantization": "Q5_K_S",
            "parameters": "13B",
            "specialized": "Finance Domain",
            "loaded": self.model is not None
        }


def create_finance_llm_provider(model_path: Optional[str] = None, **kwargs) -> FinanceLLMProvider:
    """Factory function to create FinanceLLMProvider"""
    config = None
    
    if model_path:
        config = GGUFConfig(model_path=model_path, **kwargs)
    
    return FinanceLLMProvider(config)


# Test function
def test_finance_llm():
    """Test the finance LLM provider"""
    try:
        print("🧠 Testing Finance LLM Provider")
        
        provider = create_finance_llm_provider()
        
        print(f"📊 Model Info: {provider.get_model_info()}")
        
        if provider.validate_connection():
            print("✅ Model validation successful")
            
            # Test generation
            test_prompt = "What is a P/E ratio in finance?"
            response = provider.generate_response(test_prompt, max_tokens=100)
            print(f"💬 Test Response: {response[:200]}...")
            
        else:
            print("❌ Model validation failed")
            
    except Exception as e:
        print(f"💥 Test failed: {str(e)}")


if __name__ == "__main__":
    test_finance_llm()