"""
API Key Configuration Loader
Centralized API key management for FinanceAI Framework
"""

import os
import yaml
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

class APIKeyManager:
    """Manages API keys from environment variables and configuration files"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize API key manager
        
        Args:
            config_path: Optional path to API keys YAML configuration
        """
        # Load environment variables
        load_dotenv()
        
        # Set default config path
        if config_path is None:
            config_path = Path(__file__).parent / "api_keys.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load API keys configuration from YAML file"""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load API keys config: {e}")
            return {}
    
    def get_key(self, service: str, key_name: str = "api_key") -> Optional[str]:
        """Get API key for a service
        
        Args:
            service: Service name (e.g., 'openai', 'alpha_vantage')
            key_name: Key name in config (defaults to 'api_key')
            
        Returns:
            API key string or None if not found
        """
        # Try environment variable first (preferred for production)
        env_var = f"{service.upper()}_API_KEY"
        if key_name != "api_key":
            env_var = f"{service.upper()}_{key_name.upper()}"
        
        env_key = os.getenv(env_var)
        if env_key:
            return env_key
        
        # Try configuration file
        service_config = self.config.get(service, {})
        if isinstance(service_config, dict):
            return service_config.get(key_name)
        
        return None
    
    def get_all_keys(self, service: str) -> Dict[str, str]:
        """Get all keys for a service
        
        Args:
            service: Service name
            
        Returns:
            Dictionary of all keys for the service
        """
        keys = {}
        
        # Get from environment variables
        service_upper = service.upper()
        for env_var in os.environ:
            if env_var.startswith(f"{service_upper}_"):
                key_name = env_var[len(service_upper) + 1:].lower()
                keys[key_name] = os.environ[env_var]
        
        # Get from configuration file
        service_config = self.config.get(service, {})
        if isinstance(service_config, dict):
            for key_name, value in service_config.items():
                if key_name not in keys:  # Environment variables take precedence
                    keys[key_name] = value
        
        return keys
    
    def is_configured(self, service: str) -> bool:
        """Check if a service has API key configured
        
        Args:
            service: Service name
            
        Returns:
            True if API key is available
        """
        return self.get_key(service) is not None
    
    def get_configured_services(self) -> list[str]:
        """Get list of services with configured API keys
        
        Returns:
            List of service names that have API keys
        """
        services = set()
        
        # Check environment variables
        for env_var in os.environ:
            if env_var.endswith('_API_KEY'):
                service = env_var[:-8].lower()  # Remove '_API_KEY'
                services.add(service)
        
        # Check configuration file
        for service in self.config.keys():
            if isinstance(self.config[service], dict) and 'api_key' in self.config[service]:
                services.add(service)
        
        return sorted(list(services))
    
    def validate_key(self, service: str) -> tuple[bool, str]:
        """Validate API key for a service (basic check)
        
        Args:
            service: Service name
            
        Returns:
            Tuple of (is_valid, message)
        """
        key = self.get_key(service)
        if not key:
            return False, f"No API key found for {service}"
        
        if len(key) < 10:
            return False, f"API key for {service} appears too short"
        
        if key.startswith('YOUR_') or key == 'your_api_key_here':
            return False, f"API key for {service} is a placeholder"
        
        return True, f"API key for {service} appears valid"

# Global instance
api_key_manager = APIKeyManager()

# Convenience functions
def get_api_key(service: str, key_name: str = "api_key") -> Optional[str]:
    """Get API key for a service"""
    return api_key_manager.get_key(service, key_name)

def is_service_configured(service: str) -> bool:
    """Check if service is configured"""
    return api_key_manager.is_configured(service)

def get_configured_services() -> list[str]:
    """Get list of configured services"""
    return api_key_manager.get_configured_services()

# Common API key getters
def get_openai_key() -> Optional[str]:
    """Get OpenAI API key"""
    return get_api_key('openai')

def get_anthropic_key() -> Optional[str]:
    """Get Anthropic API key"""
    return get_api_key('anthropic')

def get_alpha_vantage_key() -> Optional[str]:
    """Get Alpha Vantage API key"""
    return get_api_key('alpha_vantage')

def get_polygon_key() -> Optional[str]:
    """Get Polygon.io API key"""
    return get_api_key('polygon')

def get_coingecko_key() -> Optional[str]:
    """Get CoinGecko API key"""
    return get_api_key('coingecko')

def get_etherscan_key() -> Optional[str]:
    """Get Etherscan API key"""
    return get_api_key('etherscan')

def get_binance_keys() -> tuple[Optional[str], Optional[str]]:
    """Get Binance API key and secret"""
    return (
        get_api_key('binance', 'api_key'),
        get_api_key('binance', 'secret_key')
    )

def get_google_ai_key() -> Optional[str]:
    """Get Google AI API key"""
    return get_api_key('google_ai')