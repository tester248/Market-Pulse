"""
Configuration management for FinanceAI Framework
Supports YAML configs with environment variable overrides
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class LLMProviderConfig:
    """Configuration for LLM providers"""
    name: str
    api_key: str
    model: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 1024
    input_cost_per_1k: float = 0.000075
    output_cost_per_1k: float = 0.0003
    enabled: bool = True


@dataclass
class PluginConfig:
    """Configuration for plugins"""
    enabled: bool = True
    auto_discover: bool = True
    discovery_paths: list[str] = field(default_factory=lambda: ["plugins/calculators"])
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "json"
    include_request_id: bool = True
    include_performance_metrics: bool = True
    log_file: Optional[str] = None


@dataclass
class APIConfig:
    """Configuration for API server"""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_enabled: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = 60


@dataclass
class FrameworkConfig:
    """Main framework configuration"""
    version: str = "1.0.0"
    environment: str = "development"
    llm_providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    default_llm_provider: str = "google"
    plugins: PluginConfig = field(default_factory=PluginConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)


class ConfigManager:
    """Manages configuration loading and environment overrides"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.config: Optional[FrameworkConfig] = None
        self._env_prefix = "FINANCEAI_"
    
    def load_config(self, config_file: str = "default.yaml") -> FrameworkConfig:
        """Load configuration from file with environment overrides"""
        
        # Load base configuration
        config_path = self.config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        else:
            print(f"Config file {config_path} not found, using defaults")
            config_dict = {}
        
        # Load local overrides if exists
        local_config_path = self.config_dir / "local.yaml"
        if local_config_path.exists():
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
            config_dict = self._merge_configs(config_dict, local_config)
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Create configuration objects
        self.config = self._create_config_objects(config_dict)
        return self.config
    
    def get_config(self) -> FrameworkConfig:
        """Get loaded configuration"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        
        # Special handling for common environment variables
        env_mappings = {
            "GOOGLE_AI_API_KEY": "llm_providers.google.api_key",
            "GOOGLE_API_KEY": "llm_providers.google.api_key",
            "API_HOST": "api.host",
            "API_PORT": "api.port",
            "LOG_LEVEL": "logging.level",
            "ENVIRONMENT": "environment"
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_dict, config_path, env_value)
        
        # Generic environment variable handling
        for env_var, value in os.environ.items():
            if env_var.startswith(self._env_prefix):
                config_key = env_var[len(self._env_prefix):].lower().replace("_", ".")
                self._set_nested_value(config_dict, config_key, value)
        
        return config_dict
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: str):
        """Set nested configuration value from dot-separated path"""
        keys = path.split('.')
        current = config_dict
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value with type conversion
        final_key = keys[-1]
        current[final_key] = self._convert_env_value(value)
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _create_config_objects(self, config_dict: Dict[str, Any]) -> FrameworkConfig:
        """Create strongly typed configuration objects"""
        
        # Create LLM provider configs
        llm_providers = {}
        providers_config = config_dict.get("llm_providers", {})
        
        # Ensure Google provider exists
        if "google" not in providers_config:
            providers_config["google"] = {}
        
        for name, provider_config in providers_config.items():
            # Get API key from multiple sources
            api_key = (
                provider_config.get("api_key") or
                os.getenv("GOOGLE_AI_API_KEY") or
                os.getenv("GOOGLE_API_KEY")
            )
            
            # Always create provider config, even without API key (for testing)
            llm_providers[name] = LLMProviderConfig(
                name=name,
                api_key=api_key or "MISSING_API_KEY",
                model=provider_config.get("model", "gemini-1.5-flash"),
                temperature=provider_config.get("temperature", 0.1),
                max_tokens=provider_config.get("max_tokens", 1024),
                input_cost_per_1k=provider_config.get("input_cost_per_1k", 0.000075),
                output_cost_per_1k=provider_config.get("output_cost_per_1k", 0.0003),
                enabled=provider_config.get("enabled", True) and bool(api_key)
            )
        
        # Create plugin config
        plugins_config = config_dict.get("plugins", {})
        plugins = PluginConfig(
            enabled=plugins_config.get("enabled", True),
            auto_discover=plugins_config.get("auto_discover", True),
            discovery_paths=plugins_config.get("discovery_paths", ["plugins/calculators"]),
            config=plugins_config.get("config", {})
        )
        
        # Create logging config
        logging_config = config_dict.get("logging", {})
        logging = LoggingConfig(
            level=logging_config.get("level", "INFO"),
            format=logging_config.get("format", "json"),
            include_request_id=logging_config.get("include_request_id", True),
            include_performance_metrics=logging_config.get("include_performance_metrics", True),
            log_file=logging_config.get("log_file")
        )
        
        # Create API config
        api_config = config_dict.get("api", {})
        api = APIConfig(
            host=api_config.get("host", "127.0.0.1"),
            port=api_config.get("port", 8000),
            cors_enabled=api_config.get("cors_enabled", True),
            cors_origins=api_config.get("cors_origins", ["*"]),
            rate_limit_enabled=api_config.get("rate_limit_enabled", False),
            rate_limit_requests_per_minute=api_config.get("rate_limit_requests_per_minute", 60)
        )
        
        # Create main framework config
        return FrameworkConfig(
            version=config_dict.get("version", "1.0.0"),
            environment=config_dict.get("environment", "development"),
            llm_providers=llm_providers,
            default_llm_provider=config_dict.get("default_llm_provider", "google"),
            plugins=plugins,
            logging=logging,
            api=api
        )


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> FrameworkConfig:
    """Get the global configuration"""
    return config_manager.get_config()


def reload_config() -> FrameworkConfig:
    """Reload configuration from files"""
    return config_manager.load_config()