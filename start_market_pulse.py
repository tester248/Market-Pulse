"""
Market Pulse Startup Script

This script starts the Market Pulse API server with proper configuration.
"""

import os
import sys
import logging
import subprocess
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import aiohttp
        import pydantic
        import asyncio
        import beautifulsoup4
        logger.info("All required packages are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.info("Installing required packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            logger.info("Successfully installed required packages.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {str(e)}")
            return False

def setup_environment(api_key=None):
    """Set up environment variables"""
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    
    # Check if API key is set
    if "OPENROUTER_API_KEY" not in os.environ:
        logger.warning("OPENROUTER_API_KEY environment variable not set.")
        logger.warning("Using default API key (limited usage).")
    else:
        logger.info("Using OpenRouter API key from environment.")

def start_api_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server"""
    import uvicorn
    
    logger.info(f"Starting Market Pulse API server at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "market_pulse_api:app",
        host=host,
        port=port,
        reload=reload
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start the Market Pulse API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    # Print banner
    print("""
    ███╗   ███╗ █████╗ ██████╗ ██╗  ██╗███████╗████████╗    ██████╗ ██╗   ██╗██╗     ███████╗███████╗
    ████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝██╔════╝╚══██╔══╝    ██╔══██╗██║   ██║██║     ██╔════╝██╔════╝
    ██╔████╔██║███████║██████╔╝█████╔╝ █████╗     ██║       ██████╔╝██║   ██║██║     ███████╗█████╗  
    ██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗ ██╔══╝     ██║       ██╔═══╝ ██║   ██║██║     ╚════██║██╔══╝  
    ██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗███████╗   ██║       ██║     ╚██████╔╝███████╗███████║███████╗
    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝       ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝
                                                                                                     
    Financial Insights Engine powered by OpenRouter
    """)
    
    # Check requirements
    if not check_requirements():
        logger.error("Failed to install required packages. Please install them manually.")
        sys.exit(1)
    
    # Set up environment
    setup_environment(args.api_key)
    
    # Start API server
    start_api_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )

if __name__ == "__main__":
    main()