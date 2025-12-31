"""
Run Market Pulse API Tests

This script starts the Market Pulse API server and runs the API tests.
"""

import subprocess
import time
import sys
import os
import signal
import argparse
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Run Market Pulse API tests")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    parser.add_argument("--no-server", action="store_true", help="Don't start a server, just run tests")
    parser.add_argument("--api-key", help="OpenRouter API key")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENROUTER_API_KEY"] = args.api_key
    
    api_server_process = None
    
    try:
        # Start API server if needed
        if not args.no_server:
            print(f"{Fore.CYAN}Starting Market Pulse API server on port {args.port}...{Style.RESET_ALL}")
            
            # Start the server
            api_server_process = subprocess.Popen(
                [sys.executable, "run_market_pulse.py", "--port", str(args.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            print(f"{Fore.YELLOW}Waiting for server to start...{Style.RESET_ALL}")
            time.sleep(5)  # Give the server some time to start
            
            # Check if server started successfully
            if api_server_process.poll() is not None:
                # Process has exited
                stdout, stderr = api_server_process.communicate()
                print(f"{Fore.RED}Error starting API server:{Style.RESET_ALL}")
                print(stderr)
                return 1
            
            print(f"{Fore.GREEN}API server started successfully{Style.RESET_ALL}")
        
        # Run API tests
        print(f"{Fore.CYAN}Running API tests...{Style.RESET_ALL}")
        
        test_url = f"http://localhost:{args.port}"
        test_cmd = [sys.executable, "test_market_pulse_api.py", "--url", test_url]
        
        # Run tests and capture return code
        test_process = subprocess.run(test_cmd)
        
        return test_process.returncode
        
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}\nInterrupted by user{Style.RESET_ALL}")
        return 1
    
    finally:
        # Clean up server process if we started it
        if api_server_process is not None:
            print(f"{Fore.YELLOW}Stopping API server...{Style.RESET_ALL}")
            
            # Try to terminate gracefully first
            api_server_process.terminate()
            try:
                api_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If termination times out, kill the process
                api_server_process.kill()
            
            print(f"{Fore.YELLOW}API server stopped{Style.RESET_ALL}")

if __name__ == "__main__":
    sys.exit(main())