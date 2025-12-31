"""
Market Pulse API Endpoint Testing Script

This script tests all the endpoints of the Market Pulse API.
It provides detailed feedback on the status of each endpoint.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse
import sys
import os
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

# Default API base URL
DEFAULT_BASE_URL = "http://localhost:8000"

# Test article data
TEST_ARTICLE = {
    "title": "Test Article for API Validation",
    "content": "This is a test article to validate the Market Pulse API's article processing capabilities. " +
               "It contains financial keywords like stocks, bonds, inflation, GDP growth, and mentions " +
               "companies like AAPL, MSFT, and GOOGL to test ticker extraction.",
    "source": "API Test Suite",
    "url": "https://example.com/test-article",
    "published": datetime.now().isoformat()
}

# Test query data
TEST_QUERY = {
    "query": "What are the implications of rising inflation on tech stocks like AAPL and MSFT?",
    "use_rag": True
}

TEST_QUERY_NO_RAG = {
    "query": "How might interest rate changes affect the banking sector?",
    "use_rag": False
}

class APITester:
    """Tests all endpoints of the Market Pulse API"""
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self.session = None
        self.results = []
        self.event_ids = []  # To store event IDs for testing event details endpoint
        
    async def setup(self):
        """Set up the testing session"""
        self.session = aiohttp.ClientSession()
        print(f"{Fore.CYAN}Setting up API test session for {self.base_url}{Style.RESET_ALL}")
    
    async def teardown(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        print(f"\n{Fore.CYAN}Test session completed{Style.RESET_ALL}")
    
    def log_result(self, endpoint: str, status: str, response_time: float, details: str = None):
        """Log a test result"""
        result = {
            "endpoint": endpoint,
            "status": status,
            "response_time_ms": response_time,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.results.append(result)
        
        # Print colored result
        color = Fore.GREEN if status == "PASS" else Fore.RED
        print(f"{color}{status}{Style.RESET_ALL} - {endpoint} ({response_time:.2f}ms)")
        if details:
            print(f"       {Fore.YELLOW}{details}{Style.RESET_ALL}")
    
    async def test_root_endpoint(self):
        """Test the root endpoint that provides API information"""
        endpoint = "/"
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}{endpoint}") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if all(k in data for k in ["name", "version", "description"]):
                        self.log_result(endpoint, "PASS", response_time)
                    else:
                        self.log_result(endpoint, "FAIL", response_time, 
                                      "Response missing required fields")
                else:
                    self.log_result(endpoint, "FAIL", response_time, 
                                  f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.log_result(endpoint, "FAIL", response_time, str(e))
    
    async def test_health_endpoint(self):
        """Test the health check endpoint"""
        endpoint = "/api/v1/health"
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}{endpoint}") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "status" in data:
                        self.log_result(endpoint, "PASS", response_time, 
                                      f"API Status: {data['status']}")
                    else:
                        self.log_result(endpoint, "FAIL", response_time, 
                                      "Response missing status field")
                else:
                    self.log_result(endpoint, "FAIL", response_time, 
                                  f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.log_result(endpoint, "FAIL", response_time, str(e))
    
    async def test_seismograph_timeline(self):
        """Test the seismograph timeline endpoint with various parameters"""
        base_endpoint = "/api/v1/pulse/timeline"
        
        # Test cases with different parameters
        test_cases = [
            {"params": "", "name": "default"},
            {"params": "?hours=12", "name": "12 hours"},
            {"params": "?tickers=AAPL,MSFT", "name": "filtered by tickers"},
            {"params": "?min_confidence=0.7", "name": "high confidence"}
        ]
        
        for test_case in test_cases:
            endpoint = f"{base_endpoint}{test_case['params']}"
            start_time = time.time()
            
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        if "data_points" in data:
                            num_points = len(data["data_points"])
                            self.log_result(f"{base_endpoint} ({test_case['name']})", "PASS", 
                                          response_time, f"Returned {num_points} data points")
                            
                            # Store event IDs for later testing
                            if num_points > 0 and not self.event_ids:
                                for point in data["data_points"][:3]:  # Store up to 3 IDs
                                    if "id" in point:
                                        self.event_ids.append(point["id"])
                        else:
                            self.log_result(f"{base_endpoint} ({test_case['name']})", "FAIL", 
                                          response_time, "Response missing data_points field")
                    else:
                        self.log_result(f"{base_endpoint} ({test_case['name']})", "FAIL", 
                                      response_time, f"HTTP {response.status}: {await response.text()}")
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.log_result(f"{base_endpoint} ({test_case['name']})", "FAIL", response_time, str(e))
    
    async def test_event_details(self):
        """Test retrieving details for specific events"""
        base_endpoint = "/api/v1/insights/event"
        
        if not self.event_ids:
            self.log_result(base_endpoint, "SKIP", 0, "No event IDs available to test")
            return
        
        for event_id in self.event_ids:
            endpoint = f"{base_endpoint}/{event_id}"
            start_time = time.time()
            
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        if "id" in data and data["id"] == event_id:
                            self.log_result(endpoint, "PASS", response_time)
                        else:
                            self.log_result(endpoint, "FAIL", response_time, 
                                          "Response ID does not match requested ID")
                    else:
                        self.log_result(endpoint, "FAIL", response_time, 
                                      f"HTTP {response.status}: {await response.text()}")
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                self.log_result(endpoint, "FAIL", response_time, str(e))
    
    async def test_user_query(self):
        """Test the query processing endpoint with and without RAG"""
        endpoint = "/api/v1/insights/query"
        
        # Test with RAG
        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}",
                json=TEST_QUERY
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "success" in data and data["success"]:
                        self.log_result(f"{endpoint} (with RAG)", "PASS", response_time,
                                      f"Response length: {len(str(data.get('response', '')))} chars")
                    else:
                        self.log_result(f"{endpoint} (with RAG)", "FAIL", response_time,
                                      data.get("error", "Unknown error"))
                else:
                    self.log_result(f"{endpoint} (with RAG)", "FAIL", response_time,
                                  f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.log_result(f"{endpoint} (with RAG)", "FAIL", response_time, str(e))
        
        # Small delay between requests
        await asyncio.sleep(1)
        
        # Test without RAG
        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}",
                json=TEST_QUERY_NO_RAG
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "success" in data and data["success"]:
                        self.log_result(f"{endpoint} (without RAG)", "PASS", response_time,
                                      f"Response length: {len(str(data.get('response', '')))} chars")
                    else:
                        self.log_result(f"{endpoint} (without RAG)", "FAIL", response_time,
                                      data.get("error", "Unknown error"))
                else:
                    self.log_result(f"{endpoint} (without RAG)", "FAIL", response_time,
                                  f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.log_result(f"{endpoint} (without RAG)", "FAIL", response_time, str(e))
    
    async def test_article_processing(self):
        """Test submitting articles for processing"""
        endpoint = "/api/v1/insights/process-article"
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}",
                json=TEST_ARTICLE
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "article_id" in data:
                        self.log_result(endpoint, "PASS", response_time, 
                                      f"Article ID: {data['article_id']}")
                    else:
                        self.log_result(endpoint, "FAIL", response_time, 
                                      "Response missing article_id field")
                else:
                    self.log_result(endpoint, "FAIL", response_time, 
                                  f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.log_result(endpoint, "FAIL", response_time, str(e))
    
    async def run_all_tests(self):
        """Run all API tests"""
        print(f"\n{Fore.CYAN}====== Market Pulse API Test Suite ======{Style.RESET_ALL}")
        print(f"Testing against: {self.base_url}\n")
        
        await self.setup()
        
        # Run all tests
        print(f"\n{Fore.CYAN}1. Testing Basic API Information{Style.RESET_ALL}")
        await self.test_root_endpoint()
        
        print(f"\n{Fore.CYAN}2. Testing Health Check{Style.RESET_ALL}")
        await self.test_health_endpoint()
        
        print(f"\n{Fore.CYAN}3. Testing Seismograph Timeline{Style.RESET_ALL}")
        await self.test_seismograph_timeline()
        
        print(f"\n{Fore.CYAN}4. Testing Event Details{Style.RESET_ALL}")
        await self.test_event_details()
        
        print(f"\n{Fore.CYAN}5. Testing Article Processing{Style.RESET_ALL}")
        await self.test_article_processing()
        
        print(f"\n{Fore.CYAN}6. Testing User Query Processing{Style.RESET_ALL}")
        await self.test_user_query()
        
        # Generate summary
        self.print_summary()
        
        await self.teardown()
    
    def print_summary(self):
        """Print a summary of test results"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["status"] == "PASS")
        failed_tests = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped_tests = sum(1 for r in self.results if r["status"] == "SKIP")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n{Fore.CYAN}====== Test Summary ======{Style.RESET_ALL}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {Fore.GREEN}{passed_tests}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{failed_tests}{Style.RESET_ALL}")
        print(f"Skipped: {Fore.YELLOW}{skipped_tests}{Style.RESET_ALL}")
        print(f"Success Rate: {Fore.CYAN}{success_rate:.1f}%{Style.RESET_ALL}")
        
        if failed_tests > 0:
            print(f"\n{Fore.RED}Failed Tests:{Style.RESET_ALL}")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"- {result['endpoint']}: {result.get('details', 'Unknown error')}")

async def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Test Market Pulse API endpoints")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL of the API")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Check if API is accessible
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/") as response:
                if response.status != 200:
                    print(f"{Fore.RED}Error: Cannot connect to API at {args.url}{Style.RESET_ALL}")
                    sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Error connecting to API: {str(e)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Make sure the Market Pulse API server is running at {args.url}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Run tests
    tester = APITester(args.url)
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())