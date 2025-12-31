# Market Pulse API Testing Guide

This guide explains how to test the API endpoints of the Market Pulse financial insights engine.

## Prerequisites

Before running the tests, ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Testing Options

### Option 1: Automated Testing with Server Start

The simplest way to test the API is to use the provided script that automatically starts the server and runs the tests:

```bash
python run_api_tests.py
```

This script will:
1. Start the Market Pulse API server on port 8000
2. Wait for the server to initialize
3. Run the complete test suite against all endpoints
4. Display detailed results with color-coded output
5. Shutdown the server when complete

#### Additional Options:

- Specify a different port:
  ```bash
  python run_api_tests.py --port 8080
  ```

- Provide an OpenRouter API key:
  ```bash
  python run_api_tests.py --api-key your_api_key_here
  ```

### Option 2: Test Against a Running Server

If you already have the Market Pulse API running, you can test against it without starting a new server:

```bash
python run_api_tests.py --no-server
```

Or test directly with the test script:

```bash
python test_market_pulse_api.py --url http://localhost:8000
```

## Test Coverage

The test suite validates the following endpoints:

1. **Root Endpoint** (`/`):
   - Verifies that basic API information is returned

2. **Health Check** (`/api/v1/health`):
   - Confirms that the API health status is accessible
   - Checks for model health information

3. **Seismograph Timeline** (`/api/v1/pulse/timeline`):
   - Tests with default parameters
   - Tests filtering by hours
   - Tests filtering by ticker symbols
   - Tests filtering by confidence threshold

4. **Event Details** (`/api/v1/insights/event/{event_id}`):
   - Tests retrieving detailed information for specific events

5. **User Query Processing** (`/api/v1/insights/query`):
   - Tests query processing with RAG enabled
   - Tests query processing with RAG disabled

6. **Article Processing** (`/api/v1/insights/process-article`):
   - Tests submitting articles for analysis

## Test Report

At the end of the test run, a summary report is displayed showing:
- Total number of tests run
- Number of tests passed
- Number of tests failed
- Success rate
- Details of any failed tests

## Troubleshooting

If you encounter issues with the tests:

1. **API Server Not Starting**:
   - Check if the port is already in use
   - Verify that all dependencies are installed
   - Check if the OpenRouter API key is properly set

2. **Connection Errors**:
   - Verify that the API server is running at the specified URL
   - Check network connectivity and firewall settings

3. **Authentication Errors**:
   - Ensure that a valid OpenRouter API key is provided
   - Check that environment variables are set correctly

## Manual Testing

You can also test the API manually using tools like curl or Postman:

### Example curl commands:

```bash
# Check API information
curl http://localhost:8000/

# Check health status
curl http://localhost:8000/api/v1/health

# Get seismograph timeline data
curl http://localhost:8000/api/v1/pulse/timeline

# Submit a query
curl -X POST -H "Content-Type: application/json" -d '{"query":"How do interest rates affect tech stocks?","use_rag":true}' http://localhost:8000/api/v1/insights/query

# Submit an article for processing
curl -X POST -H "Content-Type: application/json" -d '{"title":"Test Article","content":"This is a test article about financial markets.","source":"Test"}' http://localhost:8000/api/v1/insights/process-article
```