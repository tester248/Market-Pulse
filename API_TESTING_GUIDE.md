# Market Pulse API: Startup and Testing Guide

This document provides instructions for starting the Market Pulse API server and testing each endpoint.

## Starting the API Server

To start the Market Pulse API server, follow these steps:

1. Ensure all dependencies are installed:
   ```pwsh
   pip install -r requirements_fast_smart.txt
   ```

2. Verify that the OpenRouter API key is configured in `config/api_keys.yaml`

3. Start the API server:
   ```pwsh
   python production_startup.py
   ```

   This will initialize the database and start the FastAPI server, which typically runs on `http://localhost:8000` by default.

## Testing Endpoints

You can test the endpoints using the built-in test script or manually using tools like curl, Postman, or your web browser.

### Using the Test Script

The `test_market_pulse_api.py` script tests all available endpoints:

```pwsh
python test_market_pulse_api.py
```

This will output the status of each endpoint with color-coded results (green for success, red for failure).

### Manual Testing

#### Basic Information Endpoint

```pwsh
# Using curl
curl http://localhost:8000/

# Using Invoke-RestMethod in PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get
```

#### Health Check Endpoint

```pwsh
# Using curl
curl http://localhost:8000/api/v1/health

# Using Invoke-RestMethod in PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/health" -Method Get
```

#### Seismograph Timeline Endpoint

```pwsh
# Using curl with query parameters
curl "http://localhost:8000/api/v1/pulse/timeline?hours=12&tickers=AAPL,MSFT&min_confidence=0.7"

# Using Invoke-RestMethod in PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/pulse/timeline?hours=12&tickers=AAPL,MSFT&min_confidence=0.7" -Method Get
```

#### Event Details Endpoint

```pwsh
# Using curl
curl http://localhost:8000/api/v1/insights/event/event-1234567890

# Using Invoke-RestMethod in PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/insights/event/event-1234567890" -Method Get
```

#### User Query Endpoint

```pwsh
# Using curl
curl -X POST -H "Content-Type: application/json" -d "{\"query\":\"How do rising interest rates affect tech stocks?\",\"use_rag\":true}" http://localhost:8000/api/v1/insights/query

# Using Invoke-RestMethod in PowerShell
$body = @{
    query = "How do rising interest rates affect tech stocks?"
    use_rag = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/insights/query" -Method Post -Body $body -ContentType "application/json"
```

#### Article Processing Endpoint

```pwsh
# Using curl
curl -X POST -H "Content-Type: application/json" -d "{\"title\":\"Fed Announces Interest Rate Decision\",\"content\":\"The Federal Reserve today announced it will raise interest rates by 25 basis points...\",\"source\":\"Financial Times\",\"url\":\"https://ft.com/article/123\"}" http://localhost:8000/api/v1/insights/process-article

# Using Invoke-RestMethod in PowerShell
$body = @{
    title = "Fed Announces Interest Rate Decision"
    content = "The Federal Reserve today announced it will raise interest rates by 25 basis points..."
    source = "Financial Times"
    url = "https://ft.com/article/123"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/insights/process-article" -Method Post -Body $body -ContentType "application/json"
```

## Using Swagger UI

The FastAPI server includes a built-in Swagger UI that provides interactive documentation:

1. Start the API server as described above
2. Open a web browser and navigate to `http://localhost:8000/docs`
3. The Swagger UI shows all available endpoints with request parameters and allows you to test them directly from the browser

## Troubleshooting

If you encounter issues:

1. Check the server logs for detailed error messages
2. Verify that all dependencies are correctly installed
3. Ensure the OpenRouter API key is valid and correctly configured
4. Confirm that the database has been properly initialized

For API-specific errors, check the response status code and error message. The API returns detailed error information in the response body when something goes wrong.

## Production Deployment

For production deployment, it's recommended to:

1. Use a production ASGI server like uvicorn or gunicorn behind a reverse proxy
2. Set up proper authentication
3. Configure rate limiting
4. Implement comprehensive monitoring

Refer to `README_PRODUCTION.md` for detailed production deployment instructions.