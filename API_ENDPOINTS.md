# Market Pulse API Endpoints

This document provides a quick reference for all available endpoints in the Market Pulse financial insights API.

## API Endpoints

### Basic Information

- **GET /**
  - Returns basic API information including name, version, and description
  - No parameters required
  - Example response: `{"name": "Market Pulse API", "version": "1.0.0", "description": "Financial insights engine powered by OpenRouter"}`

### Health Check

- **GET /api/v1/health**
  - Returns health status of the API and all LLM models
  - No parameters required
  - Example response includes:
    - `status`: "healthy" or "degraded"
    - `models`: Status of individual models
    - `uptime_seconds`: How long the API has been running
    - `processed_articles`: Count of articles processed
    - `processed_queries`: Count of user queries processed

### Seismograph Data

- **GET /api/v1/pulse/timeline**
  - Returns time-series sentiment and event data for the Seismograph chart
  - Query parameters:
    - `hours`: Number of hours of data to return (default: 24, range: 1-168)
    - `tickers`: Comma-separated list of ticker symbols to filter by (optional)
    - `min_confidence`: Minimum confidence threshold (default: 0.5, range: 0.0-1.0)
  - Example: `/api/v1/pulse/timeline?hours=12&tickers=AAPL,MSFT&min_confidence=0.7`
  - Response includes data points with sentiment scores, timestamps, and market impact information

### Event Details

- **GET /api/v1/insights/event/{event_id}**
  - Returns detailed insights for a specific event
  - Path parameters:
    - `event_id`: The unique identifier for the event
  - Example: `/api/v1/insights/event/event-1234567890`
  - Response includes comprehensive analysis of the event with sentiment data, structured information, and source verification

### User Queries

- **POST /api/v1/insights/query**
  - Process a user's financial question through the LLM pipeline
  - Request body:
    - `query`: The user's financial question (string)
    - `use_rag`: Whether to use Retrieval Augmented Generation with current news (boolean, default: true)
  - Example request:
    ```json
    {
      "query": "How do rising interest rates affect tech stocks?",
      "use_rag": true
    }
    ```
  - Response includes structured financial insights with contextual information from current news sources when RAG is enabled

### Article Processing

- **POST /api/v1/insights/process-article**
  - Submit a financial article for processing through the assembly line
  - Request body:
    - `title`: Article title
    - `content`: Article content
    - `source`: Source name
    - `url`: Article URL (optional)
    - `published`: Publication timestamp (optional)
  - Example request:
    ```json
    {
      "title": "Fed Announces Interest Rate Decision",
      "content": "The Federal Reserve today announced it will raise interest rates by 25 basis points...",
      "source": "Financial Times",
      "url": "https://ft.com/article/123"
    }
    ```
  - Response includes an `article_id` for tracking the processing status
  - Article processing happens in the background, and results can be accessed through the seismograph timeline endpoint

## Authentication

All API endpoints are currently publicly accessible, but throttling may be applied based on usage patterns. In production deployments, proper authentication using API keys is recommended.

## Response Format

All responses are in JSON format and generally follow this structure:

- Success responses include data relevant to the endpoint
- Error responses include an `error` field with details about what went wrong

## Usage Notes

- For optimal performance, limit the number of concurrent requests
- The API uses OpenRouter for LLM capabilities, ensuring low latency and high accuracy
- The seismograph timeline endpoint can be polled periodically (e.g., every 30 seconds) to get real-time updates