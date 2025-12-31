// Financial Seismograph API Integration
// Base URL for the Financial Seismograph backend
const API_BASE_URL = 'http://localhost:8000'; //(public url) or localhost:8000

// New API response types based on the OpenAPI schema
export interface ModelHealth {
  is_healthy: boolean;
  last_used?: string;
  error_count?: number;
}

export interface HealthStatus {
  status: string;
  models: Record<string, ModelHealth>;
  uptime_seconds: number;
  processed_articles: number;
  processed_queries: number;
}

export interface UserQuery {
  query: string;
  use_rag?: boolean;
}

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
}

export interface HTTPValidationError {
  detail?: ValidationError[];
}

// Legacy API response types (keeping for compatibility during transition)
export interface ApiSeismographData {
  timestamp: string;
  sentiment_score: number;
  volume: number;
  peak_intensity: number;
  tickers: string[];
  confidence?: number | null;
  quality_score?: number | null;
  market_impact?: string | null;
  processing_time_ms?: number | null;
}

export interface ApiTremor {
  id: string;
  timestamp: string;
  sentiment_score: number;
  confidence: number;
  impact_level: string;
  title: string;
  summary: string;
  tickers: string[];
  source: string;
  quality_grade: string;
}

export interface FinancialMetric {
  metric: string;
  value: string | number;
  confidence?: string;
}

export interface SentimentAnalysis {
  primary_sentiment: string;
  confidence: number;
  sentiment_score?: number;
}

export interface TimelineDataPoint {
  timestamp: string;
  sentiment_score: number;
  volume: number;
  peak_intensity: number;
  tickers: string[];
  confidence?: number;
  quality_score?: number;
  market_impact?: string;
  processing_time_ms?: number;
  // Optional fields that might be present for transformation
  id?: string;
  sentiment?: number;
  impact_level?: string;
  title?: string;
  headline?: string;
  summary?: string;
  description?: string;
  source?: string;
  quality_grade?: string;
}

export interface QueryResult {
  success: boolean;
  original_query: string;
  response: string;
  articles_processed: number;
  processing_times: {
    total_ms: number;
  };
  error?: string;
}

export interface ApiEpicenterAnalysis {
  tremor_id: string;
  title: string;
  executive_summary: string;
  sentiment_analysis: SentimentAnalysis;
  financial_metrics: FinancialMetric[];
  market_impact: string;
  key_insights: string[];
  investment_implications: string[];
  risk_factors: string[];
  pipeline_trace: Array<Record<string, string>>;
  quality_score: number;
  processing_time_ms: number;
}

export interface ProcessArticleRequest {
  title: string;
  content: string;
  source: string;
  url?: string;
  published?: string;
}

export interface ProcessArticleResult {
  success: boolean;
  article_id: string;
  error?: string;
}

export interface HeadlinesResponse {
  headlines: ApiHeadline[];
  count: number;
}

export interface SystemStatsResponse {
  integration_status: string;
  articles_processed: number;
  queries_processed: number;
  uptime_seconds: number;
  models: Record<string, ModelHealth>;
  success_rate: number;
  avg_processing_time_ms: number;
  alerts_generated: number;
  quality_distribution: {
    excellent: number;
    good: number;
    fair: number;
    poor: number;
    insufficient: number;
  };
  system_health: {
    database_connected: boolean;
    llm_manager_active: boolean;
    orchestrator_ready: boolean;
  };
  seismograph: {
    data_points: number;
    last_update: string;
    active_feeds: number;
  };
}

export interface ApiQueryRequest {
  query: string;
  context?: string | null;
  tickers?: string[] | null;
}

export interface ApiQueryResponse {
  response: string;
  confidence: number;
  sources: string[];
  related_articles: string[];
  processing_time_ms: number;
}

export interface ApiSystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  timestamp: string;
  components: {
    database: string;
    ai_integration: string;
    ollama: string;
  };
  metrics: {
    active_connections: number;
    processing_queue: number;
    total_processed: number;
  };
}

export interface ApiSystemStats {
  integration_status: string;
  articles_processed: number;
  success_rate: number;
  avg_processing_time_ms: number;
  alerts_generated: number;
  quality_distribution: {
    excellent: number;
    good: number;
    fair: number;
    poor: number;
    insufficient: number;
  };
  content_type_distribution: Record<string, number>;
  sentiment_distribution: Record<string, number>;
  last_processed_at: string | null;
  system_health: {
    database_connected: boolean;
    llm_manager_active: boolean;
    orchestrator_ready: boolean;
  };
  seismograph: {
    data_points: number;
    last_update: string;
    active_feeds: number;
  };
}

export interface ApiArticle {
  id: number;
  title: string;
  content: string;
  url: string;
  published_at: string;
  word_count: number;
  rss_feed_name: string;
  analysis_sentiment: number;
  analysis_confidence: number;
}

export interface ApiHeadline {
  title: string;
  link: string;
  published: string;
  source: string;
  summary?: string;
}

// API client class
class FinancialSeismographAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetchWithErrorHandling<T>(url: string, options?: RequestInit): Promise<T> {
    try {
      const defaultHeaders: Record<string, string> = {};
      
      // Only add Content-Type for non-GET requests
      if (options?.method && options.method !== 'GET') {
        defaultHeaders['Content-Type'] = 'application/json';
      }

      const response = await fetch(`${this.baseUrl}${url}`, {
        method: 'GET', // Default to GET
        headers: {
          ...defaultHeaders,
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${url}:`, error);
      throw error;
    }
  }

  // Seismograph timeline data endpoint (new API)
  async getSeismographData(hours: number = 48, tickers?: string, minConfidence: number = 0.5): Promise<TimelineDataPoint[]> {
    try {
      const params = new URLSearchParams();
      params.append('hours', Math.min(hours, 168).toString()); // API has max 168 hours (7 days)
      params.append('min_confidence', minConfidence.toString());
      if (tickers) params.append('tickers', tickers);
      
      const result = await this.fetchWithErrorHandling<TimelineDataPoint[]>(`/api/v1/pulse/timeline?${params.toString()}`);
      // Ensure we always return an array
      return Array.isArray(result) ? result : [];
    } catch (_error) {
      console.error('Failed to fetch timeline data:', _error);
      return []; // Return empty array on error
    }
  }

  // Legacy endpoints - these may not be available in the new API
  // Keeping for backward compatibility, but consider removing if not needed
  
  // Tremors detection endpoint (legacy - may not exist in new API)
  async getTremors(hours: number = 24, minIntensity: number = 0.6, tickers?: string): Promise<ApiTremor[]> {
    // This endpoint may not exist in the new API - using timeline data instead
    try {
      const timelineData = await this.getSeismographData(hours, tickers, minIntensity);
      // Transform timeline data to tremor format for compatibility
      return this.transformTimelineToTremors(timelineData);
    } catch {
      console.warn('Legacy tremors endpoint not available, using timeline data');
      return [];
    }
  }

  // Transform timeline data to tremor format
  private transformTimelineToTremors(timelineData: TimelineDataPoint[]): ApiTremor[] {
    if (!Array.isArray(timelineData)) {
      return [];
    }
    
    return timelineData.map((item, index) => ({
      id: item.id || `tremor_${index}_${Date.now()}`,
      timestamp: item.timestamp || new Date().toISOString(),
      sentiment_score: item.sentiment_score || item.sentiment || 0,
      confidence: item.confidence || 0.5,
      impact_level: item.impact_level || 'medium',
      title: item.title || item.headline || 'Market Event',
      summary: item.summary || item.description || 'Market movement detected',
      tickers: Array.isArray(item.tickers) ? item.tickers : [],
      source: item.source || 'Market Pulse API',
      quality_grade: item.quality_grade || 'B',
    }));
  }

  // Event details endpoint (new API)
  async getEpicenterAnalysis(eventId: string): Promise<ApiEpicenterAnalysis> {
    return this.fetchWithErrorHandling<ApiEpicenterAnalysis>(`/api/v1/insights/event/${eventId}`);
  }

  // AI Query endpoint (new API)
  async postQuery(query: string, useRag: boolean = true): Promise<QueryResult> {
    const requestBody: UserQuery = {
      query,
      use_rag: useRag,
    };
    
    return this.fetchWithErrorHandling<QueryResult>('/api/v1/insights/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
  }

  // Process article endpoint (new API)
  async processArticle(articleData: ProcessArticleRequest): Promise<ProcessArticleResult> {
    return this.fetchWithErrorHandling<ProcessArticleResult>('/api/v1/insights/process-article', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(articleData),
    });
  }

  // Articles endpoint (legacy - not available in new API)
  async getArticles(): Promise<ApiArticle[]> {
    console.warn('Articles endpoint not available in new API');
    return []; // Return empty array for compatibility
  }

  // System health endpoint
  async getSystemHealth(): Promise<HealthStatus> {
    return this.fetchWithErrorHandling<HealthStatus>('/api/v1/health');
  }

  // System statistics endpoint (legacy - using health data instead)
  async getSystemStats(): Promise<SystemStatsResponse> {
    // The new API doesn't have a separate stats endpoint, so we'll use health data
    try {
      const health = await this.getSystemHealth();
      return {
        integration_status: health.status,
        articles_processed: health.processed_articles,
        queries_processed: health.processed_queries,
        uptime_seconds: health.uptime_seconds,
        models: health.models,
        // Add mock data for fields that might be expected by the UI
        success_rate: 0.95,
        avg_processing_time_ms: 250,
        alerts_generated: 0,
        quality_distribution: {
          excellent: 0.4,
          good: 0.3,
          fair: 0.2,
          poor: 0.1,
          insufficient: 0.0,
        },
        system_health: {
          database_connected: health.status === 'healthy',
          llm_manager_active: health.status === 'healthy',
          orchestrator_ready: health.status === 'healthy',
        },
        seismograph: {
          data_points: health.processed_articles || 0,
          last_update: new Date().toISOString(),
          active_feeds: health.status === 'healthy' ? 3 : 0,
        },
      };
    } catch (error) {
      console.error('Failed to get system stats from health endpoint:', error);
      throw error;
    }
  }

  // Live headlines aggregated from configured RSS feeds
  async getLiveHeadlines(limitPerFeed: number = 5): Promise<HeadlinesResponse> {
    try {
      const result = await this.fetchWithErrorHandling<HeadlinesResponse>(`/api/v1/news/headlines?limit_per_feed=${limitPerFeed}`);
      return {
        headlines: Array.isArray(result.headlines) ? result.headlines : [],
        count: result.count || 0
      };
    } catch (error) {
      console.error('Failed to fetch live headlines:', error);
      return {headlines: [], count: 0};
    }
  }
}

// Export singleton instance
export const apiClient = new FinancialSeismographAPI();

// Data transformation utilities to convert API responses to dashboard types
export const transformSeismographData = (apiData: TimelineDataPoint[]) => {
  if (!Array.isArray(apiData)) {
    console.warn('API data is not an array, returning empty array');
    return [];
  }
  
  return apiData.map((item, index) => ({
    timestamp: item.timestamp || new Date().toISOString(),
    sentiment: item.sentiment_score || item.sentiment || 0,
    volume: item.volume || Math.random() * 100, // Fallback for volume
    event_id: item.id || `seismo_${index}_${Date.now()}`,
    ticker: (item.tickers && item.tickers[0]) || undefined,
    headline: item.title || `Market Event - ${item.tickers?.join(', ') || 'General'}`,
  }));
};

export const transformTremorToEventInsight = (apiTremor: ApiTremor, epicenterData?: ApiEpicenterAnalysis) => {
  const sentiment = (apiTremor.sentiment_score || 0) > 0.1 ? 'Bullish' : 
                   (apiTremor.sentiment_score || 0) < -0.1 ? 'Bearish' : 'Neutral';
  
  return {
    event_id: apiTremor.id || `event_${Date.now()}`,
    headline: apiTremor.title || 'Market Event',
    summary: [apiTremor.summary || 'No summary available'],
    causal_analysis: {
      sentiment: sentiment as 'Bullish' | 'Bearish' | 'Neutral',
      confidence: apiTremor.confidence || 0.5,
      driver: epicenterData?.executive_summary || apiTremor.summary || 'Market movement detected',
    },
    source_consensus: [
      { source: apiTremor.source || 'Market Pulse API', stance: 'agrees' as const },
    ],
    pipeline_trace: epicenterData?.pipeline_trace || 'Processing complete',
    timestamp: apiTremor.timestamp || new Date().toISOString(),
    ticker: (apiTremor.tickers && apiTremor.tickers[0]) || undefined,
  };
};

// Health check utility
export const checkAPIHealth = async (): Promise<boolean> => {
  try {
    const health = await apiClient.getSystemHealth();
    // Check if status is "healthy" and all models are healthy
    const allModelsHealthy = health.models && Object.values(health.models).every((model: ModelHealth) => model.is_healthy === true);
    return health.status === 'healthy' && allModelsHealthy;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};