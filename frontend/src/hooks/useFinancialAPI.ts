import { useState, useEffect, useCallback } from 'react';
import { 
  apiClient, 
  transformSeismographData, 
  transformTremorToEventInsight,
  checkAPIHealth,
  ApiTremor
} from '../services/api';
import { SentimentData, EventInsight } from '../types';

export interface ApiState {
  isHealthy: boolean;
  isLoading: boolean;
  error: string | null;
  lastUpdate: Date | null;
}

export const useFinancialAPI = () => {
  const [seismographData, setSeismographData] = useState<SentimentData[]>([]);
  const [tremors, setTremors] = useState<ApiTremor[]>([]);
  const [apiState, setApiState] = useState<ApiState>({
    isHealthy: false,
    isLoading: false,
    error: null,
    lastUpdate: null,
  });

  // Health check function
  const performHealthCheck = useCallback(async () => {
    try {
      console.log('Performing health check...'); // Debug log
      const isHealthy = await checkAPIHealth();
      console.log('Health check result:', isHealthy); // Debug log
      setApiState(prev => ({ ...prev, isHealthy, error: isHealthy ? null : 'API is not responding' }));
      return isHealthy;
    } catch (error) {
      console.error('Health check failed:', error);
      setApiState(prev => ({ ...prev, isHealthy: false, error: 'Failed to check API health' }));
      return false;
    }
  }, []);

  // Fetch seismograph data
  const fetchSeismographData = useCallback(async () => {
    if (!apiState.isHealthy) return;

    try {
      setApiState(prev => ({ ...prev, isLoading: true, error: null }));
      
      const data = await apiClient.getSeismographData();
      console.log('Raw API data:', data); // Debug log
      const transformedData = transformSeismographData(data);
      console.log('Transformed data:', transformedData); // Debug log
      
      setSeismographData(transformedData);
      setApiState(prev => ({ 
        ...prev, 
        isLoading: false, 
        lastUpdate: new Date(),
        error: null 
      }));
    } catch (error) {
      console.error('Failed to fetch seismograph data:', error);
      setApiState(prev => ({ 
        ...prev, 
        isLoading: false, 
        error: 'Failed to load seismograph data' 
      }));
      // Keep existing data if available
    }
  }, [apiState.isHealthy]);

  // Fetch tremors
  const fetchTremors = useCallback(async () => {
    if (!apiState.isHealthy) return;

    try {
      const tremorData = await apiClient.getTremors();
      setTremors(tremorData);
    } catch (error) {
      console.error('Failed to fetch tremors:', error);
      setApiState(prev => ({ ...prev, error: 'Failed to load tremors' }));
    }
  }, [apiState.isHealthy]);

  // Get epicenter analysis for a specific tremor
  const getEpicenterAnalysis = useCallback(async (tremorId: string): Promise<EventInsight | null> => {
    if (!apiState.isHealthy) return null;

    try {
      setApiState(prev => ({ ...prev, isLoading: true }));
      
      // Find the tremor first
      const tremor = tremors.find(t => t.id === tremorId);
      if (!tremor) {
        throw new Error('Tremor not found');
      }

      // Get detailed epicenter analysis
      const epicenterData = await apiClient.getEpicenterAnalysis(tremorId);
      
      // Transform to EventInsight format
      const insight = transformTremorToEventInsight(tremor, epicenterData);
      
      setApiState(prev => ({ ...prev, isLoading: false }));
      return insight;
    } catch (error) {
      console.error('Failed to get epicenter analysis:', error);
      setApiState(prev => ({ 
        ...prev, 
        isLoading: false, 
        error: 'Failed to load event analysis' 
      }));
      return null;
    }
  }, [tremors, apiState.isHealthy]);

  // Send query to AI
  const sendQuery = useCallback(async (query: string, useRag: boolean = true): Promise<string | null> => {
    if (!apiState.isHealthy) return null;

    try {
      setApiState(prev => ({ ...prev, isLoading: true }));
      
      const response = await apiClient.postQuery(query, useRag);
      
      setApiState(prev => ({ ...prev, isLoading: false }));
      // Handle different response formats from the new API
      return response.response || response.answer || response.result || JSON.stringify(response);
    } catch (error) {
      console.error('Failed to send query:', error);
      setApiState(prev => ({ 
        ...prev, 
        isLoading: false, 
        error: 'Failed to process query' 
      }));
      return null;
    }
  }, [apiState.isHealthy]);

  // Get system statistics for dashboard metrics
  const getSystemStats = useCallback(async () => {
    if (!apiState.isHealthy) return null;

    try {
      const stats = await apiClient.getSystemStats();
      return stats;
    } catch (error) {
      console.error('Failed to get system stats:', error);
      return null;
    }
  }, [apiState.isHealthy]);

  // Initial health check on mount
  useEffect(() => {
    performHealthCheck();
  }, [performHealthCheck]);

  // Fetch initial data when API becomes healthy
  useEffect(() => {
    if (apiState.isHealthy) {
      fetchSeismographData();
      fetchTremors();
    }
  }, [apiState.isHealthy, fetchSeismographData, fetchTremors]);

  // Set up periodic refresh
  useEffect(() => {
    if (!apiState.isHealthy) return;

    const interval = setInterval(() => {
      fetchSeismographData();
      fetchTremors();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [apiState.isHealthy, fetchSeismographData, fetchTremors]);

  // Retry function for when API is down
  const retryConnection = useCallback(async () => {
    await performHealthCheck();
  }, [performHealthCheck]);

  return {
    // Data
    seismographData,
    tremors,
    apiState,
    
    // Actions
    getEpicenterAnalysis,
    sendQuery,
    getSystemStats,
    retryConnection,
    performHealthCheck,
    
    // Manual refresh functions
    refreshSeismographData: fetchSeismographData,
    refreshTremors: fetchTremors,
  };
};