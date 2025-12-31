// Test file for the new Market Pulse API integration
import { apiClient } from '../services/api';

export const testNewAPIIntegration = async () => {
  console.log('Testing Market Pulse API integration...');
  
  try {
    // Test 1: Health check
    console.log('1. Testing health endpoint...');
    const health = await apiClient.getSystemHealth();
    console.log('Health check result:', health);
    
    // Test 2: Timeline data
    console.log('2. Testing pulse timeline endpoint...');
    const timelineData = await apiClient.getSeismographData(24);
    console.log('Timeline data:', timelineData);
    
    // Test 3: Query endpoint
    console.log('3. Testing insights query endpoint...');
    const queryResponse = await apiClient.postQuery('What is the current market sentiment?');
    console.log('Query response:', queryResponse);
    
    // Test 4: System stats (derived from health)
    console.log('4. Testing system stats...');
    const stats = await apiClient.getSystemStats();
    console.log('System stats:', stats);
    
    console.log('✅ All API tests completed successfully!');
    return true;
  } catch (error) {
    console.error('❌ API test failed:', error);
    return false;
  }
};

// Function to test individual endpoints
export const testHealthEndpoint = async () => {
  try {
    const health = await apiClient.getSystemHealth();
    console.log('Health endpoint working:', health);
    return health;
  } catch (error) {
    console.error('Health endpoint failed:', error);
    throw error;
  }
};

export const testTimelineEndpoint = async (hours: number = 24) => {
  try {
    const timeline = await apiClient.getSeismographData(hours);
    console.log('Timeline endpoint working:', timeline);
    return timeline;
  } catch (error) {
    console.error('Timeline endpoint failed:', error);
    throw error;
  }
};

export const testQueryEndpoint = async (query: string = 'Test query') => {
  try {
    const response = await apiClient.postQuery(query);
    console.log('Query endpoint working:', response);
    return response;
  } catch (error) {
    console.error('Query endpoint failed:', error);
    throw error;
  }
};

// Expose test functions to window for easy browser console access
if (typeof window !== 'undefined') {
  (window as unknown as {testMarketPulseAPI: {
    testAll: typeof testNewAPIIntegration;
    testHealth: typeof testHealthEndpoint;
    testTimeline: typeof testTimelineEndpoint;
    testQuery: typeof testQueryEndpoint;
  }}).testMarketPulseAPI = {
    testAll: testNewAPIIntegration,
    testHealth: testHealthEndpoint,
    testTimeline: testTimelineEndpoint,
    testQuery: testQueryEndpoint,
  };
}