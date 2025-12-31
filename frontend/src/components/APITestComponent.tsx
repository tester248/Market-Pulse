import React, { useEffect, useState } from 'react';
import { useFinancialAPI } from '../hooks/useFinancialAPI';

export const APITestComponent: React.FC = () => {
  const { apiState, seismographData, performHealthCheck } = useFinancialAPI();
  const [debugInfo, setDebugInfo] = useState<any>({});

  useEffect(() => {
    const testAPI = async () => {
      try {
        console.log('Testing API integration...');
        await performHealthCheck();
      } catch (error) {
        console.error('Test failed:', error);
        setDebugInfo({ error: error instanceof Error ? error.message : String(error) });
      }
    };

    testAPI();
  }, [performHealthCheck]);

  return (
    <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border">
      <h3 className="text-lg font-semibold mb-4">API Test Component</h3>
      
      <div className="space-y-2">
        <div>
          <strong>API Health:</strong> {apiState.isHealthy ? '✅ Healthy' : '❌ Unhealthy'}
        </div>
        <div>
          <strong>Loading:</strong> {apiState.isLoading ? 'Yes' : 'No'}
        </div>
        <div>
          <strong>Error:</strong> {apiState.error || 'None'}
        </div>
        <div>
          <strong>Last Update:</strong> {apiState.lastUpdate?.toISOString() || 'Never'}
        </div>
        <div>
          <strong>Data Points:</strong> {seismographData.length}
        </div>
        
        {debugInfo.error && (
          <div className="bg-red-100 p-2 rounded">
            <strong>Debug Error:</strong> {debugInfo.error}
          </div>
        )}
        
        <details className="mt-4">
          <summary className="cursor-pointer">Raw Data</summary>
          <pre className="text-xs bg-gray-100 p-2 rounded mt-2 overflow-auto">
            {JSON.stringify({ apiState, seismographData, debugInfo }, null, 2)}
          </pre>
        </details>
      </div>
    </div>
  );
};