import React, { useState, useEffect } from 'react';
import { useFinancialAPI } from '../hooks/useFinancialAPI';
import { apiClient, ApiSystemStats } from '../services/api';
import { 
  Database, 
  Brain, 
  Rss, 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp
} from 'lucide-react';

export const SystemStatus: React.FC = () => {
  const { apiState } = useFinancialAPI();
  const [systemStats, setSystemStats] = useState<ApiSystemStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchSystemStats = async () => {
    if (!apiState.isHealthy) return;

    try {
      setIsLoading(true);
      const stats = await apiClient.getSystemStats();
      setSystemStats(stats);
    } catch (error) {
      console.error('Failed to fetch system stats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (apiState.isHealthy) {
      fetchSystemStats();
    }
  }, [apiState.isHealthy]);

  // Refresh stats every minute
  useEffect(() => {
    if (!apiState.isHealthy) return;

    const interval = setInterval(fetchSystemStats, 60000);
    return () => clearInterval(interval);
  }, [apiState.isHealthy]);

  const StatusIcon = ({ status }: { status: boolean }) => (
    status ? (
      <CheckCircle className="w-4 h-4 text-emerald-500" />
    ) : (
      <XCircle className="w-4 h-4 text-red-500" />
    )
  );

  if (!apiState.isHealthy) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-slate-500" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">System Status</h3>
        </div>
        <div className="text-center py-8">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
          <p className="text-slate-500 dark:text-slate-400 mb-2">API Disconnected</p>
          <p className="text-xs text-slate-400">Connect to backend to see system status</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-emerald-600" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">System Status</h3>
          {isLoading && (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          )}
        </div>
        <span className="text-xs text-emerald-600 font-medium">
          {systemStats?.integration_status || 'Connected'}
        </span>
      </div>

      {systemStats && (
        <div className="space-y-4">
          {/* Component Status */}
          <div className="grid grid-cols-1 gap-3">
            <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                <span className="text-sm text-slate-700 dark:text-slate-300">Database</span>
              </div>
              <StatusIcon status={systemStats.system_health?.database_connected || false} />
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                <span className="text-sm text-slate-700 dark:text-slate-300">LLM Manager</span>
              </div>
              <StatusIcon status={systemStats.system_health?.llm_manager_active || false} />
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Rss className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                <span className="text-sm text-slate-700 dark:text-slate-300">RSS Feeds</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  {systemStats.seismograph?.active_feeds || 0} active
                </span>
                <StatusIcon status={(systemStats.seismograph?.active_feeds || 0) > 0} />
              </div>
            </div>

            <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-slate-600 dark:text-slate-400" />
                <span className="text-sm text-slate-700 dark:text-slate-300">Orchestrator</span>
              </div>
              <StatusIcon status={systemStats.system_health?.orchestrator_ready || false} />
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="border-t border-slate-200 dark:border-slate-600 pt-4">
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">Performance</h4>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <p className="text-slate-500 dark:text-slate-400">Articles Processed</p>
                <p className="font-semibold text-slate-900 dark:text-white">
                  {(systemStats.articles_processed || 0).toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-slate-500 dark:text-slate-400">Success Rate</p>
                <p className="font-semibold text-slate-900 dark:text-white">
                  {((systemStats.success_rate || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-slate-500 dark:text-slate-400">Avg Processing</p>
                <p className="font-semibold text-slate-900 dark:text-white">
                  {(systemStats.avg_processing_time_ms || 0).toFixed(0)}ms
                </p>
              </div>
              <div>
                <p className="text-slate-500 dark:text-slate-400">Data Points</p>
                <p className="font-semibold text-slate-900 dark:text-white">
                  {systemStats.seismograph?.data_points?.toLocaleString() || '0'}
                </p>
              </div>
            </div>
          </div>

          {/* Last Update */}
          {systemStats.last_processed_at && (
            <div className="border-t border-slate-200 dark:border-slate-600 pt-3">
              <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                <Clock className="w-3 h-3" />
                <span>
                  Last update: {new Date(systemStats.last_processed_at).toLocaleString()}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};