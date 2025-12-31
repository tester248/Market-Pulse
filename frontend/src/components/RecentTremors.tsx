import React, { useState, useEffect } from 'react';
import { useFinancialAPI } from '../hooks/useFinancialAPI';
import { 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  Target, 
  AlertTriangle,
  Zap,
  ExternalLink,
  Filter
} from 'lucide-react';
import { format } from 'date-fns';

interface RecentTremorsProps {
  onTremorClick?: (tremorId: string) => void;
  maxTremors?: number;
}

type IntensityFilter = 'all' | 'high' | 'medium' | 'low';
type SentimentFilter = 'all' | 'bullish' | 'bearish' | 'neutral';

export const RecentTremors: React.FC<RecentTremorsProps> = ({ 
  onTremorClick, 
  maxTremors = 10 
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [intensityFilter, setIntensityFilter] = useState<IntensityFilter>('all');
  const [sentimentFilter, setSentimentFilter] = useState<SentimentFilter>('all');
  const [showFilters, setShowFilters] = useState(false);
  
  const { apiState, tremors, refreshTremors } = useFinancialAPI();

  const fetchTremors = async () => {
    if (!apiState.isHealthy) return;

    try {
      setIsLoading(true);
      await refreshTremors();
    } catch (error) {
      console.error('Failed to fetch tremors:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (apiState.isHealthy) {
      fetchTremors();
    }
  }, [apiState.isHealthy, intensityFilter]);

  // Filter tremors based on selected filters
  const filteredTremors = tremors
    .filter(tremor => {
      // Sentiment filter
      if (sentimentFilter !== 'all') {
        const sentiment = tremor.sentiment_score > 0.1 ? 'bullish' : 
                         tremor.sentiment_score < -0.1 ? 'bearish' : 'neutral';
        if (sentiment !== sentimentFilter) return false;
      }
      
      return true;
    })
    .slice(0, maxTremors);

  const getSentimentIcon = (sentimentScore: number) => {
    if (sentimentScore > 0.1) {
      return <TrendingUp className="w-4 h-4 text-emerald-600" />;
    } else if (sentimentScore < -0.1) {
      return <TrendingDown className="w-4 h-4 text-red-600" />;
    }
    return <Target className="w-4 h-4 text-slate-500" />;
  };

  const getSentimentColor = (sentimentScore: number) => {
    if (sentimentScore > 0.1) return 'text-emerald-600';
    if (sentimentScore < -0.1) return 'text-red-600';
    return 'text-slate-600';
  };

  const getIntensityBadge = (confidence: number) => {
    if (confidence >= 0.8) {
      return (
        <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 text-xs rounded-full font-medium">
          High
        </span>
      );
    } else if (confidence >= 0.6) {
      return (
        <span className="px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-xs rounded-full font-medium">
          Medium
        </span>
      );
    }
    return (
      <span className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 text-xs rounded-full font-medium">
        Low
      </span>
    );
  };

  if (!apiState.isHealthy) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-slate-500" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Recent Tremors</h3>
        </div>
        <div className="text-center py-8">
          <p className="text-slate-500 dark:text-slate-400">
            Connect to API to see recent market tremors
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Recent Tremors</h3>
          {isLoading && (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="p-2 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors"
          >
            <Filter className="w-4 h-4 text-slate-500 dark:text-slate-400" />
          </button>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {filteredTremors.length} tremors
          </span>
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="mb-4 p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg space-y-3">
          <div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-2 block">
              Intensity
            </label>
            <div className="flex gap-2">
              {(['all', 'high', 'medium', 'low'] as IntensityFilter[]).map((filter) => (
                <button
                  key={filter}
                  onClick={() => setIntensityFilter(filter)}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    intensityFilter === filter
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-500'
                  }`}
                >
                  {filter.charAt(0).toUpperCase() + filter.slice(1)}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-2 block">
              Sentiment
            </label>
            <div className="flex gap-2">
              {(['all', 'bullish', 'bearish', 'neutral'] as SentimentFilter[]).map((filter) => (
                <button
                  key={filter}
                  onClick={() => setSentimentFilter(filter)}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    sentimentFilter === filter
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-400 hover:bg-slate-300 dark:hover:bg-slate-500'
                  }`}
                >
                  {filter.charAt(0).toUpperCase() + filter.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tremors List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredTremors.length === 0 && !isLoading ? (
          <p className="text-center text-slate-500 dark:text-slate-400 py-4">
            No tremors found matching your filters
          </p>
        ) : (
          filteredTremors.map((tremor) => (
            <div
              key={tremor.id}
              className="border border-slate-100 dark:border-slate-700 rounded-lg p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer"
              onClick={() => onTremorClick?.(tremor.id)}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    {getSentimentIcon(tremor.sentiment_score)}
                    <h4 className="font-medium text-slate-900 dark:text-white text-sm leading-tight truncate">
                      {tremor.title}
                    </h4>
                  </div>
                  
                  <p className="text-xs text-slate-600 dark:text-slate-400 mb-3 line-clamp-2">
                    {tremor.summary}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {format(new Date(tremor.timestamp), 'MMM dd, HH:mm')}
                      </div>
                      <span className="px-2 py-1 bg-slate-100 dark:bg-slate-600 rounded text-xs">
                        {tremor.source}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      {getIntensityBadge(tremor.confidence)}
                      <ExternalLink className="w-3 h-3 text-slate-400" />
                    </div>
                  </div>

                  {tremor.tickers.length > 0 && (
                    <div className="flex items-center gap-1 mt-2">
                      <span className="text-xs text-slate-500 dark:text-slate-400">Tickers:</span>
                      {tremor.tickers.slice(0, 3).map((ticker) => (
                        <span
                          key={ticker}
                          className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 text-xs rounded font-mono"
                        >
                          {ticker}
                        </span>
                      ))}
                      {tremor.tickers.length > 3 && (
                        <span className="text-xs text-slate-500 dark:text-slate-400">
                          +{tremor.tickers.length - 3} more
                        </span>
                      )}
                    </div>
                  )}

                  <div className="flex items-center justify-between mt-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500 dark:text-slate-400">Sentiment:</span>
                      <span className={`text-xs font-medium ${getSentimentColor(tremor.sentiment_score)}`}>
                        {tremor.sentiment_score > 0 ? '+' : ''}{(tremor.sentiment_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      Quality: {tremor.quality_grade}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {filteredTremors.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
          <button
            onClick={fetchTremors}
            disabled={isLoading}
            className="w-full py-2 text-sm text-blue-600 hover:text-blue-700 transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Refreshing...' : 'Refresh Tremors'}
          </button>
        </div>
      )}
    </div>
  );
};