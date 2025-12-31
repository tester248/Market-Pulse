import React, { useState, useEffect } from 'react';
import { apiClient, ApiArticle } from '../services/api';
import { useFinancialAPI } from '../hooks/useFinancialAPI';
import { ExternalLink, Clock, Rss } from 'lucide-react';
import { format } from 'date-fns';

interface NewsTickerProps {
  maxArticles?: number;
}

export const NewsTicker: React.FC<NewsTickerProps> = ({ maxArticles = 5 }) => {
  const [articles, setArticles] = useState<ApiArticle[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { apiState } = useFinancialAPI();

  const fetchArticles = async () => {
    if (!apiState.isHealthy) return;

    try {
      setIsLoading(true);
      const data = await apiClient.getArticles(maxArticles);
      setArticles(data);
    } catch (error) {
      console.error('Failed to fetch articles:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (apiState.isHealthy) {
      fetchArticles();
    }
  }, [apiState.isHealthy]);

  // Refresh articles every 5 minutes
  useEffect(() => {
    if (!apiState.isHealthy) return;

    const interval = setInterval(fetchArticles, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [apiState.isHealthy]);

  if (!apiState.isHealthy) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <Rss className="w-5 h-5 text-slate-500" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Live News</h3>
        </div>
        <div className="text-center py-8">
          <p className="text-slate-500 dark:text-slate-400">
            Connect to API to see live financial news
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Rss className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Live News</h3>
          {isLoading && (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          )}
        </div>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {articles.length} articles
        </span>
      </div>

      <div className="space-y-4 max-h-96 overflow-y-auto">
        {articles.length === 0 && !isLoading ? (
          <p className="text-center text-slate-500 dark:text-slate-400 py-4">
            No articles available
          </p>
        ) : (
          articles.map((article) => (
            <div
              key={article.id}
              className="border border-slate-100 dark:border-slate-700 rounded-lg p-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <h4 className="font-medium text-slate-900 dark:text-white text-sm leading-tight mb-2">
                    {article.title}
                  </h4>
                  
                  <div className="flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400 mb-2">
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {format(new Date(article.published_at), 'MMM dd, HH:mm')}
                    </div>
                    <span className="px-2 py-1 bg-slate-100 dark:bg-slate-600 rounded text-xs">
                      {article.rss_feed_name}
                    </span>
                  </div>

                  {article.analysis_sentiment !== 0 && (
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs text-slate-600 dark:text-slate-400">Sentiment:</span>
                      <span className={`text-xs font-medium ${
                        article.analysis_sentiment > 0 
                          ? 'text-emerald-600' 
                          : 'text-red-600'
                      }`}>
                        {article.analysis_sentiment > 0 ? 'Bullish' : 'Bearish'}
                      </span>
                      <span className="text-xs text-slate-500">
                        ({(article.analysis_confidence * 100).toFixed(0)}%)
                      </span>
                    </div>
                  )}
                </div>

                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-shrink-0 p-2 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-lg transition-colors"
                  title="Open article"
                >
                  <ExternalLink className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                </a>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};