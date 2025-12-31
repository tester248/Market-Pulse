import React, { useState, useEffect } from 'react';
import { SeismographChart } from './SeismographChart';
import { EpicenterPanel } from './EpicenterPanel';
import { CommandQuery } from './CommandQuery';
import { NewsTicker } from './NewsTicker';
import { SystemStatus } from './SystemStatus';
import { RecentTremors } from './RecentTremors';
import { AIChatModal } from './AIChatModal';
import { SentimentData, EventInsight, TimeframeOption } from '../types';
import { generateMockSentimentData, timeframeOptions } from '../data/mockData';
import { useFinancialAPI } from '../hooks/useFinancialAPI';
import { Activity, Zap, Clock, AlertCircle, Wifi, WifiOff, MessageSquare } from 'lucide-react';
import { ThemeSwitcher } from './ThemeSwitcher';

export const Dashboard: React.FC = () => {
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [selectedInsight, setSelectedInsight] = useState<EventInsight | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframeOption>(timeframeOptions[2]); // 24H default
  const [isLoading, setIsLoading] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatInitialQuery, setChatInitialQuery] = useState<string>('');

  // Use the Financial API hook
  const {
    seismographData,
    apiState,
    getEpicenterAnalysis,
    retryConnection
  } = useFinancialAPI();

  // Fallback to mock data if API is not available
  const [mockData, setMockData] = useState<SentimentData[]>([]);
  const displayData = apiState.isHealthy && seismographData.length > 0 ? seismographData : mockData;

  // Load mock data as fallback
  useEffect(() => {
    const data = generateMockSentimentData(selectedTimeframe.hours);
    setMockData(data);
  }, [selectedTimeframe]);

  // Simulate real-time updates for mock data only
  useEffect(() => {
    if (apiState.isHealthy) return; // Don't run mock updates when API is working

    const interval = setInterval(() => {
      setMockData(current => {
        const newData = [...current];
        // Add new data point
        const now = new Date();
        const newPoint: SentimentData = {
          timestamp: now.toISOString(),
          sentiment: (Math.random() - 0.5) * 0.4,
          volume: Math.random() * 30 + 10,
          event_id: `live_${now.getTime()}`,
          ticker: ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'][Math.floor(Math.random() * 5)],
        };
        
        // Keep only recent data points
        const cutoffTime = new Date(now.getTime() - selectedTimeframe.hours * 60 * 60 * 1000);
        return [...newData.filter(d => new Date(d.timestamp) > cutoffTime), newPoint];
      });
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [selectedTimeframe, apiState.isHealthy]);

  const handleEventClick = async (eventId: string) => {
    setSelectedEventId(eventId);
    setIsLoading(true);
    
    if (apiState.isHealthy) {
      // Use real API for epicenter analysis
      const insight = await getEpicenterAnalysis(eventId);
      setSelectedInsight(insight);
    } else {
      // Simulate API call delay for mock data
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Use mock insight for demo
      const mockInsight: EventInsight = {
        event_id: eventId,
        headline: 'Market Event Analysis',
        summary: ['This is a simulated event analysis while the API is offline.'],
        causal_analysis: {
          sentiment: 'Neutral',
          confidence: 0.75,
          driver: 'Demo analysis - API connection needed for real insights',
        },
        source_consensus: [
          { source: 'Demo Source', stance: 'neutral' },
        ],
        pipeline_trace: 'Demo Mode - Connect to API for real pipeline traces',
        timestamp: new Date().toISOString(),
      };
      setSelectedInsight(mockInsight);
    }
    
    setIsLoading(false);
  };

  const handleClosePanel = () => {
    setSelectedEventId(null);
    setSelectedInsight(null);
  };

  const handleQuery = async (query: string) => {
    console.log('Processing query:', query);
    
    // Open chat modal with the query
    setChatInitialQuery(query);
    setIsChatOpen(true);
  };

  const handleTremorClick = (tremorId: string) => {
    handleEventClick(tremorId);
  };

  const currentSentiment = displayData.length > 0 
    ? displayData[displayData.length - 1].sentiment 
    : 0;
  
  const sentimentLabel = currentSentiment > 0.1 ? 'Bullish' : currentSentiment < -0.1 ? 'Bearish' : 'Neutral';
  const sentimentColor = currentSentiment > 0 ? 'text-emerald-600' : currentSentiment < 0 ? 'text-red-600' : 'text-slate-600';

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-200">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-6 py-4 transition-colors duration-200">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Market Pulse</h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">Real-time sentiment seismograph</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <ThemeSwitcher />
              
              {/* AI Chat Button */}
              <button
                onClick={() => setIsChatOpen(true)}
                className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <MessageSquare className="w-4 h-4" />
                <span className="text-sm font-medium">AI Chat</span>
              </button>
              
              {/* API Status Indicator */}
              <div className="flex items-center gap-2">
                {apiState.isHealthy ? (
                  <Wifi className="w-4 h-4 text-emerald-500" />
                ) : (
                  <WifiOff className="w-4 h-4 text-red-500" />
                )}
                <span className={`text-xs font-medium ${apiState.isHealthy ? 'text-emerald-600' : 'text-red-600'}`}>
                  {apiState.isHealthy ? 'API Connected' : 'API Offline'}
                </span>
                {!apiState.isHealthy && (
                  <button
                    onClick={retryConnection}
                    className="text-xs text-blue-600 hover:text-blue-700 underline"
                  >
                    Retry
                  </button>
                )}
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <Zap className="w-4 h-4 text-amber-500" />
                <span className="text-slate-600 dark:text-slate-400">Live:</span>
                <span className={`font-semibold ${sentimentColor}`}>
                  {sentimentLabel}
                </span>
                {apiState.error && (
                  <div title={apiState.error}>
                    <AlertCircle className="w-4 h-4 text-red-500 ml-2" />
                  </div>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                <select
                  value={selectedTimeframe.value}
                  onChange={(e) => {
                    const timeframe = timeframeOptions.find(t => t.value === e.target.value);
                    if (timeframe) setSelectedTimeframe(timeframe);
                  }}
                  className="px-3 py-1 border border-slate-300 dark:border-slate-600 rounded-lg text-sm bg-white dark:bg-slate-700 text-slate-900 dark:text-white"
                >
                  {timeframeOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Command Query Bar */}
        <div className="mb-8">
          <CommandQuery onQuery={handleQuery} />
        </div>

        {/* Seismograph Chart */}
        <div className="mb-8">
          <SeismographChart
            data={displayData}
            onEventClick={handleEventClick}
            selectedEventId={selectedEventId || undefined}
          />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 transition-colors duration-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Major Tremors</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {displayData.filter((d: SentimentData) => Math.abs(d.sentiment) > 0.5).length}
                </p>
              </div>
              <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
                <Zap className="w-5 h-5 text-red-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 transition-colors duration-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Bullish Events</p>
                <p className="text-2xl font-bold text-emerald-600">
                  {displayData.filter((d: SentimentData) => d.sentiment > 0.1).length}
                </p>
              </div>
              <div className="p-3 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg">
                <Activity className="w-5 h-5 text-emerald-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 transition-colors duration-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Bearish Events</p>
                <p className="text-2xl font-bold text-red-600">
                  {displayData.filter((d: SentimentData) => d.sentiment < -0.1).length}
                </p>
              </div>
              <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-lg">
                <Activity className="w-5 h-5 text-red-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700 transition-colors duration-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Avg Volume</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {displayData.length > 0 
                    ? Math.round(displayData.reduce((acc: number, d: SentimentData) => acc + d.volume, 0) / displayData.length)
                    : 0
                  }
                </p>
              </div>
              <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                <Clock className="w-5 h-5 text-blue-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Secondary Grid - News, Tremors, and System Status */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <NewsTicker maxArticles={6} />
          <RecentTremors 
            onTremorClick={handleTremorClick} 
            maxTremors={8}
          />
          <SystemStatus />
        </div>
      </main>

      {/* AI Chat Modal */}
      <AIChatModal
        isOpen={isChatOpen}
        onClose={() => {
          setIsChatOpen(false);
          setChatInitialQuery('');
        }}
        initialQuery={chatInitialQuery}
      />

      {/* Epicenter Panel */}
      <EpicenterPanel
        insight={isLoading ? null : selectedInsight}
        onClose={handleClosePanel}
      />
      
      {/* Loading Overlay for Panel */}
      {isLoading && selectedEventId && (
        <div className="fixed inset-0 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 z-60 flex items-center justify-center">
          <div className="bg-white dark:bg-slate-800 rounded-xl p-8 shadow-2xl">
            <div className="flex items-center gap-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <div>
                <p className="font-semibold text-slate-900 dark:text-white">Processing Event</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">Running multi-LLM analysis...</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};