import { SentimentData, EventInsight } from '../types';
import { subHours } from 'date-fns';

// Generate mock seismograph data
export const generateMockSentimentData = (hours: number = 24): SentimentData[] => {
  const data: SentimentData[] = [];
  const now = new Date();
  
  // Generate data points every 30 minutes
  for (let i = 0; i < hours * 2; i++) {
    const timestamp = subHours(now, (hours * 2 - i) * 0.5);
    
    // Create realistic market sentiment patterns
    const baseNoise = (Math.random() - 0.5) * 0.2;
    let sentiment = baseNoise;
    let volume = Math.random() * 20 + 5;
    
    // Add major events occasionally
    if (Math.random() < 0.1) {
      sentiment = (Math.random() - 0.5) * 1.8; // Major tremor
      volume = Math.random() * 60 + 40;
    }
    
    data.push({
      timestamp: timestamp.toISOString(),
      sentiment: Math.max(-1, Math.min(1, sentiment)),
      volume,
      event_id: `event_${i}_${Math.random().toString(36).substr(2, 9)}`,
      ticker: ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'][Math.floor(Math.random() * 5)],
    });
  }
  
  return data.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
};

// Mock detailed insights for events
export const mockEventInsights: Record<string, EventInsight> = {
  default: {
    event_id: 'default',
    headline: 'NVIDIA Announces Revolutionary AI Chip Architecture',
    summary: [
      'NVIDIA unveiled its next-generation AI processing unit with 40% improved performance',
      'Market concerns emerge over pricing strategy and competitive positioning'
    ],
    causal_analysis: {
      sentiment: 'Bearish',
      confidence: 0.87,
      driver: 'Despite technological advancement, sentiment is driven by concerns over high manufacturing costs and potential market adoption barriers, as highlighted across multiple financial news sources.'
    },
    source_consensus: [
      { source: 'Reuters', stance: 'agrees' },
      { source: 'Bloomberg', stance: 'agrees' },
      { source: 'Wall Street Journal', stance: 'neutral' },
      { source: 'Reddit r/investing', stance: 'disagrees' },
      { source: 'Financial Times', stance: 'agrees' },
      { source: 'MarketWatch', stance: 'neutral' }
    ],
    pipeline_trace: 'Llama 3 Triage → FinBERT Sentiment + FinGPT Summary → Gemini Synthesis',
    timestamp: new Date().toISOString(),
    ticker: 'NVDA'
  }
};

export const timeframeOptions = [
  { label: '6H', value: '6h', hours: 6 },
  { label: '12H', value: '12h', hours: 12 },
  { label: '24H', value: '24h', hours: 24 },
  { label: '48H', value: '48h', hours: 48 },
  { label: '7D', value: '7d', hours: 168 },
];