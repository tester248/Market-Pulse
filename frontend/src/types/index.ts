export interface SentimentData {
  timestamp: string;
  sentiment: number; // -1 to 1, where 0 is neutral
  volume: number; // 0-100, intensity of the event
  event_id: string;
  ticker?: string;
  headline?: string;
}

export interface EventInsight {
  event_id: string;
  headline: string;
  summary: string[];
  causal_analysis: {
    sentiment: 'Bullish' | 'Bearish' | 'Neutral';
    confidence: number;
    driver: string;
  };
  source_consensus: SourceStance[];
  pipeline_trace: string;
  timestamp: string;
  ticker?: string;
}

export interface SourceStance {
  source: string;
  stance: 'agrees' | 'neutral' | 'disagrees';
  logo?: string;
}

export interface TimeframeOption {
  label: string;
  value: string;
  hours: number;
}