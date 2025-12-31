import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Dot
} from 'recharts';
import { SentimentData } from '../types';
import { format } from 'date-fns';

interface SeismographChartProps {
  data: SentimentData[];
  onEventClick: (eventId: string) => void;
  selectedEventId?: string;
}

interface CustomDotProps {
  cx?: number;
  cy?: number;
  payload?: SentimentData;
  onClick?: (eventId: string) => void;
  isSelected?: boolean;
}

const CustomDot: React.FC<CustomDotProps> = ({ cx, cy, payload, onClick, isSelected }) => {
  if (!payload || !cx || !cy || Math.abs(payload.sentiment) < 0.3) return null;
  
  const isMajorEvent = payload.volume > 50;
  const radius = isMajorEvent ? 6 : 4;
  const color = payload.sentiment > 0 ? '#10B981' : '#EF4444';
  
  return (
    <Dot
      cx={cx}
      cy={cy}
      r={radius}
      fill={color}
      stroke={isSelected ? '#F59E0B' : color}
      strokeWidth={isSelected ? 3 : 1}
      className="cursor-pointer hover:stroke-yellow-500 transition-all duration-200 hover:r-7"
      onClick={() => onClick?.(payload.event_id)}
      style={{ filter: isMajorEvent ? 'drop-shadow(0 0 4px rgba(0,0,0,0.3))' : undefined }}
    />
  );
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload || !payload[0]) return null;
  
  const data = payload[0].payload as SentimentData;
  const sentiment = data.sentiment;
  const sentimentText = sentiment > 0.1 ? 'Bullish' : sentiment < -0.1 ? 'Bearish' : 'Neutral';
  const color = sentiment > 0 ? 'text-emerald-600' : sentiment < 0 ? 'text-red-600' : 'text-slate-600';
  
  return (
    <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-medium text-slate-900 dark:text-white">
        {format(new Date(data.timestamp), 'MMM dd, HH:mm')}
      </p>
      <p className={`font-semibold ${color}`}>
        {sentimentText} ({(sentiment * 100).toFixed(1)}%)
      </p>
      <p className="text-slate-600 dark:text-slate-400">
        Volume: {Math.round(data.volume)}
      </p>
      {data.ticker && (
        <p className="text-slate-500 dark:text-slate-400 text-xs">
          ${data.ticker}
        </p>
      )}
    </div>
  );
};

export const SeismographChart: React.FC<SeismographChartProps> = ({
  data,
  onEventClick,
  selectedEventId
}) => {
  const chartData = data.map(item => ({
    ...item,
    displayTime: format(new Date(item.timestamp), 'HH:mm'),
    amplitude: Math.abs(item.sentiment) * item.volume,
  }));

  return (
    <div className="h-80 w-full bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 transition-colors duration-200">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-1">Market Pulse Timeline</h2>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Live sentiment seismograph • Click tremors for detailed analysis
        </p>
      </div>
      
      <ResponsiveContainer width="100%" height="85%">
        <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="bullishGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10B981" stopOpacity={0.3}/>
              <stop offset="100%" stopColor="#10B981" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="bearishGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#EF4444" stopOpacity={0}/>
              <stop offset="100%" stopColor="#EF4444" stopOpacity={0.3}/>
            </linearGradient>
          </defs>
          
          <XAxis 
            dataKey="displayTime"
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: '#64748B' }}
            interval="preserveStartEnd"
          />
          <YAxis 
            domain={[-1, 1]}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: '#64748B' }}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          {/* Neutral reference line */}
          <ReferenceLine 
            y={0} 
            stroke="#94A3B8" 
            strokeDasharray="2 2" 
            strokeWidth={1}
          />
          
          {/* Bullish area (above 0) */}
          <Area
            type="monotone"
            dataKey="sentiment"
            stroke="#10B981"
            strokeWidth={2}
            fill="url(#bullishGradient)"
            connectNulls
            dot={<CustomDot onClick={onEventClick} isSelected={false} />}
          />
          
          {/* Interactive dots for events */}
          {chartData.map((item, index) => (
            <CustomDot
              key={item.event_id}
              cx={index * (100 / chartData.length) + '%' as any}
              cy={((1 - item.sentiment) / 2) * 100 + '%' as any}
              payload={item}
              onClick={onEventClick}
              isSelected={item.event_id === selectedEventId}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};