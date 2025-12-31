import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { EventInsight } from '../types';
import { format } from 'date-fns';
import { 
  X, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Clock,
  Target,
  GitBranch,
  Check,
  AlertTriangle,
  Zap
} from 'lucide-react';

interface EpicenterPanelProps {
  insight: EventInsight | null;
  onClose: () => void;
}

const SourceStanceIcon = ({ stance }: { stance: string }) => {
  switch (stance) {
    case 'agrees':
      return <Check className="w-4 h-4 text-emerald-600" />;
    case 'disagrees':
      return <X className="w-4 h-4 text-red-600" />;
    default:
      return <Minus className="w-4 h-4 text-slate-500" />;
  }
};

const SentimentIcon = ({ sentiment }: { sentiment: string }) => {
  switch (sentiment) {
    case 'Bullish':
      return <TrendingUp className="w-5 h-5 text-emerald-600" />;
    case 'Bearish':
      return <TrendingDown className="w-5 h-5 text-red-600" />;
    default:
      return <Minus className="w-5 h-5 text-slate-500" />;
  }
};

export const EpicenterPanel: React.FC<EpicenterPanelProps> = ({ insight, onClose }) => {
  return (
    <AnimatePresence>
      {insight && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
          />
          
          {/* Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white dark:bg-slate-900 shadow-2xl z-50 overflow-y-auto transition-colors duration-200"
          >
            <div className="p-6">
              {/* Header */}
              <div className="flex items-start justify-between mb-6">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Clock className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      {format(new Date(insight.timestamp), 'MMM dd, yyyy • HH:mm')}
                    </span>
                    {insight.ticker && (
                      <span className="px-2 py-1 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-medium rounded">
                        ${insight.ticker}
                      </span>
                    )}
                  </div>
                  <h1 className="text-2xl font-bold text-slate-900 dark:text-white leading-tight">
                    {insight.headline}
                  </h1>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                </button>
              </div>

              {/* Summary */}
              <div className="mb-8">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-3 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-amber-500" />
                  Executive Summary
                </h2>
                <div className="space-y-2">
                  {insight.summary.map((point, index) => (
                    <div key={index} className="flex items-start gap-3">
                      <div className="w-2 h-2 bg-slate-400 dark:bg-slate-500 rounded-full mt-2 flex-shrink-0" />
                      <p className="text-slate-700 dark:text-slate-300 leading-relaxed">{point}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Causal Analysis */}
              <div className="mb-8 bg-slate-50 dark:bg-slate-800 rounded-xl p-6 transition-colors duration-200">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-500" />
                  Causal Analysis
                </h2>
                
                <div className="flex items-center gap-4 mb-4">
                  <div className="flex items-center gap-2">
                    <SentimentIcon sentiment={insight.causal_analysis.sentiment} />
                    <span className={`font-semibold ${
                      insight.causal_analysis.sentiment === 'Bullish' 
                        ? 'text-emerald-600' 
                        : insight.causal_analysis.sentiment === 'Bearish'
                        ? 'text-red-600'
                        : 'text-slate-600'
                    }`}>
                      {insight.causal_analysis.sentiment}
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-600 dark:text-slate-400">Confidence:</span>
                    <div className="flex items-center gap-1">
                      <div className="w-20 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 transition-all duration-300"
                          style={{ width: `${insight.causal_analysis.confidence * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {Math.round(insight.causal_analysis.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                </div>

                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                  {insight.causal_analysis.driver}
                </p>
              </div>

              {/* Source Consensus */}
              <div className="mb-8">
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-orange-500" />
                  Source Consensus
                </h2>
                
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {insight.source_consensus.map((source, index) => (
                    <div
                      key={index}
                      className={`flex items-center gap-3 p-3 rounded-lg border ${
                        source.stance === 'agrees'
                          ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/30'
                          : source.stance === 'disagrees'
                          ? 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/30'
                          : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800'
                      }`}
                    >
                      <SourceStanceIcon stance={source.stance} />
                      <span className="text-sm font-medium text-slate-800 dark:text-slate-200">
                        {source.source}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Pipeline Trace */}
              <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-6 transition-colors duration-200">
                <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <GitBranch className="w-5 h-5 text-green-400" />
                  Pipeline Trace
                </h2>
                <div className="font-mono text-green-400 text-sm bg-slate-800 dark:bg-black rounded-lg p-3">
                  {insight.pipeline_trace}
                </div>
                <p className="text-slate-400 dark:text-slate-500 text-xs mt-2">
                  Multi-LLM synthesis pipeline • Processing time: ~2.3s
                </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};