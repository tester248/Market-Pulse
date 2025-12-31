import React, { useState } from 'react';
import { Search, Zap, Clock, TrendingUp } from 'lucide-react';

interface CommandQueryProps {
  onQuery: (query: string) => void;
  placeholder?: string;
}

const SUGGESTED_QUERIES = [
  'Compare AAPL vs GOOGL sentiment over 24h',
  'Show me bearish tremors for tech stocks',
  'What drove the NVIDIA spike at 14:30?',
  'Analyze correlation between Reddit and institutional sentiment',
];

export const CommandQuery: React.FC<CommandQueryProps> = ({ 
  onQuery, 
  placeholder = "Ask anything about market sentiment..." 
}) => {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onQuery(query.trim());
      setQuery('');
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    onQuery(suggestion);
    setShowSuggestions(false);
  };

  return (
    <div className="relative">
      <form onSubmit={handleSubmit} className="relative">
        <div className={`
          flex items-center bg-white dark:bg-slate-800 border rounded-xl shadow-sm transition-all duration-200
          ${isFocused ? 'border-blue-500 shadow-md' : 'border-slate-300 dark:border-slate-600'}
        `}>
          <Search className="w-5 h-5 text-slate-400 dark:text-slate-500 ml-4" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => {
              setIsFocused(true);
              setShowSuggestions(true);
            }}
            onBlur={() => {
              setIsFocused(false);
              setTimeout(() => setShowSuggestions(false), 200);
            }}
            placeholder={placeholder}
            className="flex-1 px-4 py-3 rounded-xl outline-none text-slate-900 dark:text-white placeholder-slate-500 dark:placeholder-slate-400 bg-transparent"
          />
          {query && (
            <button
              type="submit"
              className="mr-2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Zap className="w-4 h-4" />
            </button>
          )}
        </div>
      </form>

      {/* Suggestions Dropdown */}
      {showSuggestions && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl shadow-lg z-30 max-h-64 overflow-y-auto">
          <div className="p-3 border-b border-slate-100 dark:border-slate-700">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide">
              Suggested Queries
            </span>
          </div>
          {SUGGESTED_QUERIES.map((suggestion, index) => (
            <button
              key={index}
              onMouseDown={() => handleSuggestionClick(suggestion)}
              className="w-full text-left px-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-start gap-3 group"
            >
              <div className="p-1 bg-slate-100 dark:bg-slate-700 rounded group-hover:bg-blue-100 dark:group-hover:bg-blue-900/30 transition-colors">
                {index === 0 && <TrendingUp className="w-3 h-3 text-slate-600 dark:text-slate-400 group-hover:text-blue-600" />}
                {index === 1 && <TrendingUp className="w-3 h-3 text-slate-600 dark:text-slate-400 group-hover:text-blue-600" />}
                {index === 2 && <Clock className="w-3 h-3 text-slate-600 dark:text-slate-400 group-hover:text-blue-600" />}
                {index === 3 && <Search className="w-3 h-3 text-slate-600 dark:text-slate-400 group-hover:text-blue-600" />}
              </div>
              <span className="text-sm text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white">
                {suggestion}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};