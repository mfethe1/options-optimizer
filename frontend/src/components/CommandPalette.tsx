import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

interface Command {
  id: string;
  name: string;
  description: string;
  action: () => void;
  keywords: string[];
  shortcut?: string;
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  // Define all commands
  const commands: Command[] = [
    // Navigation
    {
      id: 'nav-dashboard',
      name: 'Dashboard',
      description: 'Go to main dashboard',
      action: () => navigate('/'),
      keywords: ['dashboard', 'home', 'main'],
      shortcut: 'Ctrl+D'
    },
    {
      id: 'nav-custom-dashboard',
      name: 'Custom Dashboard',
      description: 'Personalized widget dashboard',
      action: () => navigate('/custom-dashboard'),
      keywords: ['custom', 'dashboard', 'widgets', 'personalize'],
      shortcut: 'Ctrl+W'
    },
    {
      id: 'nav-options-chain',
      name: 'Options Chain',
      description: 'View real-time options chain',
      action: () => navigate('/options-chain'),
      keywords: ['options', 'chain', 'omon', 'greeks'],
      shortcut: 'Ctrl+O'
    },
    {
      id: 'nav-options-analytics',
      name: 'Options Analytics',
      description: 'IV surface, skew, term structure',
      action: () => navigate('/options-analytics'),
      keywords: ['options', 'analytics', 'volatility', 'iv', 'skew', 'term structure', 'surface'],
      shortcut: 'Ctrl+V'
    },
    {
      id: 'nav-risk-dashboard',
      name: 'Risk Dashboard',
      description: 'View portfolio risk metrics',
      action: () => navigate('/risk-dashboard'),
      keywords: ['risk', 'var', 'portfolio', 'port', 'greeks'],
      shortcut: 'Ctrl+R'
    },
    {
      id: 'nav-calendar',
      name: 'Economic Calendar',
      description: 'Earnings and economic events',
      action: () => navigate('/calendar'),
      keywords: ['calendar', 'earnings', 'events', 'economic', 'fed', 'cpi', 'evts'],
      shortcut: 'Ctrl+E'
    },
    {
      id: 'nav-backtest',
      name: 'Strategy Backtesting',
      description: 'Test historical strategy performance',
      action: () => navigate('/backtest'),
      keywords: ['backtest', 'test', 'strategy', 'historical', 'performance', 'metrics'],
      shortcut: 'Ctrl+B'
    },
    {
      id: 'nav-execution',
      name: 'Execution Quality',
      description: 'Track fill quality and slippage',
      action: () => navigate('/execution'),
      keywords: ['execution', 'quality', 'slippage', 'fill', 'broker', 'trade'],
      shortcut: 'Ctrl+X'
    },
    {
      id: 'nav-schwab-connection',
      name: 'Schwab Connection',
      description: 'Connect to Charles Schwab account',
      action: () => navigate('/schwab-connection'),
      keywords: ['schwab', 'connection', 'oauth', 'login', 'account', 'link', 'broker'],
      shortcut: 'Ctrl+L'
    },
    {
      id: 'nav-schwab-trading',
      name: 'Schwab Live Trading',
      description: 'Place live orders with Schwab',
      action: () => navigate('/schwab-trading'),
      keywords: ['schwab', 'trading', 'live', 'order', 'trade', 'execute', 'real'],
      shortcut: 'Ctrl+U'
    },
    {
      id: 'nav-multi-monitor',
      name: 'Multi-Monitor Layout',
      description: 'Professional multi-screen workspace',
      action: () => navigate('/multi-monitor'),
      keywords: ['multi', 'monitor', 'layout', 'workspace', 'screens', 'windows', 'bloomberg'],
      shortcut: 'Ctrl+M'
    },
    {
      id: 'nav-ai-recommendations',
      name: 'AI Recommendations',
      description: 'AI-powered platform analysis and insights',
      action: () => navigate('/ai-recommendations'),
      keywords: ['ai', 'recommendations', 'swarm', 'analysis', 'critique', 'risk', 'insights', 'expert'],
      shortcut: 'Ctrl+I'
    },
    {
      id: 'nav-news',
      name: 'News Feed',
      description: 'View financial news',
      action: () => navigate('/news'),
      keywords: ['news', 'headlines', 'articles'],
      shortcut: 'Ctrl+N'
    },
    {
      id: 'nav-chart',
      name: 'Chart Analysis',
      description: 'Analyze charts with AI',
      action: () => navigate('/chart-analysis'),
      keywords: ['chart', 'analysis', 'technical', 'vision'],
      shortcut: 'Ctrl+C'
    },
    {
      id: 'nav-sentiment',
      name: 'Sentiment Analysis',
      description: 'View market sentiment',
      action: () => navigate('/sentiment'),
      keywords: ['sentiment', 'social', 'twitter', 'mood'],
      shortcut: 'Ctrl+S'
    },
    {
      id: 'nav-anomalies',
      name: 'Anomaly Detection',
      description: 'Detect market anomalies',
      action: () => navigate('/anomalies'),
      keywords: ['anomaly', 'alerts', 'detection', 'unusual'],
      shortcut: 'Ctrl+A'
    },
    {
      id: 'nav-paper-trading',
      name: 'Paper Trading',
      description: 'AI-powered paper trading',
      action: () => navigate('/paper-trading'),
      keywords: ['paper', 'trading', 'simulation', 'test'],
      shortcut: 'Ctrl+T'
    },
    {
      id: 'nav-conversational',
      name: 'Conversational Trading',
      description: 'Chat with AI trading assistant',
      action: () => navigate('/conversational'),
      keywords: ['chat', 'conversational', 'ai', 'assistant'],
      shortcut: 'Ctrl+Shift+C'
    },
    {
      id: 'nav-positions',
      name: 'Positions',
      description: 'Manage your positions',
      action: () => navigate('/positions'),
      keywords: ['positions', 'portfolio', 'holdings'],
      shortcut: 'Ctrl+P'
    },

    // Actions
    {
      id: 'action-refresh',
      name: 'Refresh Data',
      description: 'Refresh all data',
      action: () => window.location.reload(),
      keywords: ['refresh', 'reload', 'update'],
      shortcut: 'Ctrl+Shift+R'
    },
    {
      id: 'action-search-symbol',
      name: 'Search Symbol',
      description: 'Search for a stock symbol',
      action: () => {
        const symbol = prompt('Enter symbol:');
        if (symbol) navigate(`/options-chain?symbol=${symbol.toUpperCase()}`);
      },
      keywords: ['search', 'symbol', 'stock', 'ticker'],
      shortcut: 'Ctrl+K'
    },
    {
      id: 'action-help',
      name: 'Help',
      description: 'Show help and keyboard shortcuts',
      action: () => alert('Keyboard Shortcuts:\nCtrl+K - Command Palette\nCtrl+D - Dashboard\nCtrl+W - Custom Dashboard\nCtrl+O - Options Chain\nCtrl+V - Options Analytics\nCtrl+R - Risk Dashboard\nCtrl+E - Economic Calendar\nCtrl+B - Backtest\nCtrl+X - Execution Quality\nCtrl+L - Schwab Connection\nCtrl+U - Schwab Trading\nCtrl+M - Multi-Monitor\nCtrl+I - AI Recommendations\nCtrl+N - News\nCtrl+C - Charts\nCtrl+S - Sentiment\nCtrl+A - Anomalies\nCtrl+T - Paper Trading\nCtrl+P - Positions'),
      keywords: ['help', 'shortcuts', 'keyboard', 'commands'],
      shortcut: '?'
    }
  ];

  // Filter commands based on query
  const filteredCommands = query.length > 0
    ? commands.filter(cmd => {
        const searchTerms = `${cmd.name} ${cmd.description} ${cmd.keywords.join(' ')}`.toLowerCase();
        return searchTerms.includes(query.toLowerCase());
      })
    : commands;

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev =>
            prev < filteredCommands.length - 1 ? prev + 1 : prev
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            executeCommand(filteredCommands[selectedIndex]);
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selectedIndex, filteredCommands, onClose]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const executeCommand = (command: Command) => {
    command.action();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50"
        onClick={onClose}
      />

      {/* Command Palette */}
      <div className="relative bg-white rounded-lg shadow-2xl w-full max-w-2xl">
        {/* Search Input */}
        <div className="p-4 border-b border-gray-200">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Type a command or search..."
            className="w-full px-4 py-3 text-lg border-0 focus:outline-none focus:ring-0"
          />
        </div>

        {/* Command List */}
        <div className="max-h-96 overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              No commands found
            </div>
          ) : (
            <div className="py-2">
              {filteredCommands.map((command, index) => (
                <button
                  key={command.id}
                  onClick={() => executeCommand(command)}
                  className={`w-full px-4 py-3 flex items-center justify-between hover:bg-gray-100 transition-colors ${
                    index === selectedIndex ? 'bg-blue-50 border-l-4 border-blue-600' : ''
                  }`}
                  onMouseEnter={() => setSelectedIndex(index)}
                >
                  <div className="flex-1 text-left">
                    <div className="font-semibold text-gray-900">
                      {command.name}
                    </div>
                    <div className="text-sm text-gray-600">
                      {command.description}
                    </div>
                  </div>
                  {command.shortcut && (
                    <div className="ml-4 px-2 py-1 bg-gray-200 rounded text-xs font-mono text-gray-700">
                      {command.shortcut}
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-200 bg-gray-50 text-xs text-gray-600 flex items-center justify-between">
          <span>
            <kbd className="px-2 py-1 bg-white border border-gray-300 rounded">↑↓</kbd> Navigate
            <kbd className="ml-2 px-2 py-1 bg-white border border-gray-300 rounded">Enter</kbd> Select
            <kbd className="ml-2 px-2 py-1 bg-white border border-gray-300 rounded">Esc</kbd> Close
          </span>
          <span className="text-gray-500">
            {filteredCommands.length} commands
          </span>
        </div>
      </div>
    </div>
  );
};

export default CommandPalette;
