import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Home,
  TrendingUp,
  Assessment,
  AccountBalance,
  Psychology,
  Science,
  Speed,
  Biotech,
  ExpandMore,
  ExpandLess,
  ChevronRight,
  Dashboard,
  ShowChart,
  Analytics,
  Warning,
  CalendarMonth,
  Storage
} from '@mui/icons-material';

interface NavItem {
  label: string;
  path?: string;
  icon?: React.ReactNode;
  children?: NavItem[];
  description?: string;
}

const NavigationSidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedSections, setExpandedSections] = useState<string[]>(['neural-networks']);

  const navigationItems: NavItem[] = [
    {
      label: 'Unified Analysis',
      path: '/',
      icon: <Home />,
      description: 'All models overlay'
    },
    {
      label: 'ML Models Info',
      icon: <Psychology />,
      description: 'All models shown in Unified Analysis',
      children: [
        {
          label: '→ TFT (Temporal Fusion Transformer)',
          path: '/',
          icon: <ShowChart />,
          description: 'Multi-horizon forecasting'
        },
        {
          label: '→ Epidemic Volatility (SIR/SEIR)',
          path: '/',
          icon: <Biotech />,
          description: 'VIX contagion modeling'
        },
        {
          label: '→ GNN (Graph Neural Network)',
          path: '/',
          icon: <AccountBalance />,
          description: 'Stock correlation networks'
        },
        {
          label: '→ Mamba State-Space (O(N))',
          path: '/',
          icon: <Speed />,
          description: 'Linear complexity forecasting'
        },
        {
          label: '→ PINN (Physics-Informed NN)',
          path: '/',
          icon: <Science />,
          description: 'Black-Scholes constraints'
        },
        {
          label: '→ Ensemble (All Models)',
          path: '/',
          icon: <Assessment />,
          description: 'Weighted consensus'
        }
      ]
    },
    {
      label: 'Trading',
      icon: <TrendingUp />,
      children: [
        {
          label: 'Positions',
          path: '/positions',
          description: 'Current holdings'
        },
        {
          label: 'Options Chain',
          path: '/options-chain',
          description: 'Live options data'
        },
        {
          label: 'Paper Trading',
          path: '/paper-trading',
          description: 'Practice trades'
        },
        {
          label: 'Schwab Trading',
          path: '/schwab-trading',
          description: 'Live execution'
        }
      ]
    },
    {
      label: 'Analytics',
      icon: <Analytics />,
      children: [
        {
          label: 'Risk Dashboard',
          path: '/risk-dashboard',
          description: 'Portfolio risk metrics'
        },
        {
          label: 'Options Analytics',
          path: '/options-analytics',
          description: 'IV surface & Greeks'
        },
        {
          label: 'ML Predictions',
          path: '/ml-predictions',
          description: 'Machine learning forecasts'
        },
        {
          label: 'Backtest',
          path: '/backtest',
          description: 'Strategy testing'
        }
      ]
    },
    {
      label: 'Market Data',
      icon: <ShowChart />,
      children: [
        {
          label: 'Real-Time Quotes',
          path: '/market-data',
          description: 'Live market data'
        },
        {
          label: 'Economic Calendar',
          path: '/calendar',
          description: 'Events & earnings'
        },
        {
          label: 'News Feed',
          path: '/news',
          description: 'Financial news'
        },
        {
          label: 'Sentiment Analysis',
          path: '/sentiment',
          description: 'Market mood'
        }
      ]
    },
    {
      label: 'Risk & Testing',
      icon: <Warning />,
      children: [
        {
          label: 'Stress Testing',
          path: '/stress-testing',
          description: 'Portfolio stress tests'
        },
        {
          label: 'Anomaly Detection',
          path: '/anomalies',
          description: 'Unusual patterns'
        },
        {
          label: 'Execution Quality',
          path: '/execution',
          description: 'Fill analysis'
        }
      ]
    }
  ];

  const toggleSection = (label: string) => {
    setExpandedSections(prev =>
      prev.includes(label)
        ? prev.filter(s => s !== label)
        : [...prev, label]
    );
  };

  const renderNavItem = (item: NavItem, level = 0) => {
    const isExpanded = expandedSections.includes(item.label);
    const isActive = location.pathname === item.path;
    const hasChildren = item.children && item.children.length > 0;

    return (
      <div key={item.label}>
        <div
          className={`
            flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer
            transition-colors duration-150 hover:bg-gray-100
            ${isActive ? 'bg-blue-50 text-blue-600' : 'text-gray-700'}
            ${level > 0 ? 'ml-4' : ''}
          `}
          onClick={() => {
            if (hasChildren) {
              toggleSection(item.label);
            } else if (item.path) {
              navigate(item.path);
            }
          }}
        >
          <div className="flex items-center gap-2">
            {item.icon && <span className="text-sm">{item.icon}</span>}
            <div>
              <div className="text-sm font-medium">{item.label}</div>
              {item.description && level > 0 && (
                <div className="text-xs text-gray-500">{item.description}</div>
              )}
            </div>
          </div>
          {hasChildren && (
            <span className="text-gray-400">
              {isExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
            </span>
          )}
        </div>
        {hasChildren && isExpanded && (
          <div className="mt-1">
            {item.children!.map(child => renderNavItem(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-64 h-screen bg-white border-r border-gray-200 overflow-y-auto">
      <div className="p-4">
        <h1 className="text-xl font-bold text-gray-800 mb-1">Options Analysis</h1>
        <p className="text-xs text-gray-500">Neural Network Trading System</p>
      </div>
      <nav className="px-2 pb-4">
        {navigationItems.map(item => renderNavItem(item))}
      </nav>
      <div className="border-t border-gray-200 p-4">
        <div className="text-xs text-gray-500">
          <div>Press Ctrl+K for command palette</div>
          <div className="mt-1">5 Neural Networks Active</div>
        </div>
      </div>
    </div>
  );
};

export default NavigationSidebar;