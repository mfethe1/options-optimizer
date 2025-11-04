/**
 * Charts Demo Page
 *
 * Comprehensive demonstration of TradingView Lightweight Charts integration
 * Showcases all features: candlestick charts, indicators, multi-timeframe, themes
 *
 * This page serves as:
 * 1. Documentation by example
 * 2. Visual testing ground
 * 3. Feature showcase for stakeholders
 */

import React, { useState, useMemo } from 'react';
import {
  CandlestickChart,
  MultiTimeframeChart,
  TradingViewChart,
  generateSampleData,
  TRADING_PRESETS,
  OHLCVData,
  IndicatorConfig,
} from '../components/charts';

const ChartsDemo: React.FC = () => {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [activeDemo, setActiveDemo] = useState<'single' | 'multi' | 'indicators'>('single');

  /**
   * Generate sample data for different timeframes
   */
  const sampleData = useMemo(() => {
    return {
      '1m': generateSampleData(500, 150), // 500 minutes
      '5m': generateSampleData(300, 150), // 1500 minutes
      '15m': generateSampleData(200, 150), // 3000 minutes
      '1h': generateSampleData(168, 150), // 1 week
      '4h': generateSampleData(180, 150), // 30 days
      '1d': generateSampleData(365, 150), // 1 year
      '1w': generateSampleData(104, 150), // 2 years
      '1M': generateSampleData(60, 150), // 5 years
    };
  }, []);

  /**
   * Popular technical indicators
   */
  const popularIndicators: IndicatorConfig[] = [
    { type: 'sma', period: 20, label: 'SMA 20', color: '#2196f3' },
    { type: 'sma', period: 50, label: 'SMA 50', color: '#ff9800' },
    { type: 'ema', period: 12, label: 'EMA 12', color: '#4caf50' },
  ];

  /**
   * Theme toggle button style
   */
  const containerStyle: React.CSSProperties = {
    minHeight: '100vh',
    backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
    color: theme === 'dark' ? '#d1d4dc' : '#191919',
    padding: '20px',
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  };

  const buttonStyle = (isActive: boolean): React.CSSProperties => ({
    padding: '10px 20px',
    backgroundColor: isActive ? '#2962ff' : theme === 'dark' ? '#1e222d' : '#f0f3fa',
    color: isActive ? '#ffffff' : theme === 'dark' ? '#d1d4dc' : '#191919',
    border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.2s',
  });

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div
        style={{
          maxWidth: '1600px',
          margin: '0 auto',
          marginBottom: '24px',
        }}
      >
        <h1 style={{ fontSize: '32px', fontWeight: 600, marginBottom: '8px' }}>
          📊 TradingView Lightweight Charts Demo
        </h1>
        <p style={{ fontSize: '16px', opacity: 0.7, marginBottom: '24px' }}>
          Professional-grade charting system • Bloomberg Terminal quality • 100K+ data points at 60 FPS
        </p>

        {/* Control Bar */}
        <div
          style={{
            display: 'flex',
            gap: '16px',
            alignItems: 'center',
            flexWrap: 'wrap',
            padding: '16px',
            backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
            borderRadius: '8px',
            marginBottom: '24px',
          }}
        >
          {/* Theme Toggle */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '14px', fontWeight: 500 }}>Theme:</span>
            <button onClick={() => setTheme('dark')} style={buttonStyle(theme === 'dark')}>
              🌙 Dark
            </button>
            <button onClick={() => setTheme('light')} style={buttonStyle(theme === 'light')}>
              ☀️ Light
            </button>
          </div>

          {/* Demo Mode Selector */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginLeft: 'auto' }}>
            <span style={{ fontSize: '14px', fontWeight: 500 }}>Demo:</span>
            <button onClick={() => setActiveDemo('single')} style={buttonStyle(activeDemo === 'single')}>
              Single Chart
            </button>
            <button onClick={() => setActiveDemo('multi')} style={buttonStyle(activeDemo === 'multi')}>
              Multi-Timeframe
            </button>
            <button onClick={() => setActiveDemo('indicators')} style={buttonStyle(activeDemo === 'indicators')}>
              Indicators
            </button>
          </div>

          {/* Symbol Selector */}
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '14px', fontWeight: 500 }}>Symbol:</span>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              style={{
                padding: '8px 12px',
                backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
                color: theme === 'dark' ? '#d1d4dc' : '#191919',
                border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
                borderRadius: '6px',
                fontSize: '14px',
                cursor: 'pointer',
              }}
            >
              <option value="AAPL">AAPL - Apple Inc.</option>
              <option value="MSFT">MSFT - Microsoft</option>
              <option value="GOOGL">GOOGL - Alphabet</option>
              <option value="TSLA">TSLA - Tesla</option>
              <option value="NVDA">NVDA - NVIDIA</option>
            </select>
          </div>
        </div>

        {/* Feature Highlights */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '16px',
            marginBottom: '24px',
          }}
        >
          {[
            { icon: '⚡', title: 'High Performance', desc: '100K+ data points at 60 FPS' },
            { icon: '📈', title: 'Full Indicators', desc: 'SMA, EMA, RSI, MACD, Bollinger, ATR' },
            { icon: '🎨', title: 'Bloomberg Theme', desc: 'Professional dark/light themes' },
            { icon: '🔄', title: 'Real-time Updates', desc: 'WebSocket-ready streaming' },
            { icon: '📊', title: 'Multi-Timeframe', desc: '2x2, 3x3 grid layouts' },
            { icon: '💾', title: 'Tiny Bundle', desc: 'Only 45KB gzipped' },
          ].map((feature, idx) => (
            <div
              key={idx}
              style={{
                padding: '16px',
                backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
                borderRadius: '8px',
                border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
              }}
            >
              <div style={{ fontSize: '24px', marginBottom: '8px' }}>{feature.icon}</div>
              <div style={{ fontSize: '16px', fontWeight: 600, marginBottom: '4px' }}>{feature.title}</div>
              <div style={{ fontSize: '13px', opacity: 0.7 }}>{feature.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Chart Display Area */}
      <div
        style={{
          maxWidth: '1600px',
          margin: '0 auto',
        }}
      >
        {/* Single Chart Demo */}
        {activeDemo === 'single' && (
          <div>
            <h2 style={{ fontSize: '24px', fontWeight: 600, marginBottom: '16px' }}>
              Full-Featured Candlestick Chart
            </h2>
            <p style={{ fontSize: '14px', opacity: 0.7, marginBottom: '16px' }}>
              Complete trading interface with price stats, volume histogram, crosshair tooltips, and interval selector
            </p>
            <CandlestickChart
              symbol={selectedSymbol}
              data={sampleData['1d']}
              interval="1d"
              theme={theme}
              showVolume={true}
              showControls={true}
            />
          </div>
        )}

        {/* Multi-Timeframe Demo */}
        {activeDemo === 'multi' && (
          <div>
            <h2 style={{ fontSize: '24px', fontWeight: 600, marginBottom: '16px' }}>
              Multi-Timeframe Analysis Grid
            </h2>
            <p style={{ fontSize: '14px', opacity: 0.7, marginBottom: '16px' }}>
              Bloomberg Terminal-style layout with synchronized crosshairs across multiple timeframes
            </p>
            <MultiTimeframeChart
              symbol={selectedSymbol}
              data={sampleData}
              layout="2x2"
              theme={theme}
              showVolume={true}
            />

            {/* Trading Presets */}
            <div style={{ marginTop: '24px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '12px' }}>
                Trading Style Presets
              </h3>
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {Object.entries(TRADING_PRESETS).map(([name, preset]) => (
                  <div
                    key={name}
                    style={{
                      padding: '12px 16px',
                      backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
                      borderRadius: '6px',
                      border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
                    }}
                  >
                    <div style={{ fontSize: '14px', fontWeight: 600, textTransform: 'capitalize' }}>
                      {name.replace(/([A-Z])/g, ' $1').trim()}
                    </div>
                    <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '4px' }}>
                      {preset.timeframes.map((tf) => tf.label).join(', ')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Indicators Demo */}
        {activeDemo === 'indicators' && (
          <div>
            <h2 style={{ fontSize: '24px', fontWeight: 600, marginBottom: '16px' }}>
              Technical Indicators
            </h2>
            <p style={{ fontSize: '14px', opacity: 0.7, marginBottom: '16px' }}>
              Professional technical analysis with SMA, EMA, Bollinger Bands, RSI, MACD, and more
            </p>
            <CandlestickChart
              symbol={selectedSymbol}
              data={sampleData['1d']}
              interval="1d"
              theme={theme}
              showVolume={true}
              showControls={true}
              indicators={popularIndicators}
            />

            {/* Available Indicators List */}
            <div style={{ marginTop: '24px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '12px' }}>
                Available Technical Indicators
              </h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '12px',
                }}
              >
                {[
                  { name: 'SMA', desc: 'Simple Moving Average' },
                  { name: 'EMA', desc: 'Exponential Moving Average' },
                  { name: 'Bollinger Bands', desc: 'Volatility indicator' },
                  { name: 'RSI', desc: 'Relative Strength Index' },
                  { name: 'MACD', desc: 'Moving Average Convergence Divergence' },
                  { name: 'ATR', desc: 'Average True Range' },
                  { name: 'Stochastic', desc: 'Momentum oscillator' },
                  { name: 'VWAP', desc: 'Volume Weighted Average Price' },
                  { name: 'OBV', desc: 'On-Balance Volume' },
                  { name: 'Parabolic SAR', desc: 'Stop and Reverse' },
                ].map((indicator, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '12px',
                      backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
                      borderRadius: '6px',
                      border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
                    }}
                  >
                    <div style={{ fontSize: '14px', fontWeight: 600, color: '#2962ff' }}>
                      {indicator.name}
                    </div>
                    <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '4px' }}>
                      {indicator.desc}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Performance Stats */}
        <div
          style={{
            marginTop: '32px',
            padding: '24px',
            backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
            borderRadius: '8px',
            border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
          }}
        >
          <h3 style={{ fontSize: '20px', fontWeight: 600, marginBottom: '16px' }}>
            📊 Performance Metrics
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '28px', fontWeight: 600, color: '#4caf50' }}>45KB</div>
              <div style={{ fontSize: '13px', opacity: 0.7 }}>Bundle size (gzipped)</div>
            </div>
            <div>
              <div style={{ fontSize: '28px', fontWeight: 600, color: '#2196f3' }}>100K+</div>
              <div style={{ fontSize: '13px', opacity: 0.7 }}>Data points supported</div>
            </div>
            <div>
              <div style={{ fontSize: '28px', fontWeight: 600, color: '#ff9800' }}>60 FPS</div>
              <div style={{ fontSize: '13px', opacity: 0.7 }}>Smooth rendering</div>
            </div>
            <div>
              <div style={{ fontSize: '28px', fontWeight: 600, color: '#9c27b0' }}>FREE</div>
              <div style={{ fontSize: '13px', opacity: 0.7 }}>Apache 2.0 license</div>
            </div>
          </div>
        </div>

        {/* API Integration Note */}
        <div
          style={{
            marginTop: '16px',
            padding: '16px',
            backgroundColor: theme === 'dark' ? 'rgba(41, 98, 255, 0.1)' : 'rgba(41, 98, 255, 0.05)',
            borderLeft: '4px solid #2962ff',
            borderRadius: '4px',
          }}
        >
          <div style={{ fontSize: '14px', fontWeight: 600, marginBottom: '8px' }}>
            🔌 Ready for API Integration
          </div>
          <div style={{ fontSize: '13px', opacity: 0.8 }}>
            This charting system is ready to connect to your market data APIs. Simply replace the sample data with
            real-time feeds from Alpha Vantage, Polygon.io, or your backend endpoints. WebSocket streaming is fully
            supported for live updates.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartsDemo;
