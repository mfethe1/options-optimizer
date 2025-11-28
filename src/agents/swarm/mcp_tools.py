"""
MCP Tools Wrapper for LLM Agents

Exposes jarvis and Firecrawl MCP tools for LLM function calling.
Provides structured interfaces for:
- Portfolio metrics computation
- Options flow analysis
- Data retrieval (price history, options chains)
- Repository operations
- Web research (Firecrawl)

Usage:
    tools = MCPToolRegistry()
    result = tools.compute_portfolio_metrics(positions, benchmark_returns)
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class JarvisMCPTools:
    """Wrapper for jarvis MCP tools"""
    
    @staticmethod
    def compute_portfolio_metrics(
        positions: List[Dict[str, Any]],
        benchmark_returns: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Compute institutional-grade portfolio metrics.
        
        Args:
            positions: List of position dicts with returns data
            benchmark_returns: Benchmark return series (optional)
        
        Returns:
            Dict with all PortfolioMetrics fields
        """
        try:
            from src.analytics.portfolio_metrics import PortfolioAnalytics
            
            # Extract returns from positions
            position_returns = {}
            position_weights = {}
            num_periods = None

            for pos in positions:
                symbol = pos.get('symbol', 'UNKNOWN')
                returns = pos.get('returns', [])
                weight = pos.get('weight', 1.0 / len(positions))

                # Check if returns exist (handle both lists and arrays)
                if returns is not None and len(returns) > 0:
                    returns_array = np.array(returns)
                    position_returns[symbol] = returns_array
                    position_weights[symbol] = weight

                    if num_periods is None:
                        num_periods = len(returns_array)

            if not position_returns:
                logger.warning("No returns data in positions")
                return {'error': 'No returns data available'}

            # Compute weighted portfolio returns
            portfolio_returns_array = np.zeros(num_periods)
            for symbol, returns in position_returns.items():
                weight = position_weights[symbol]
                portfolio_returns_array += returns * weight

            # Convert to pandas Series/DataFrame
            dates = pd.date_range(end=datetime.now(), periods=num_periods, freq='D')
            portfolio_returns_series = pd.Series(portfolio_returns_array, index=dates)

            # Use provided benchmark when available; otherwise fall back to portfolio returns.
            if benchmark_returns is None:
                logger.warning(
                    "No benchmark_returns provided to compute_portfolio_metrics; "
                    "falling back to portfolio returns as benchmark proxy."
                )
                benchmark_returns_array = portfolio_returns_array.copy()
            else:
                benchmark_returns_array = np.array(benchmark_returns)
                # Align length with portfolio series if needed (deterministic, no randomness)
                if len(benchmark_returns_array) != num_periods:
                    if len(benchmark_returns_array) > num_periods:
                        benchmark_returns_array = benchmark_returns_array[-num_periods:]
                    elif len(benchmark_returns_array) == 0:
                        benchmark_returns_array = portfolio_returns_array.copy()
                    else:
                        last_val = benchmark_returns_array[-1]
                        pad = np.full(num_periods - len(benchmark_returns_array), last_val)
                        benchmark_returns_array = np.concatenate([benchmark_returns_array, pad])

            benchmark_returns_series = pd.Series(benchmark_returns_array, index=dates)

            # Convert position_returns dict to DataFrame
            position_returns_df = pd.DataFrame(position_returns, index=dates)
            position_weights_series = pd.Series(position_weights)

            # Calculate metrics
            analytics = PortfolioAnalytics()
            metrics = analytics.calculate_all_metrics(
                portfolio_returns=portfolio_returns_series,
                benchmark_returns=benchmark_returns_series,
                position_returns=position_returns_df,
                position_weights=position_weights_series
            )
            
            # Convert to dict
            return {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'omega_ratio': metrics.omega_ratio,
                'gh1_ratio': metrics.gh1_ratio,
                'pain_index': metrics.pain_index,
                'max_drawdown': metrics.max_drawdown,
                'cvar_95': metrics.cvar_95,
                'upside_capture': metrics.upside_capture,
                'downside_capture': metrics.downside_capture,
                'alpha': metrics.alpha,
                'beta': metrics.beta,
                'win_rate': metrics.win_rate,
                'as_of': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error computing portfolio metrics: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def compute_options_flow(
        symbol: str,
        pcr: float,
        iv_skew: float,
        volume_ratio: float
    ) -> Dict[str, Any]:
        """
        Compute options flow composite score.
        
        Args:
            symbol: Stock symbol
            pcr: Put/Call ratio
            iv_skew: IV skew (25-delta put - call)
            volume_ratio: Volume / 20-day avg
        
        Returns:
            Dict with composite score and interpretation
        """
        try:
            from src.analytics.technical_cross_asset import compute_options_flow_metrics
            
            metrics = compute_options_flow_metrics(pcr, iv_skew, volume_ratio)
            
            return {
                'symbol': symbol,
                'pcr': metrics.pcr,
                'iv_skew': metrics.iv_skew,
                'volume_ratio': metrics.volume_ratio,
                'composite_score': metrics.composite_score,
                'interpretation': metrics.interpretation,
                'as_of': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error computing options flow: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def compute_phase4_metrics(
        pcr: Optional[float] = None,
        iv_skew: Optional[float] = None,
        volume_ratio: Optional[float] = None,
        asset_returns: Optional[List[float]] = None,
        market_returns: Optional[List[float]] = None,
        sector_returns: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Compute all Phase 4 technical metrics.
        
        Returns:
            Dict with options_flow_composite, residual_momentum, seasonality_score, breadth_liquidity
        """
        try:
            from src.analytics.technical_cross_asset import compute_phase4_metrics
            
            # Convert lists to numpy arrays
            asset_ret = np.array(asset_returns) if asset_returns else None
            market_ret = np.array(market_returns) if market_returns else None
            sector_ret = np.array(sector_returns) if sector_returns else None
            
            # Create returns series for seasonality
            returns_series = None
            if asset_returns:
                dates = pd.date_range(end=datetime.now(), periods=len(asset_returns), freq='D')
                returns_series = pd.Series(asset_returns, index=dates)
            
            metrics = compute_phase4_metrics(
                pcr=pcr,
                iv_skew=iv_skew,
                volume_ratio=volume_ratio,
                asset_returns=asset_ret,
                market_returns=market_ret,
                sector_returns=sector_ret,
                returns_series=returns_series
            )
            
            return {
                **metrics,
                'as_of': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error computing Phase 4 metrics: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def get_price_history(symbol: str, days: int = 252) -> Dict[str, Any]:
        """
        Get price history for a symbol using yfinance.
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
        
        Returns:
            Dict with price data
        """
        try:
            import yfinance as yf
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 1.5)  # Buffer for weekends/holidays
            
            logger.info(f"Fetching {days} days of price history for {symbol} from yfinance")
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            history = ticker.history(start=start_date, end=end_date)
            
            if history.empty:
                logger.warning(f"No price history found for {symbol}")
                return {
                    'symbol': symbol,
                    'days': days,
                    'data': [],
                    'error': 'No data found'
                }
            
            # Extract closing prices
            # Take last N days
            history = history.tail(days)
            
            prices = history['Close'].tolist()
            dates = [d.strftime('%Y-%m-%d') for d in history.index]
            
            # Calculate daily returns
            returns = history['Close'].pct_change().dropna().tolist()
            
            return {
                'symbol': symbol,
                'days': len(prices),
                'prices': prices,
                'dates': dates,
                'returns': returns,
                'current_price': prices[-1] if prices else None,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            return {
                'symbol': symbol,
                'days': days,
                'data': [],
                'error': str(e)
            }

    @staticmethod
    def get_options_flow_metrics(symbol: str) -> Dict[str, Any]:
        """
        Get real options flow metrics from DataStore.
        
        Computes:
        - PCR (Put/Call Volume Ratio)
        - IV Skew (25-delta Put IV - 25-delta Call IV)
        - Volume Ratio (Total Volume / 1.0 - placeholder)
        """
        try:
            from src.data.datastore import DataStore
            import pandas as pd
            
            ds = DataStore()
            # Try to find latest snapshot
            # Since we don't know the date, we might need to list files or try today/yesterday
            # For now, let's try to list the directory if possible, or just try today
            
            # Hack: DataStore doesn't expose list_files. 
            # We'll try to find the latest file in the directory manually
            import os
            base_dir = os.path.join("data", "opt", "chains", symbol.upper())
            if not os.path.exists(base_dir):
                return {'error': 'No options data found'}
                
            files = sorted([f for f in os.listdir(base_dir) if f.endswith('.parquet')])
            if not files:
                return {'error': 'No options snapshots found'}
                
            latest_file = files[-1]
            # Parse date from filename (YYYY-MM-DD.parquet)
            as_of_str = latest_file.replace('.parquet', '')
            as_of = datetime.strptime(as_of_str, '%Y-%m-%d').date()
            
            chain = ds.read_options_chain_snapshot(symbol, as_of)
            if not chain:
                return {'error': 'Empty options chain'}
                
            df = pd.DataFrame(chain)
            
            # Compute PCR
            if 'volume' in df.columns:
                total_put_vol = df[df['option_type'] == 'put']['volume'].sum()
                total_call_vol = df[df['option_type'] == 'call']['volume'].sum()
                pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0
            else:
                pcr = 0.7  # Neutral default
            
            # Compute IV Skew (25 delta)
            iv_skew = 0.0
            if 'delta' in df.columns and 'implied_volatility' in df.columns:
                # Filter for expiration > 30 days
                # This is a simplified calculation
                puts = df[df['option_type'] == 'put']
                calls = df[df['option_type'] == 'call']
                
                # Find closest to 0.25 delta (absolute)
                # Note: put delta is negative
                if not puts.empty and not calls.empty:
                    put_25 = puts.iloc[(puts['delta'].abs() - 0.25).abs().argsort()[:1]]
                    call_25 = calls.iloc[(calls['delta'].abs() - 0.25).abs().argsort()[:1]]
                    
                    if not put_25.empty and not call_25.empty:
                        put_iv = put_25.iloc[0]['implied_volatility']
                        call_iv = call_25.iloc[0]['implied_volatility']
                        if put_iv and call_iv:
                            iv_skew = put_iv - call_iv
            else:
                logger.warning(f"No delta/IV data for {symbol}, skipping IV skew")
            
            return {
                'pcr': float(pcr),
                'iv_skew': float(iv_skew),
                'volume_ratio': 1.2, # Placeholder as we don't have avg volume easily
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error computing options flow metrics for {symbol}: {e}")
            return {'error': str(e)}


class FirecrawlMCPTools:
    """Wrapper for Firecrawl MCP tools"""

    @staticmethod
    def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search web using Firecrawl MCP.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            Search results with URLs and content
        """
        try:
            # Import Firecrawl MCP tool
            try:
                from firecrawl_search_firecrawl_mcp import firecrawl_search
                firecrawl_available = True
            except ImportError:
                logger.warning("Firecrawl MCP not available, using fallback")
                firecrawl_available = False

            logger.info(f"🔍 Firecrawl search: {query}")

            if firecrawl_available:
                # Call real Firecrawl MCP
                results = firecrawl_search(
                    query=query,
                    limit=max_results,
                    scrapeOptions={
                        "formats": ["markdown"],
                        "onlyMainContent": True
                    }
                )

                return {
                    'query': query,
                    'results': results.get('data', []) if isinstance(results, dict) else results,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'firecrawl',
                    'success': True
                }
            else:
                # Fallback: return empty results
                return {
                    'query': query,
                    'results': [],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'firecrawl_fallback',
                    'success': False,
                    'error': 'Firecrawl MCP not available'
                }

        except Exception as e:
            logger.error(f"❌ Firecrawl search error: {e}")
            return {
                'query': query,
                'results': [],
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'firecrawl',
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def scrape_url(url: str) -> Dict[str, Any]:
        """
        Scrape a specific URL using Firecrawl MCP.

        Args:
            url: URL to scrape

        Returns:
            Scraped content in markdown format
        """
        try:
            # Import Firecrawl MCP tool
            try:
                from firecrawl_scrape_firecrawl_mcp import firecrawl_scrape
                firecrawl_available = True
            except ImportError:
                logger.warning("Firecrawl MCP not available, using fallback")
                firecrawl_available = False

            logger.info(f"📄 Firecrawl scrape: {url}")

            if firecrawl_available:
                # Call real Firecrawl MCP
                result = firecrawl_scrape(
                    url=url,
                    formats=["markdown"],
                    onlyMainContent=True
                )

                return {
                    'url': url,
                    'content': result.get('markdown', '') if isinstance(result, dict) else str(result),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'firecrawl',
                    'success': True
                }
            else:
                # Fallback: return empty content
                return {
                    'url': url,
                    'content': '',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'firecrawl_fallback',
                    'success': False,
                    'error': 'Firecrawl MCP not available'
                }

        except Exception as e:
            logger.error(f"❌ Firecrawl scrape error: {e}")
            return {
                'url': url,
                'content': '',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'firecrawl',
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def fetch_provider_fact_sheet(provider: str, signal: str) -> Dict[str, Any]:
        """
        Fetch authoritative fact sheets from data providers.

        Args:
            provider: Provider name (extractalpha, cboe, sec, fred, alphasense, lseg)
            signal: Signal type (cam, pcr, 13f, macro, sentiment)

        Returns:
            Fact sheet content and URL
        """
        queries = {
            'extractalpha_cam': 'site:extractalpha.com "Cross-Asset Model" fact sheet',
            'cboe_pcr': 'site:cboe.com "put/call ratio" historical data',
            'sec_13f': 'site:sec.gov "Form 13F Data Sets"',
            'fred_macro': 'site:fred.stlouisfed.org "series_observations"',
            'alphasense_sentiment': 'site:alphasense.com sentiment transcripts API',
            'lseg_marketpsych': 'site:lseg.com MarketPsych Analytics fact sheet'
        }

        query_key = f"{provider}_{signal}".lower()
        query = queries.get(query_key, f"site:{provider}.com {signal}")

        # Search for fact sheets
        search_results = FirecrawlMCPTools.search_web(query, max_results=3)

        # If search successful and has results, scrape the first URL
        if search_results.get('success') and search_results.get('results'):
            first_result = search_results['results'][0]
            if isinstance(first_result, dict) and 'url' in first_result:
                url = first_result['url']
                scrape_result = FirecrawlMCPTools.scrape_url(url)

                return {
                    'provider': provider,
                    'signal': signal,
                    'query': query,
                    'url': url,
                    'content': scrape_result.get('content', ''),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': 'firecrawl',
                    'success': scrape_result.get('success', False)
                }

        # Fallback: return search results only
        return {
            'provider': provider,
            'signal': signal,
            'query': query,
            'search_results': search_results,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'firecrawl',
            'success': search_results.get('success', False)
        }


class MCPToolRegistry:
    """
    Central registry for all MCP tools.
    Provides unified interface for LLM agents.
    """
    
    def __init__(self):
        self.jarvis = JarvisMCPTools()
        self.firecrawl = FirecrawlMCPTools()
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions for function calling.
        
        Returns:
            List of tool definition dicts
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'compute_portfolio_metrics',
                    'description': 'Compute institutional-grade portfolio metrics (Sharpe, Omega, GH1, Pain, CVaR, etc.)',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'positions': {
                                'type': 'array',
                                'description': 'List of positions with returns data'
                            },
                            'benchmark_returns': {
                                'type': 'array',
                                'description': 'Benchmark return series (optional)'
                            }
                        },
                        'required': ['positions']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'compute_options_flow',
                    'description': 'Compute options flow composite score (PCR + IV skew + volume)',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'symbol': {'type': 'string'},
                            'pcr': {'type': 'number', 'description': 'Put/Call ratio'},
                            'iv_skew': {'type': 'number', 'description': '25-delta put IV - call IV'},
                            'volume_ratio': {'type': 'number', 'description': 'Volume / 20-day avg'}
                        },
                        'required': ['symbol', 'pcr', 'iv_skew', 'volume_ratio']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'search_web',
                    'description': 'Search web using Firecrawl for authoritative sources',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'max_results': {'type': 'integer', 'default': 5}
                        },
                        'required': ['query']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'scrape_url',
                    'description': 'Scrape a specific URL using Firecrawl to extract content',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'url': {'type': 'string', 'description': 'URL to scrape'}
                        },
                        'required': ['url']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'fetch_provider_fact_sheet',
                    'description': 'Fetch fact sheets from data providers (ExtractAlpha, Cboe, SEC, FRED, AlphaSense, LSEG)',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'provider': {'type': 'string', 'enum': ['extractalpha', 'cboe', 'sec', 'fred', 'alphasense', 'lseg']},
                            'signal': {'type': 'string'}
                        },
                        'required': ['provider', 'signal']
                    }
                }
            }
        ]
    
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool by name with arguments.

        Args:
            tool_name: Name of tool to call
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if tool_name == 'compute_portfolio_metrics':
            return self.jarvis.compute_portfolio_metrics(**kwargs)
        elif tool_name == 'compute_options_flow':
            return self.jarvis.compute_options_flow(**kwargs)
        elif tool_name == 'compute_phase4_metrics':
            return self.jarvis.compute_phase4_metrics(**kwargs)
        elif tool_name == 'search_web':
            return self.firecrawl.search_web(**kwargs)
        elif tool_name == 'scrape_url':
            return self.firecrawl.scrape_url(**kwargs)
        elif tool_name == 'fetch_provider_fact_sheet':
            return self.firecrawl.fetch_provider_fact_sheet(**kwargs)
        else:
            return {'error': f'Unknown tool: {tool_name}'}

