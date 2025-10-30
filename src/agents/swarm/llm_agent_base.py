"""
LLM-Powered Agent Base Class

This base class enables swarm agents to make actual LLM calls to OpenAI, Anthropic, or LMStudio
for intelligent analysis instead of using hardcoded logic.
"""

import logging
import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMAgentBase:
    """
    Base class for LLM-powered agents.
    
    Provides methods to call OpenAI, Anthropic, and LMStudio APIs.
    """
    
    def __init__(self, agent_id: str, agent_type: str, preferred_model: str = "openai"):
        """
        Initialize LLM agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (MarketAnalyst, RiskManager, etc.)
            preferred_model: Preferred LLM provider (openai, anthropic, lmstudio)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.preferred_model = preferred_model

        # API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.lmstudio_base_url = os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1')

        # Check LMStudio availability on init
        self._lmstudio_available = self._check_lmstudio_available()

        # Auto-fallback if preferred model is lmstudio but not available
        if self.preferred_model == "lmstudio" and not self._lmstudio_available:
            if self.anthropic_api_key:
                logger.warning(f"{agent_id}: LMStudio not available, auto-switching to Anthropic")
                self.preferred_model = "anthropic"
            elif self.openai_api_key:
                logger.warning(f"{agent_id}: LMStudio not available, auto-switching to OpenAI")
                self.preferred_model = "openai"
            else:
                logger.warning(f"{agent_id}: LMStudio not available and no API keys found")

        logger.info(f"Initialized LLM-powered {agent_type} (ID: {agent_id}, Model: {self.preferred_model})")

    def _check_lmstudio_available(self) -> bool:
        """Check if LMStudio is available by making a quick health check"""
        try:
            import requests
            response = requests.get(
                f'{self.lmstudio_base_url.replace("/v1", "")}/health',
                timeout=2
            )
            return response.status_code == 200
        except:
            # Try alternative health check
            try:
                response = requests.get(
                    f'{self.lmstudio_base_url}/models',
                    timeout=2
                )
                return response.status_code in [200, 404]  # 404 is ok, means server is up
            except:
                return False

    def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Call the preferred LLM with fallback to other providers.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        # Try preferred model first
        if self.preferred_model == "openai" and self.openai_api_key:
            response = self._call_openai(prompt, system_prompt, temperature, max_tokens)
            if response:
                return response
        
        elif self.preferred_model == "anthropic" and self.anthropic_api_key:
            response = self._call_anthropic(prompt, system_prompt, temperature, max_tokens)
            if response:
                return response
        
        elif self.preferred_model == "lmstudio":
            response = self._call_lmstudio(prompt, system_prompt, temperature, max_tokens)
            if response:
                return response
        
        # Fallback chain: Anthropic -> OpenAI (skip LMStudio if not available)
        logger.warning(f"{self.agent_id}: Preferred model failed, trying fallbacks")

        # Try Anthropic first (generally no rate limits)
        if self.anthropic_api_key:
            response = self._call_anthropic(prompt, system_prompt, temperature, max_tokens)
            if response:
                logger.info(f"{self.agent_id}: Fallback to Anthropic successful")
                return response

        # Try OpenAI (may have rate limits)
        if self.openai_api_key:
            response = self._call_openai(prompt, system_prompt, temperature, max_tokens)
            if response:
                logger.info(f"{self.agent_id}: Fallback to OpenAI successful")
                return response

        # Try LMStudio only if available
        if self._lmstudio_available:
            response = self._call_lmstudio(prompt, system_prompt, temperature, max_tokens)
            if response:
                logger.info(f"{self.agent_id}: Fallback to LMStudio successful")
                return response

        # All failed - return graceful degradation
        logger.error(f"{self.agent_id}: All LLM providers failed")
        return self._generate_fallback_response(prompt)
    
    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call OpenAI API"""
        if not self.openai_api_key:
            return None
        
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            data = {
                'model': 'gpt-4',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"{self.agent_id}: OpenAI rate limit exceeded (429). Skipping OpenAI fallback.")
            else:
                logger.error(f"{self.agent_id}: OpenAI API error: {e}")
            return None
        except Exception as e:
            logger.error(f"{self.agent_id}: OpenAI API error: {e}")
            return None
    
    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call Anthropic API"""
        if not self.anthropic_api_key:
            return None
        
        try:
            headers = {
                'x-api-key': self.anthropic_api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'claude-3-5-sonnet-20241022',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            if system_prompt:
                data['system'] = system_prompt
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Anthropic API error: {e}")
            return None
    
    def _call_lmstudio(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> Optional[str]:
        """Call LMStudio API (OpenAI-compatible)"""
        try:
            headers = {'Content-Type': 'application/json'}
            
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            data = {
                'model': 'local-model',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            response = requests.post(
                f'{self.lmstudio_base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"{self.agent_id}: LMStudio API error: {e}")
            return None
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent type.
        Override in subclasses for specialized prompts.
        """
        return f"You are an expert {self.agent_type} specializing in options and stock trading analysis."


class FirecrawlMixin:
    """
    Mixin class to add Firecrawl capabilities to agents.
    """
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web using Firecrawl MCP.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        # TODO: Integrate with actual Firecrawl MCP
        # For now, return placeholder
        logger.info(f"Firecrawl search: {query}")
        
        return {
            'query': query,
            'results': [],
            'summary': f"Web search for: {query}",
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'firecrawl_placeholder'
        }
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape a specific URL using Firecrawl.
        
        Args:
            url: URL to scrape
            
        Returns:
            Scraped content
        """
        # TODO: Integrate with actual Firecrawl MCP
        logger.info(f"Firecrawl scrape: {url}")
        
        return {
            'url': url,
            'content': '',
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'firecrawl_placeholder'
        }
    
    def get_news(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get news for a symbol using Firecrawl.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            News articles
        """
        query = f"{symbol} stock news last {days} days"
        return self.search_web(query, max_results=10)
    
    def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get social media sentiment using Firecrawl.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Social sentiment data
        """
        queries = [
            f"${symbol} site:twitter.com",
            f"{symbol} site:reddit.com/r/wallstreetbets",
            f"{symbol} site:reddit.com/r/stocks"
        ]
        
        results = []
        for query in queries:
            results.append(self.search_web(query, max_results=5))
        
        return {
            'symbol': symbol,
            'platforms': results,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a graceful fallback response when all LLM providers fail.
        Returns a basic analysis based on the agent type.
        """
        fallback_responses = {
            "MarketAnalyst": "Market analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
            "RiskManager": "Risk analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
            "FundamentalAnalyst": "Fundamental analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
            "MacroEconomist": "Macro analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
            "SentimentAnalyst": "Sentiment analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
            "VolatilitySpecialist": "Volatility analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
            "RecommendationAgent": "Recommendations unavailable. All LLM providers failed. Please check API keys and connectivity."
        }

        return fallback_responses.get(
            self.agent_type,
            f"[{self.agent_type}] Analysis unavailable. All LLM providers failed."
        )

