"""
Multi-Model Discussion System

Orchestrates discussions between multiple AI models to analyze stocks and options.
Uses GPT-4, Claude Sonnet 4.5, and LM Studio in a 5-round discussion format.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import requests

from .multi_model_config import (
    get_agent_model,
    DISCUSSION_CONFIG,
    ModelProvider
)

logger = logging.getLogger(__name__)


class MultiModelDiscussion:
    """
    Orchestrates multi-round discussions between different AI models.
    
    Each round:
    1. Agents analyze data from their perspective
    2. Share insights with other agents
    3. Build on previous insights
    4. Reach consensus after 5 rounds
    """
    
    def __init__(self):
        self.discussion_history = []
        self.insights = {}
        self.consensus = None
        
    def start_discussion(
        self,
        symbol: str,
        position_data: Dict[str, Any],
        market_data: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        firecrawl_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start a multi-round discussion about a stock/option.
        
        Args:
            symbol: Stock symbol
            position_data: Position information
            market_data: Market data
            sentiment_data: Sentiment analysis
            firecrawl_data: Additional research data from Firecrawl
            
        Returns:
            Discussion results with consensus and insights
        """
        logger.info(f"Starting multi-model discussion for {symbol}")
        
        # Initialize discussion context
        context = {
            'symbol': symbol,
            'position_data': position_data,
            'market_data': market_data,
            'sentiment_data': sentiment_data,
            'firecrawl_data': firecrawl_data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Run discussion rounds
        for round_num in range(1, DISCUSSION_CONFIG['rounds'] + 1):
            logger.info(f"Discussion round {round_num}/{DISCUSSION_CONFIG['rounds']}")
            round_insights = self._run_discussion_round(round_num, context)
            self.discussion_history.append({
                'round': round_num,
                'insights': round_insights,
                'timestamp': datetime.now().isoformat()
            })
        
        # Build consensus
        self.consensus = self._build_consensus()
        
        return {
            'symbol': symbol,
            'discussion_history': self.discussion_history,
            'consensus': self.consensus,
            'final_recommendation': self._generate_recommendation(),
            'confidence_score': self._calculate_confidence(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_discussion_round(
        self,
        round_num: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single discussion round with all agents.
        
        Args:
            round_num: Current round number
            context: Discussion context
            
        Returns:
            Insights from all agents in this round
        """
        round_insights = {}
        
        # Get previous round insights for context
        previous_insights = []
        if round_num > 1:
            previous_insights = [
                h['insights'] for h in self.discussion_history
            ]
        
        # Agent 1: Market Intelligence (GPT-4)
        round_insights['market_intelligence'] = self._get_agent_insight(
            agent_name='market_intelligence',
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus='market conditions, trends, and opportunities'
        )
        
        # Agent 2: Risk Analysis (Claude)
        round_insights['risk_analysis'] = self._get_agent_insight(
            agent_name='risk_analysis',
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus='risk assessment, hedging strategies, and portfolio protection'
        )
        
        # Agent 3: Quantitative Analysis (LM Studio)
        round_insights['quant_analysis'] = self._get_agent_insight(
            agent_name='quant_analysis',
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus='quantitative metrics, probabilities, and statistical analysis'
        )
        
        # Agent 4: Sentiment Research (GPT-4)
        round_insights['sentiment_research'] = self._get_agent_insight(
            agent_name='sentiment_research',
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus='market sentiment, news analysis, and social trends'
        )
        
        # Agent 5: Technical Analysis (Claude)
        round_insights['technical_analysis'] = self._get_agent_insight(
            agent_name='technical_analysis',
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus='technical indicators, chart patterns, and price action'
        )
        
        # Agent 6: Fundamental Analysis (LM Studio)
        round_insights['fundamental_analysis'] = self._get_agent_insight(
            agent_name='fundamental_analysis',
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus='company fundamentals, valuation, and financial health'
        )
        
        return round_insights
    
    def _get_agent_insight(
        self,
        agent_name: str,
        round_num: int,
        context: Dict[str, Any],
        previous_insights: List[Dict[str, Any]],
        focus: str
    ) -> Dict[str, Any]:
        """
        Get insight from a specific agent using its assigned model.
        
        Args:
            agent_name: Name of the agent
            round_num: Current round number
            context: Discussion context
            previous_insights: Insights from previous rounds
            focus: Agent's area of focus
            
        Returns:
            Agent's insight for this round
        """
        model_config = get_agent_model(agent_name)
        
        # Build prompt for the agent
        prompt = self._build_agent_prompt(
            agent_name=agent_name,
            round_num=round_num,
            context=context,
            previous_insights=previous_insights,
            focus=focus
        )
        
        # Call the appropriate model
        try:
            if model_config.provider == ModelProvider.OPENAI:
                response = self._call_openai(model_config, prompt)
            elif model_config.provider == ModelProvider.ANTHROPIC:
                response = self._call_anthropic(model_config, prompt)
            elif model_config.provider == ModelProvider.LMSTUDIO:
                response = self._call_lmstudio(model_config, prompt)
            else:
                response = "Model provider not supported"
            
            return {
                'agent': agent_name,
                'model': model_config.model_name,
                'provider': model_config.provider.value,
                'round': round_num,
                'insight': response,
                'focus': focus,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting insight from {agent_name}: {e}")
            return {
                'agent': agent_name,
                'model': model_config.model_name,
                'provider': model_config.provider.value,
                'round': round_num,
                'insight': f"Error: {str(e)}",
                'focus': focus,
                'timestamp': datetime.now().isoformat()
            }
    
    def _build_agent_prompt(
        self,
        agent_name: str,
        round_num: int,
        context: Dict[str, Any],
        previous_insights: List[Dict[str, Any]],
        focus: str
    ) -> str:
        """Build prompt for an agent"""
        prompt = f"""You are a {agent_name.replace('_', ' ').title()} agent in a multi-agent discussion system.

DISCUSSION ROUND: {round_num}/5

YOUR FOCUS: {focus}

STOCK/OPTION DATA:
Symbol: {context['symbol']}
Position: {json.dumps(context['position_data'], indent=2)}
Market Data: {json.dumps(context['market_data'], indent=2)}
Sentiment: {json.dumps(context['sentiment_data'], indent=2)}

"""
        
        if context.get('firecrawl_data'):
            prompt += f"\nADDITIONAL RESEARCH (Firecrawl):\n{json.dumps(context['firecrawl_data'], indent=2)}\n"
        
        if previous_insights:
            prompt += "\nPREVIOUS ROUND INSIGHTS:\n"
            for round_insights in previous_insights:
                for agent, insight in round_insights.items():
                    prompt += f"\n{agent}: {insight.get('insight', 'N/A')}\n"
        
        prompt += f"""
INSTRUCTIONS:
1. Analyze the data from your perspective ({focus})
2. Consider insights from other agents in previous rounds
3. Provide your unique perspective and analysis
4. Build on or challenge previous insights if appropriate
5. Be specific and actionable

Provide your analysis in 3-5 concise paragraphs.
"""
        
        return prompt
    
    def _call_openai(self, config, prompt: str) -> str:
        """Call OpenAI API"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, returning placeholder")
            return "[GPT-4 Analysis] OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."

        try:
            import time
            api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            # Add optional org/project headers
            if os.getenv('OPENAI_ORG_ID'):
                headers['OpenAI-Organization'] = os.getenv('OPENAI_ORG_ID')
            if os.getenv('OPENAI_PROJECT'):
                headers['OpenAI-Project'] = os.getenv('OPENAI_PROJECT')

            data = {
                'model': config.model_name,
                'messages': [
                    {'role': 'system', 'content': 'You are an expert financial analyst specializing in options and stock trading.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': config.temperature,
                'max_tokens': config.max_tokens
            }

            # Retry logic for rate limits
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f'{api_base}/chat/completions',
                        headers=headers,
                        json=data,
                        timeout=60
                    )
                    response.raise_for_status()

                    result = response.json()
                    return result['choices'][0]['message']['content']

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                            logger.warning(f"OpenAI rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"OpenAI API 429 - Rate limit exceeded after {max_retries} retries")
                            return "[GPT-4 Error] Rate limit exceeded. Please try again later or check your OpenAI quota."
                    elif e.response.status_code == 401:
                        logger.error(f"OpenAI API 401 - Invalid API key")
                        return "[GPT-4 Error] Invalid API key. Please check your OPENAI_API_KEY in .env file."
                    else:
                        raise  # Re-raise for outer exception handler

        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenAI API HTTP error: {e}")
            return f"[GPT-4 Error] HTTP {e.response.status_code}: {str(e)}"
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[GPT-4 Error] {str(e)}"

    def _call_anthropic(self, config, prompt: str) -> str:
        """Call Anthropic API with retry logic"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, skipping Anthropic")
            return "[Claude Analysis] Anthropic API key not configured. Please set ANTHROPIC_API_KEY in .env file to enable Claude analysis."

        try:
            api_base = os.getenv('ANTHROPIC_API_BASE', 'https://api.anthropic.com')
            headers = {
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            }

            data = {
                'model': config.model_name,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'system': 'You are an expert financial analyst specializing in options and stock trading.',
                'temperature': config.temperature,
                'max_tokens': config.max_tokens
            }

            response = requests.post(
                f'{api_base}/v1/messages',
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result['content'][0]['text']

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Anthropic API 404 - Check API key validity and endpoint")
                return "[Claude Error] API endpoint not found. Please verify your ANTHROPIC_API_KEY is valid."
            elif e.response.status_code == 401:
                logger.error(f"Anthropic API 401 - Invalid API key")
                return "[Claude Error] Invalid API key. Please check your ANTHROPIC_API_KEY in .env file."
            elif e.response.status_code == 429:
                logger.error(f"Anthropic API 429 - Rate limit exceeded")
                return "[Claude Error] Rate limit exceeded. Please try again later."
            else:
                logger.error(f"Anthropic API HTTP error: {e}")
                return f"[Claude Error] HTTP {e.response.status_code}: {str(e)}"
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"[Claude Error] {str(e)}"

    def _call_lmstudio(self, config, prompt: str) -> str:
        """Call LM Studio API (OpenAI-compatible)"""
        api_base = os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1')

        try:
            headers = {
                'Content-Type': 'application/json'
            }

            # LM Studio uses OpenAI-compatible API
            model_name = os.getenv('LMSTUDIO_MODEL', config.model_name)

            data = {
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': 'You are an expert financial analyst specializing in options and stock trading.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': config.temperature,
                'max_tokens': config.max_tokens
            }

            response = requests.post(
                f'{api_base}/chat/completions',
                headers=headers,
                json=data,
                timeout=120  # LM Studio can be slower
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.ConnectionError:
            logger.warning("LM Studio not running or not accessible")
            return "[LM Studio] Not running. Please start LM Studio and enable the local server."
        except Exception as e:
            logger.error(f"LM Studio API error: {e}")
            return f"[LM Studio Error] {str(e)}"
    
    def _build_consensus(self) -> Dict[str, Any]:
        """Build consensus from all discussion rounds"""
        # TODO: Implement consensus building logic
        return {
            'recommendation': 'HOLD',
            'confidence': 0.75,
            'key_points': [
                'Market conditions are favorable',
                'Risk is moderate',
                'Sentiment is positive'
            ]
        }
    
    def _generate_recommendation(self) -> str:
        """Generate final recommendation"""
        # TODO: Implement recommendation logic
        return "Based on 5 rounds of multi-model discussion, the consensus is to HOLD the position."
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence score"""
        # TODO: Implement confidence calculation
        return 0.75

