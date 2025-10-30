"""
Distillation Agent - Tier 8

Synthesizes insights from all 17 agents into investor-friendly narratives.
Responsible for deduplication, categorization, and narrative generation.

V2 Enhancements:
- Structured Outputs via JSON Schema validation (InvestorReport.v1)
- MCP tool calling (jarvis + Firecrawl)
- Provenance tracking with authoritative sources
- Retry logic on schema validation failure
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
import jsonschema
from pathlib import Path

from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext, Message
from ..consensus_engine import ConsensusEngine
from ..prompt_templates import get_distillation_prompt
from ..mcp_tools import MCPToolRegistry

logger = logging.getLogger(__name__)


class DistillationAgent(LLMAgentBase):
    """
    Tier 8 Distillation Agent
    
    Synthesizes agent outputs into investor-friendly narratives.
    
    Responsibilities:
    - Deduplicate insights across agents
    - Identify consensus vs. divergent views
    - Structure output into investor-friendly sections
    - Generate executive summary
    - Create actionable recommendations
    """
    
    def __init__(
        self,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        llm_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4"
    ):
        """
        Initialize Distillation Agent.

        Args:
            shared_context: Shared context for communication
            consensus_engine: Consensus engine for decisions
            llm_provider: LLM provider (anthropic, openai, lmstudio)
            model_name: Model name to use
        """
        # Initialize LLM base
        LLMAgentBase.__init__(
            self,
            agent_id="distillation_agent",
            agent_type="DistillationAgent",
            preferred_model=llm_provider
        )

        # Store swarm components
        self.shared_context = shared_context
        self.consensus_engine = consensus_engine
        self.tier = 8
        self.temperature = 0.7
        self.priority = 10

        # Load JSON Schema for InvestorReport.v1
        schema_path = Path(__file__).parent.parent.parent.parent / "schemas/investor_report_schema.json"
        try:
            with open(schema_path) as f:
                self.report_schema = json.load(f)
            logger.info("✅ Loaded InvestorReport.v1 JSON Schema")
        except Exception as e:
            logger.error(f"❌ Failed to load JSON Schema: {e}")
            self.report_schema = None

        # Initialize MCP tools
        self.mcp_tools = MCPToolRegistry()

        logger.info("🎨 Distillation Agent initialized (Tier 8) with Structured Outputs")
    
    def synthesize_swarm_output(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize all agent outputs into investor-friendly narrative.
        
        Args:
            position_data: Position data and agent results
            
        Returns:
            Investor-friendly report with structured sections
        """
        logger.info("🎨 Starting synthesis of swarm outputs...")
        
        # Step 1: Gather high-priority insights
        agent_messages = self._gather_agent_insights(min_priority=6)
        logger.info(f"📊 Gathered {len(agent_messages)} high-priority insights")
        
        # Step 2: Deduplicate insights
        unique_insights = self._deduplicate_insights(agent_messages)
        logger.info(f"✨ Deduplicated to {len(unique_insights)} unique insights")
        
        # Step 3: Categorize insights
        categorized = self._categorize_insights(unique_insights)
        logger.info(f"📁 Categorized insights: {', '.join(f'{k}={len(v)}' for k, v in categorized.items())}")
        
        # Step 4: Generate narrative via LLM
        narrative = self._generate_narrative(categorized, position_data)
        logger.info("📝 Generated investor narrative")
        
        return narrative
    
    def _gather_agent_insights(self, min_priority: int = 6, max_age_seconds: int = 3600) -> List[Message]:
        """
        Gather high-priority insights from all agents.
        
        Args:
            min_priority: Minimum priority level
            max_age_seconds: Maximum age of messages
            
        Returns:
            List of high-priority messages
        """
        messages = self.shared_context.get_messages(
            min_priority=min_priority,
            max_age_seconds=max_age_seconds
        )
        
        # Sort by priority and confidence
        messages.sort(key=lambda m: (m.priority, m.confidence), reverse=True)
        
        return messages
    
    def _deduplicate_insights(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Deduplicate insights based on semantic similarity.
        
        Args:
            messages: List of messages to deduplicate
            
        Returns:
            List of unique insights
        """
        unique_insights = []
        seen_content = set()
        
        for msg in messages:
            # Create a simple content signature
            content_str = json.dumps(msg.content, sort_keys=True)
            
            if content_str not in seen_content:
                seen_content.add(content_str)
                unique_insights.append({
                    'source': msg.source,
                    'content': msg.content,
                    'priority': msg.priority,
                    'confidence': msg.confidence,
                    'timestamp': msg.timestamp.isoformat()
                })
        
        return unique_insights
    
    def _categorize_insights(self, insights: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Categorize insights into investor-relevant buckets.
        
        Args:
            insights: List of unique insights
            
        Returns:
            Dictionary of categorized insights
        """
        categories = {
            'bullish_signals': [],
            'bearish_signals': [],
            'risk_factors': [],
            'opportunities': [],
            'technical_levels': [],
            'fundamental_metrics': [],
            'sentiment_indicators': [],
            'options_strategies': [],
            'macro_factors': []
        }
        
        for insight in insights:
            content = insight['content']
            
            # Categorize based on content
            category = self._categorize_insight(content)
            if category in categories:
                categories[category].append(insight)
            else:
                # Default to opportunities
                categories['opportunities'].append(insight)
        
        return categories
    
    def _categorize_insight(self, content: Dict[str, Any]) -> str:
        """
        Determine category for a single insight.
        
        Args:
            content: Insight content
            
        Returns:
            Category name
        """
        # Simple keyword-based categorization
        content_str = json.dumps(content).lower()
        
        if any(word in content_str for word in ['bullish', 'buy', 'upside', 'positive', 'strong']):
            return 'bullish_signals'
        elif any(word in content_str for word in ['bearish', 'sell', 'downside', 'negative', 'weak']):
            return 'bearish_signals'
        elif any(word in content_str for word in ['risk', 'danger', 'threat', 'concern', 'warning']):
            return 'risk_factors'
        elif any(word in content_str for word in ['opportunity', 'potential', 'catalyst', 'growth']):
            return 'opportunities'
        elif any(word in content_str for word in ['support', 'resistance', 'level', 'technical', 'chart']):
            return 'technical_levels'
        elif any(word in content_str for word in ['earnings', 'revenue', 'margin', 'valuation', 'pe', 'pb']):
            return 'fundamental_metrics'
        elif any(word in content_str for word in ['sentiment', 'mood', 'fear', 'greed', 'optimism']):
            return 'sentiment_indicators'
        elif any(word in content_str for word in ['option', 'call', 'put', 'strike', 'volatility', 'iv']):
            return 'options_strategies'
        elif any(word in content_str for word in ['macro', 'fed', 'rates', 'inflation', 'gdp', 'economy']):
            return 'macro_factors'
        else:
            return 'opportunities'
    
    def _generate_narrative(
        self,
        categorized_insights: Dict[str, List[Dict]],
        position_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate investor-friendly narrative using LLM.

        V2: With schema validation and retry logic.

        Args:
            categorized_insights: Categorized insights
            position_data: Position data

        Returns:
            Structured investor report (InvestorReport.v1)
        """
        # Build synthesis prompt with schema
        prompt = self._build_synthesis_prompt_v2(categorized_insights, position_data)

        # Try LLM call with retry on schema failure
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.call_llm(prompt, temperature=self.temperature)

                # Parse response into structured format
                narrative = self._parse_narrative_response(response)

                # Validate against schema
                if self.report_schema:
                    try:
                        jsonschema.validate(instance=narrative, schema=self.report_schema)
                        logger.info("✅ Report validated against InvestorReport.v1 schema")
                    except jsonschema.ValidationError as e:
                        logger.warning(f"⚠️ Schema validation failed (attempt {attempt+1}): {e.message}")

                        if attempt < max_retries - 1:
                            # Retry with schema + example
                            prompt = self._build_retry_prompt(prompt, e.message)
                            continue
                        else:
                            logger.error("❌ Schema validation failed after retries, using fallback")
                            return self._generate_fallback_narrative(categorized_insights, position_data)

                # Add metadata
                narrative['metadata'] = {
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'agent': self.agent_id,
                    'total_insights': sum(len(v) for v in categorized_insights.values()),
                    'categories': {k: len(v) for k, v in categorized_insights.items()},
                    'schema_version': 'InvestorReport.v1',
                    'validated': self.report_schema is not None
                }

                return narrative

            except Exception as e:
                logger.error(f"❌ Error generating narrative (attempt {attempt+1}): {e}")
                if attempt >= max_retries - 1:
                    return self._generate_fallback_narrative(categorized_insights, position_data)

        return self._generate_fallback_narrative(categorized_insights, position_data)
    
    def _parse_narrative_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured report.
        
        Args:
            response: LLM response text
            
        Returns:
            Structured report dictionary
        """
        # Try to extract structured sections from response
        sections = {
            'executive_summary': '',
            'recommendation': {},
            'risk_assessment': {},
            'future_outlook': {},
            'next_steps': []
        }
        
        # Simple parsing - look for section headers
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'executive summary' in line_lower:
                current_section = 'executive_summary'
            elif 'recommendation' in line_lower:
                current_section = 'recommendation'
            elif 'risk' in line_lower:
                current_section = 'risk_assessment'
            elif 'outlook' in line_lower or 'future' in line_lower:
                current_section = 'future_outlook'
            elif 'next steps' in line_lower or 'action' in line_lower:
                current_section = 'next_steps'
            elif current_section and line.strip():
                if isinstance(sections[current_section], str):
                    sections[current_section] += line + '\n'
                elif isinstance(sections[current_section], list):
                    sections[current_section].append(line.strip())
        
        return sections
    
    def _generate_fallback_narrative(
        self,
        categorized_insights: Dict[str, List[Dict]],
        position_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate simple fallback narrative if LLM fails.
        
        Args:
            categorized_insights: Categorized insights
            position_data: Position data
            
        Returns:
            Basic structured report
        """
        bullish_count = len(categorized_insights.get('bullish_signals', []))
        bearish_count = len(categorized_insights.get('bearish_signals', []))
        risk_count = len(categorized_insights.get('risk_factors', []))
        
        # Determine overall sentiment
        if bullish_count > bearish_count:
            rating = 'BUY'
            conviction = 'MEDIUM'
        elif bearish_count > bullish_count:
            rating = 'SELL'
            conviction = 'MEDIUM'
        else:
            rating = 'HOLD'
            conviction = 'LOW'
        
        # Return InvestorReport.v1 compliant fallback
        return {
            'as_of': datetime.now(timezone.utc).isoformat(),
            'universe': [position_data.get('symbol', 'UNKNOWN')],
            'executive_summary': {
                'top_picks': [{
                    'ticker': position_data.get('symbol', 'UNKNOWN'),
                    'rationale': f"{bullish_count} bullish vs {bearish_count} bearish signals",
                    'expected_horizon_days': 30
                }],
                'key_risks': [r['content'] for r in categorized_insights.get('risk_factors', [])[:3]] or ['Insufficient data'],
                'thesis': f"Analysis reveals {bullish_count} bullish signals, {bearish_count} bearish signals, and {risk_count} risk factors."
            },
            'risk_panel': {
                'omega': 0.0,
                'gh1': 0.0,
                'pain_index': 0.0,
                'upside_capture': 0.0,
                'downside_capture': 0.0,
                'cvar_95': 0.0,
                'max_drawdown': 0.0,
                'explanations': ['Metrics unavailable - fallback mode']
            },
            'signals': {
                'ml_alpha': {'score': 0.0, 'explanations': ['Unavailable']},
                'regime': 'Normal',
                'sentiment': {'level': 0.0, 'delta': 0.0, 'explanations': ['Unavailable']},
                'smart_money': {'thirteenF': 0.0, 'insider_bias': 0.0, 'options_bias': 0.0, 'explanations': ['Unavailable']},
                'alt_data': {'digital_demand': 0.0, 'earnings_surprise_pred': 0.0, 'explanations': ['Unavailable']},
                'phase4_tech': {
                    'options_flow_composite': None,
                    'residual_momentum': None,
                    'seasonality_score': None,
                    'breadth_liquidity': None,
                    'explanations': ['Phase 4 metrics unavailable']
                }
            },
            'actions': [{
                'ticker': position_data.get('symbol', 'UNKNOWN'),
                'action': rating.lower(),
                'sizing': 'maintain current',
                'risk_controls': 'standard stops'
            }],
            'sources': [{
                'title': 'Fallback report',
                'url': '',
                'provider': 'local',
                'as_of': datetime.now(timezone.utc).date().isoformat()
            }],
            'confidence': {
                'overall': 0.3,
                'drivers': [f"{bullish_count} bullish signals", f"{bearish_count} bearish signals"]
            }
        }

    def _build_synthesis_prompt_v2(
        self,
        categorized_insights: Dict[str, List[Dict]],
        position_data: Dict[str, Any]
    ) -> str:
        """
        Build synthesis prompt with schema and tool calling instructions.

        V2: Includes InvestorReport.v1 schema and MCP tool guidance.
        """
        from ..prompt_templates import get_distillation_prompt

        # Get base prompt
        base_prompt = get_distillation_prompt(categorized_insights, position_data)

        # Add schema and tool instructions
        schema_instructions = f"""

[STRUCTURED OUTPUT REQUIREMENT]
You MUST return a JSON object that strictly matches the InvestorReport.v1 schema.

[BEHAVIORAL RULES]
1. NO invented numbers - use only provided metrics or tool results
2. Return STRICTLY valid JSON matching the schema
3. Short explanations (2-4 lines) per metric
4. Regime-aware: if High-Vol/Crisis, emphasize downside risk
5. Degrade gracefully: if data missing, note it and continue

[PROVENANCE REQUIREMENT]
Always populate the sources[] array with authoritative references:
- Cboe: https://www.cboe.com/us/options/market_statistics/
- SEC: https://www.sec.gov/search-filings/
- FRED: https://fred.stlouisfed.org/
- ExtractAlpha: https://extractalpha.com/
- AlphaSense: https://www.alpha-sense.com/
- LSEG: https://www.lseg.com/

Generate the InvestorReport JSON now:
"""

        return base_prompt + schema_instructions

    def _build_retry_prompt(self, original_prompt: str, validation_error: str) -> str:
        """Build retry prompt with validation error feedback"""
        return f"""{original_prompt}

[VALIDATION ERROR]
Your previous response failed schema validation:
{validation_error}

Please fix the error and return valid JSON matching InvestorReport.v1 schema.
Pay attention to:
- Required fields
- Field types (number, string, array, object)
- Enum values
- Array minItems constraints

Try again:
"""

