# Distillation Agent & Investor-Friendly Output Implementation Plan

## Executive Summary

This plan implements a comprehensive distillation and coordination system for the 17-agent swarm that:
1. **Prevents redundant outputs** through context engineering and deduplication
2. **Ensures diverse perspectives** via temperature variation and role-specific prompts
3. **Creates investor-friendly narratives** instead of technical JSON outputs
4. **Structures recommendations** into digestible sections (Buy/Sell/Hold, Risk, Outlook)
5. **Synthesizes agent outputs** into cohesive stories via a Distillation Agent

---

## Research-Based Design Principles

### From Multi-Agent Best Practices Research:
- **Context Engineering**: Agents must factor in collective knowledge before acting
- **Stigmergic Communication**: Priority-based message filtering (1-10 scale)
- **Orchestrator Pattern**: Central distillation agent coordinates synthesis
- **Temperature Diversity**: 0.3-0.9 range across agent roles
- **Shared State Management**: Prevent conflicting assumptions via unified context

### From Investor Reporting Best Practices:
- **Executive Summary**: High-level overview (2-3 paragraphs)
- **Visual Engagement**: Charts, metrics, style boxes
- **Clear Recommendations**: Buy/Sell/Hold with rationale
- **Risk-Return Analysis**: Downside scenarios and probability
- **Future Outlook**: Market projections and catalysts
- **Actionable Insights**: Specific next steps for investors

---

## Phase 1: Agent Temperature & Prompt Diversity

### 1.1 Temperature Configuration by Tier

**File**: `src/agents/swarm/base_swarm_agent.py`

Add temperature configuration to BaseSwarmAgent:

```python
# Tier-specific temperature profiles
TIER_TEMPERATURES = {
    1: 0.3,  # Oversight & Coordination (focused, deterministic)
    2: 0.5,  # Market Intelligence (balanced)
    3: 0.4,  # Fundamental & Macro (analytical)
    4: 0.6,  # Risk & Sentiment (exploratory)
    5: 0.5,  # Options & Volatility (balanced)
    6: 0.2,  # Execution & Compliance (highly deterministic)
    7: 0.7   # Recommendation Engine (creative synthesis)
}

class BaseSwarmAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        priority: int = 5,
        confidence_threshold: float = 0.6,
        tier: int = 1,  # NEW
        temperature: Optional[float] = None  # NEW
    ):
        # ... existing code ...
        self.tier = tier
        self.temperature = temperature or TIER_TEMPERATURES.get(tier, 0.5)
        logger.info(f"{self.agent_type} initialized with temperature={self.temperature}")
```

### 1.2 Role-Specific Prompt Templates

**File**: `src/agents/swarm/prompt_templates.py` (NEW)

Create differentiated prompts for each agent role:

```python
ROLE_PERSPECTIVES = {
    "MarketAnalyst": {
        "perspective": "You are a market microstructure specialist focused on price action, volume, and liquidity.",
        "focus": "Analyze order flow, bid-ask spreads, and market depth.",
        "avoid": "Do not duplicate fundamental or sentiment analysis."
    },
    "FundamentalAnalyst": {
        "perspective": "You are a value investor focused on intrinsic worth and financial health.",
        "focus": "Analyze earnings, cash flow, balance sheet strength, and valuation multiples.",
        "avoid": "Do not duplicate technical or sentiment analysis."
    },
    "RiskManager": {
        "perspective": "You are a risk officer focused on downside protection and tail risk.",
        "focus": "Identify potential losses, correlation risks, and stress scenarios.",
        "avoid": "Do not duplicate performance or fundamental analysis."
    },
    # ... add all 17 agents
}

def get_agent_prompt(agent_type: str, base_context: str) -> str:
    """Generate role-specific prompt with unique perspective"""
    role_config = ROLE_PERSPECTIVES.get(agent_type, {})
    return f"""
{role_config.get('perspective', '')}

FOCUS AREAS:
{role_config.get('focus', '')}

AVOID DUPLICATION:
{role_config.get('avoid', '')}

CONTEXT:
{base_context}

Provide your unique perspective based on your specialized role.
"""
```

---

## Phase 2: Deduplication & Context Engineering

### 2.1 Enhanced SharedContext with Deduplication

**File**: `src/agents/swarm/shared_context.py`

Add deduplication logic:

```python
class SharedContext:
    def __init__(self, max_messages: int = 1000):
        # ... existing code ...
        self._message_hashes: Set[str] = set()  # NEW: Track content hashes
        self._topic_coverage: Dict[str, List[str]] = defaultdict(list)  # NEW: Track topics
    
    def _hash_content(self, content: Dict[str, Any]) -> str:
        """Generate hash of message content for deduplication"""
        import hashlib
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def send_message(self, message: Message) -> bool:
        """
        Send message with deduplication check.
        Returns True if message was added, False if duplicate.
        """
        with self._lock:
            content_hash = self._hash_content(message.content)
            
            # Check for duplicate
            if content_hash in self._message_hashes:
                logger.info(f"Duplicate message from {message.source} - skipping")
                return False
            
            # Add message
            self._message_hashes.add(content_hash)
            self._messages.append(message)
            self._messages_by_source[message.source].append(message)
            
            # Track topic coverage
            topic = message.content.get('topic', 'general')
            self._topic_coverage[topic].append(message.source)
            
            self._metrics['total_messages'] += 1
            return True
    
    def get_uncovered_topics(self, agent_id: str) -> List[str]:
        """Get topics this agent hasn't covered yet"""
        covered = set()
        for topic, agents in self._topic_coverage.items():
            if agent_id in agents:
                covered.add(topic)
        
        all_topics = {'market_structure', 'fundamentals', 'risk', 'sentiment', 
                      'options', 'volatility', 'macro', 'technical'}
        return list(all_topics - covered)
```

### 2.2 Context Injection for Agents

**File**: `src/agents/swarm/base_swarm_agent.py`

Add context awareness:

```python
class BaseSwarmAgent(ABC):
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of what other agents have already analyzed"""
        messages = self.shared_context.get_messages(
            min_priority=7,  # Only high-priority insights
            max_age_seconds=3600
        )
        
        return {
            'topics_covered': list(self.shared_context._topic_coverage.keys()),
            'key_insights': [m.content for m in messages[:10]],
            'uncovered_topics': self.shared_context.get_uncovered_topics(self.agent_id)
        }
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analyze with context awareness"""
        # Get what others have done
        swarm_context = self.get_context_summary()
        
        # Add to analysis context
        enhanced_context = {
            **context,
            'swarm_insights': swarm_context['key_insights'],
            'focus_on': swarm_context['uncovered_topics']
        }
        
        return self._perform_analysis(enhanced_context)
    
    @abstractmethod
    def _perform_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-specific analysis implementation"""
        pass
```

---

## Phase 3: Distillation Agent (Tier 8)

### 3.1 Create DistillationAgent

**File**: `src/agents/swarm/agents/distillation_agent.py` (NEW)

```python
"""
Distillation Agent - Tier 8 Synthesis Layer

Synthesizes outputs from all 17 agents into investor-friendly narratives.
Prevents redundancy and creates cohesive stories.
"""

import logging
from typing import Dict, Any, List
from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext, Message
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class DistillationAgent(LLMAgentBase):
    """
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
        super().__init__(
            agent_id="distillation_agent",
            agent_type="DistillationAgent",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            llm_provider=llm_provider,
            model_name=model_name,
            tier=8,  # New tier for synthesis
            temperature=0.7,  # Creative synthesis
            priority=10  # Highest priority
        )
    
    def synthesize_swarm_output(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize all agent outputs into investor-friendly narrative.
        
        Returns structured report with:
        - Executive Summary
        - Investment Recommendation (Buy/Sell/Hold)
        - Risk Assessment
        - Future Outlook
        - Key Catalysts
        - Actionable Next Steps
        """
        # Gather all agent insights
        agent_messages = self._gather_agent_insights()
        
        # Deduplicate and prioritize
        unique_insights = self._deduplicate_insights(agent_messages)
        
        # Generate narrative sections
        narrative = self._generate_narrative(unique_insights, position_data)
        
        return narrative
    
    def _gather_agent_insights(self) -> List[Message]:
        """Gather high-priority insights from all agents"""
        return self.shared_context.get_messages(
            min_priority=6,
            max_age_seconds=3600
        )
    
    def _deduplicate_insights(self, messages: List[Message]) -> Dict[str, List[Dict]]:
        """Group and deduplicate insights by category"""
        categories = {
            'bullish_signals': [],
            'bearish_signals': [],
            'risk_factors': [],
            'opportunities': [],
            'technical_levels': [],
            'fundamental_metrics': []
        }
        
        seen_content = set()
        
        for msg in messages:
            content_key = str(msg.content)
            if content_key in seen_content:
                continue
            
            seen_content.add(content_key)
            
            # Categorize insight
            category = self._categorize_insight(msg.content)
            if category in categories:
                categories[category].append({
                    'source': msg.source,
                    'content': msg.content,
                    'confidence': msg.confidence,
                    'priority': msg.priority
                })
        
        return categories
    
    def _categorize_insight(self, content: Dict[str, Any]) -> str:
        """Categorize insight into appropriate bucket"""
        # Simple keyword-based categorization
        text = str(content).lower()
        
        if any(word in text for word in ['bullish', 'upside', 'buy', 'positive']):
            return 'bullish_signals'
        elif any(word in text for word in ['bearish', 'downside', 'sell', 'negative']):
            return 'bearish_signals'
        elif any(word in text for word in ['risk', 'volatility', 'uncertainty']):
            return 'risk_factors'
        elif any(word in text for word in ['opportunity', 'catalyst', 'growth']):
            return 'opportunities'
        elif any(word in text for word in ['support', 'resistance', 'level']):
            return 'technical_levels'
        else:
            return 'fundamental_metrics'
    
    def _generate_narrative(
        self, 
        insights: Dict[str, List[Dict]], 
        position_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investor-friendly narrative from insights"""
        
        prompt = self._build_synthesis_prompt(insights, position_data)
        
        response = self.call_llm(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=2000
        )
        
        # Parse LLM response into structured sections
        narrative = self._parse_narrative_response(response)
        
        return narrative
    
    def _build_synthesis_prompt(
        self, 
        insights: Dict[str, List[Dict]], 
        position_data: Dict[str, Any]
    ) -> str:
        """Build prompt for narrative synthesis"""
        return f"""
You are an institutional investment analyst creating a client-facing report.

POSITION DATA:
{json.dumps(position_data, indent=2)}

AGENT INSIGHTS:
Bullish Signals: {len(insights['bullish_signals'])} insights
Bearish Signals: {len(insights['bearish_signals'])} insights
Risk Factors: {len(insights['risk_factors'])} insights
Opportunities: {len(insights['opportunities'])} insights

DETAILED INSIGHTS:
{json.dumps(insights, indent=2)}

Create an investor-friendly report with these sections:

1. EXECUTIVE SUMMARY (2-3 paragraphs)
   - High-level investment thesis
   - Key takeaways
   - Overall recommendation

2. INVESTMENT RECOMMENDATION
   - Clear Buy/Sell/Hold rating
   - Conviction level (High/Medium/Low)
   - Price target (if applicable)
   - Rationale (3-5 bullet points)

3. RISK ASSESSMENT
   - Primary risks (ranked by severity)
   - Probability and impact
   - Mitigation strategies

4. FUTURE OUTLOOK
   - 3-month, 6-month, 12-month projections
   - Key catalysts to watch
   - Scenarios (bull/base/bear case)

5. ACTIONABLE NEXT STEPS
   - Specific actions for investor
   - Monitoring triggers
   - Rebalancing recommendations

Use clear, professional language. Avoid jargon. Be specific and actionable.
"""
    
    def _parse_narrative_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM narrative into structured sections"""
        # Simple section parsing (can be enhanced with structured output)
        sections = {
            'executive_summary': '',
            'recommendation': {},
            'risk_assessment': {},
            'future_outlook': {},
            'next_steps': []
        }
        
        # Parse sections from response
        # (Implementation depends on LLM output format)
        
        return {
            'narrative': llm_response,
            'sections': sections,
            'generated_at': datetime.utcnow().isoformat()
        }
```

---

## Phase 4: Integration into Swarm Coordinator

**File**: `src/agents/swarm/swarm_coordinator.py`

Add distillation step:

```python
from .agents.distillation_agent import DistillationAgent

class SwarmCoordinator:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add Distillation Agent (Tier 8)
        self.distillation_agent = DistillationAgent(
            shared_context=self.shared_context,
            consensus_engine=self.consensus_engine
        )
    
    async def analyze_positions(self, positions: List[Dict]) -> Dict[str, Any]:
        """Enhanced analysis with distillation"""
        
        # Run all 17 agents (existing code)
        agent_results = await self._run_all_agents(positions)
        
        # NEW: Synthesize into investor narrative
        investor_report = self.distillation_agent.synthesize_swarm_output({
            'positions': positions,
            'agent_results': agent_results
        })
        
        return {
            'investor_report': investor_report,  # NEW: Narrative output
            'technical_details': agent_results,  # Keep for debugging
            'consensus_decisions': self.consensus_engine.get_decisions(),
            'swarm_health': self._get_swarm_health()
        }
```

---

## Phase 5: Frontend Display Components

### 5.1 Investor Report Viewer

**File**: `frontend/src/components/InvestorReportViewer.tsx` (NEW)

```typescript
interface InvestorReport {
  executive_summary: string;
  recommendation: {
    rating: 'BUY' | 'SELL' | 'HOLD';
    conviction: 'HIGH' | 'MEDIUM' | 'LOW';
    price_target?: number;
    rationale: string[];
  };
  risk_assessment: {
    primary_risks: Array<{
      risk: string;
      severity: 'HIGH' | 'MEDIUM' | 'LOW';
      probability: number;
      mitigation: string;
    }>;
  };
  future_outlook: {
    projections: {
      '3_month': string;
      '6_month': string;
      '12_month': string;
    };
    catalysts: string[];
    scenarios: {
      bull: string;
      base: string;
      bear: string;
    };
  };
  next_steps: string[];
}

export const InvestorReportViewer: React.FC<{report: InvestorReport}> = ({report}) => {
  return (
    <div className="investor-report">
      {/* Executive Summary */}
      <section className="executive-summary">
        <h2>Executive Summary</h2>
        <p>{report.executive_summary}</p>
      </section>
      
      {/* Recommendation Card */}
      <section className="recommendation-card">
        <div className={`rating-badge ${report.recommendation.rating.toLowerCase()}`}>
          {report.recommendation.rating}
        </div>
        <div className="conviction">
          Conviction: {report.recommendation.conviction}
        </div>
        {report.recommendation.price_target && (
          <div className="price-target">
            Target: ${report.recommendation.price_target}
          </div>
        )}
        <ul className="rationale">
          {report.recommendation.rationale.map((point, i) => (
            <li key={i}>{point}</li>
          ))}
        </ul>
      </section>
      
      {/* Risk Assessment */}
      <section className="risk-assessment">
        <h2>Risk Assessment</h2>
        {report.risk_assessment.primary_risks.map((risk, i) => (
          <div key={i} className="risk-item">
            <div className={`severity-badge ${risk.severity.toLowerCase()}`}>
              {risk.severity}
            </div>
            <h3>{risk.risk}</h3>
            <div className="probability">
              Probability: {(risk.probability * 100).toFixed(0)}%
            </div>
            <p className="mitigation">{risk.mitigation}</p>
          </div>
        ))}
      </section>
      
      {/* Future Outlook */}
      <section className="future-outlook">
        <h2>Future Outlook</h2>
        <div className="projections">
          <div>3M: {report.future_outlook.projections['3_month']}</div>
          <div>6M: {report.future_outlook.projections['6_month']}</div>
          <div>12M: {report.future_outlook.projections['12_month']}</div>
        </div>
        <div className="scenarios">
          <div className="bull">Bull: {report.future_outlook.scenarios.bull}</div>
          <div className="base">Base: {report.future_outlook.scenarios.base}</div>
          <div className="bear">Bear: {report.future_outlook.scenarios.bear}</div>
        </div>
      </section>
      
      {/* Next Steps */}
      <section className="next-steps">
        <h2>Recommended Actions</h2>
        <ol>
          {report.next_steps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      </section>
    </div>
  );
};
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Add temperature configuration to BaseSwarmAgent
- [ ] Create prompt_templates.py with role-specific prompts
- [ ] Update all 17 agents to use tier-specific temperatures

### Week 2: Deduplication
- [ ] Enhance SharedContext with deduplication logic
- [ ] Add context awareness to BaseSwarmAgent
- [ ] Test deduplication with sample runs

### Week 3: Distillation Agent
- [ ] Create DistillationAgent class
- [ ] Implement synthesis logic
- [ ] Integrate into SwarmCoordinator

### Week 4: Frontend & Testing
- [ ] Create InvestorReportViewer component
- [ ] Update SwarmAnalysisPage to display narrative
- [ ] End-to-end testing with real positions
- [ ] Gather feedback and iterate

---

## Success Metrics

1. **Redundancy Reduction**: <10% duplicate insights across agents
2. **Output Diversity**: Shannon entropy >0.7 across agent outputs
3. **Investor Readability**: Flesch-Kincaid grade level 10-12
4. **Synthesis Quality**: >80% of insights incorporated into narrative
5. **User Satisfaction**: Qualitative feedback from investors

---

## Where to Find Results

After implementation:
- **Code**: `src/agents/swarm/agents/distillation_agent.py`
- **Prompts**: `src/agents/swarm/prompt_templates.py`
- **Frontend**: `frontend/src/components/InvestorReportViewer.tsx`
- **API Response**: `/api/swarm/analyze-csv` returns `investor_report` field
- **Tests**: `tests/test_distillation_agent.py`

