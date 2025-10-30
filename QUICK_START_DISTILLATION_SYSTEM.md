# Quick Start: Distillation System Implementation

## 🎯 What We're Building

Transform the 17-agent swarm from producing technical JSON outputs to generating **investor-friendly narrative reports** with:
- ✅ No redundant analysis across agents
- ✅ Diverse perspectives via temperature variation
- ✅ Cohesive storytelling via Distillation Agent
- ✅ Digestible sections (Buy/Sell/Hold, Risk, Outlook, etc.)

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Working 17-agent swarm system
- ✅ Backend running on `http://localhost:8000`
- ✅ Frontend running on `http://localhost:5173`
- ✅ API keys configured in `.env`

---

## 🚀 Implementation Steps

### Step 1: Add Temperature Diversity (30 min)

**File**: `src/agents/swarm/base_swarm_agent.py`

```python
# Add at top of file
TIER_TEMPERATURES = {
    1: 0.3,  # Oversight (focused)
    2: 0.5,  # Market Intelligence (balanced)
    3: 0.4,  # Fundamental (analytical)
    4: 0.6,  # Risk & Sentiment (exploratory)
    5: 0.5,  # Options (balanced)
    6: 0.2,  # Compliance (deterministic)
    7: 0.7   # Recommendation (creative)
}

# Update __init__ method
def __init__(self, ..., tier: int = 1, temperature: Optional[float] = None):
    # ... existing code ...
    self.tier = tier
    self.temperature = temperature or TIER_TEMPERATURES.get(tier, 0.5)
```

**Test**:
```bash
python -c "from src.agents.swarm.base_swarm_agent import TIER_TEMPERATURES; print(TIER_TEMPERATURES)"
```

---

### Step 2: Create Role-Specific Prompts (45 min)

**File**: `src/agents/swarm/prompt_templates.py` (NEW)

```python
ROLE_PERSPECTIVES = {
    "MarketAnalyst": {
        "perspective": "You are a market microstructure specialist.",
        "focus": "Analyze order flow, volume, and liquidity.",
        "avoid": "Do not duplicate fundamental analysis."
    },
    # ... add all 17 agents
}

def get_agent_prompt(agent_type: str, context: str) -> str:
    role = ROLE_PERSPECTIVES.get(agent_type, {})
    return f"{role['perspective']}\n\nFOCUS: {role['focus']}\n\nCONTEXT: {context}"
```

**Test**:
```bash
python -c "from src.agents.swarm.prompt_templates import get_agent_prompt; print(get_agent_prompt('MarketAnalyst', 'AAPL analysis'))"
```

---

### Step 3: Add Deduplication to SharedContext (60 min)

**File**: `src/agents/swarm/shared_context.py`

```python
class SharedContext:
    def __init__(self, max_messages: int = 1000):
        # ... existing code ...
        self._message_hashes: Set[str] = set()  # NEW
        self._topic_coverage: Dict[str, List[str]] = defaultdict(list)  # NEW
    
    def send_message(self, message: Message) -> bool:
        """Returns True if added, False if duplicate"""
        content_hash = self._hash_content(message.content)
        if content_hash in self._message_hashes:
            logger.info(f"Duplicate from {message.source} - skipping")
            return False
        
        self._message_hashes.add(content_hash)
        self._messages.append(message)
        return True
    
    def _hash_content(self, content: Dict) -> str:
        import hashlib, json
        return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
```

**Test**:
```python
# In Python REPL
from src.agents.swarm.shared_context import SharedContext, Message
ctx = SharedContext()
msg1 = Message("agent1", {"analysis": "bullish"}, priority=5)
msg2 = Message("agent2", {"analysis": "bullish"}, priority=5)  # Duplicate content
print(ctx.send_message(msg1))  # Should print True
print(ctx.send_message(msg2))  # Should print False (duplicate)
```

---

### Step 4: Create Distillation Agent (90 min)

**File**: `src/agents/swarm/agents/distillation_agent.py` (NEW)

See full implementation in `DISTILLATION_AGENT_IMPLEMENTATION_PLAN.md` (lines 150-300)

Key methods:
- `synthesize_swarm_output()` - Main entry point
- `_gather_agent_insights()` - Collect high-priority messages
- `_deduplicate_insights()` - Remove redundant insights
- `_generate_narrative()` - Create investor-friendly story

**Test**:
```bash
python -c "from src.agents.swarm.agents.distillation_agent import DistillationAgent; print('Import successful')"
```

---

### Step 5: Integrate into SwarmCoordinator (30 min)

**File**: `src/agents/swarm/swarm_coordinator.py`

```python
from .agents.distillation_agent import DistillationAgent

class SwarmCoordinator:
    def __init__(self, ...):
        # ... existing agents ...
        
        # Add Distillation Agent
        self.distillation_agent = DistillationAgent(
            shared_context=self.shared_context,
            consensus_engine=self.consensus_engine
        )
    
    async def analyze_positions(self, positions: List[Dict]) -> Dict[str, Any]:
        # Run all agents
        agent_results = await self._run_all_agents(positions)
        
        # NEW: Synthesize narrative
        investor_report = self.distillation_agent.synthesize_swarm_output({
            'positions': positions,
            'agent_results': agent_results
        })
        
        return {
            'investor_report': investor_report,  # NEW
            'technical_details': agent_results,
            'consensus_decisions': self.consensus_engine.get_decisions()
        }
```

**Test**:
```bash
# Run analysis with CSV
curl -X POST http://localhost:8000/api/swarm/analyze-csv \
  -F "file=@data/examples/positions.csv" \
  | jq '.investor_report'
```

---

### Step 6: Create Frontend Component (60 min)

**File**: `frontend/src/components/InvestorReportViewer.tsx` (NEW)

See full implementation in `DISTILLATION_AGENT_IMPLEMENTATION_PLAN.md` (lines 302-370)

Key sections:
- Executive Summary
- Recommendation Card (Buy/Sell/Hold)
- Risk Assessment
- Future Outlook
- Next Steps

**Test**:
```bash
cd frontend
npm run dev
# Navigate to http://localhost:5173/swarm-analysis
# Upload CSV and verify investor report displays
```

---

### Step 7: Update SwarmAnalysisPage (30 min)

**File**: `frontend/src/pages/SwarmAnalysisPage.tsx`

```typescript
import { InvestorReportViewer } from '../components/InvestorReportViewer';

// In render method, add:
{analysisResult?.investor_report && (
  <div className="investor-report-section">
    <h2>Investment Analysis Report</h2>
    <InvestorReportViewer report={analysisResult.investor_report} />
  </div>
)}

{/* Keep technical details in collapsible section */}
<details>
  <summary>Technical Details (for debugging)</summary>
  {/* Existing agent insights, consensus, etc. */}
</details>
```

---

## ✅ Verification Checklist

After implementation, verify:

- [ ] **Temperature Diversity**: Check logs show different temps per agent
  ```bash
  grep "temperature=" logs/swarm.log | sort -u
  ```

- [ ] **Deduplication**: Verify duplicate messages are rejected
  ```bash
  grep "Duplicate" logs/swarm.log | wc -l
  ```

- [ ] **Distillation Agent**: Runs after all 17 agents complete
  ```bash
  grep "DistillationAgent" logs/swarm.log
  ```

- [ ] **Investor Report**: API returns `investor_report` field
  ```bash
  curl -X POST http://localhost:8000/api/swarm/analyze-csv \
    -F "file=@data/examples/positions.csv" \
    | jq 'has("investor_report")'
  ```

- [ ] **Frontend Display**: Report renders with all sections
  - Executive Summary ✓
  - Recommendation Card ✓
  - Risk Assessment ✓
  - Future Outlook ✓
  - Next Steps ✓

---

## 🐛 Troubleshooting

### Issue: Agents still producing duplicate insights

**Solution**: Check that agents are using `shared_context.send_message()` correctly

```python
# In each agent's analyze() method
success = self.shared_context.send_message(Message(
    source=self.agent_id,
    content={"analysis": result},
    priority=self.priority
))
if not success:
    logger.warning(f"{self.agent_id}: Duplicate insight detected")
```

### Issue: Distillation Agent not synthesizing properly

**Solution**: Verify LLM provider is configured

```python
# Check .env file
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Test LLM call
from src.agents.swarm.agents.distillation_agent import DistillationAgent
agent = DistillationAgent(...)
response = agent.call_llm("Test prompt", temperature=0.7)
print(response)
```

### Issue: Frontend not displaying investor report

**Solution**: Check API response structure

```bash
# Verify response includes investor_report
curl -X POST http://localhost:8000/api/swarm/analyze-csv \
  -F "file=@data/examples/positions.csv" \
  | jq '.investor_report | keys'

# Should show: ["executive_summary", "recommendation", "risk_assessment", ...]
```

---

## 📊 Monitoring Progress

### Backend Logs
```bash
tail -f logs/swarm.log | grep -E "(temperature|Duplicate|DistillationAgent)"
```

### Agent Statistics
```bash
curl http://localhost:8000/api/monitoring/agents/statistics | jq
```

### Message Deduplication Rate
```bash
# Count total messages vs duplicates
grep "send_message" logs/swarm.log | wc -l
grep "Duplicate" logs/swarm.log | wc -l
```

---

## 🎯 Success Criteria

Your implementation is successful when:

1. ✅ **Diversity**: Each agent uses different temperature (check logs)
2. ✅ **Deduplication**: <10% duplicate messages (check metrics)
3. ✅ **Synthesis**: Distillation agent runs after all agents
4. ✅ **Narrative Output**: API returns investor-friendly report
5. ✅ **Frontend Display**: Report renders with all sections
6. ✅ **User Feedback**: Investors find report clear and actionable

---

## 📚 Additional Resources

- **Full Implementation Plan**: `DISTILLATION_AGENT_IMPLEMENTATION_PLAN.md`
- **Section Recommendations**: `INVESTOR_REPORT_SECTIONS_RECOMMENDATIONS.md`
- **Monitoring Guide**: `MONITORING_AND_DIAGNOSTICS_GUIDE.md`
- **E2E Test Results**: `E2E_TEST_RESULTS.md`

---

## 🚀 Next Steps

After basic implementation:

1. **Gather Feedback**: Show reports to real investors
2. **Iterate on Sections**: Add/remove based on feedback (see `INVESTOR_REPORT_SECTIONS_RECOMMENDATIONS.md`)
3. **Enhance Visuals**: Add charts, style boxes, metrics dashboards
4. **Optimize Performance**: Cache common analyses, parallelize agents
5. **Add Customization**: Allow investors to choose report sections

---

## 💡 Pro Tips

1. **Start Simple**: Implement core 5 sections first, add more later
2. **Test Incrementally**: Verify each step before moving to next
3. **Monitor Closely**: Watch logs for duplicate insights and errors
4. **Iterate Fast**: Get feedback early and often
5. **Document Changes**: Update README with each modification

---

## Where to Find Results

After implementation:
- **Code**: `src/agents/swarm/agents/distillation_agent.py`
- **Prompts**: `src/agents/swarm/prompt_templates.py`
- **Frontend**: `frontend/src/components/InvestorReportViewer.tsx`
- **API**: `POST /api/swarm/analyze-csv` → `investor_report` field
- **Logs**: `logs/swarm.log` (search for "DistillationAgent")
- **Tests**: `tests/test_distillation_agent.py`

---

## Estimated Timeline

- **Step 1-3** (Foundation): 2-3 hours
- **Step 4-5** (Distillation): 2-3 hours
- **Step 6-7** (Frontend): 1-2 hours
- **Testing & Iteration**: 2-4 hours

**Total**: 1-2 days for basic implementation

---

## Questions?

Refer to:
- Implementation plan for detailed code
- Section recommendations for report structure
- Monitoring guide for debugging
- E2E test results for expected behavior

Good luck! 🚀

