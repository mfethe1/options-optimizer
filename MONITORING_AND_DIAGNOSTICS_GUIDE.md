# 🔍 Swarm Analysis Monitoring & Diagnostics Guide

**Complete guide to monitoring, troubleshooting, and understanding the 17-agent swarm system**

---

## 📋 **TABLE OF CONTENTS**

1. [Frontend Monitoring](#frontend-monitoring)
2. [Backend Monitoring](#backend-monitoring)
3. [Agent Conversation Viewer](#agent-conversation-viewer)
4. [Progress Tracking](#progress-tracking)
5. [Diagnostic Endpoints](#diagnostic-endpoints)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Performance Monitoring](#performance-monitoring)

---

## 🎨 **FRONTEND MONITORING**

### **1. Analysis Progress Tracker**

**Component**: `AnalysisProgressTracker.tsx`

**What it shows**:
- Overall progress (X/17 agents complete)
- Current analysis step
- Elapsed time
- Estimated time remaining
- Agent-by-agent progress grouped by tier
- Real-time status updates

**How to use**:
```typescript
<AnalysisProgressTracker
  isAnalyzing={loading}
  currentStep="Running 17-agent swarm analysis..."
  estimatedTimeRemaining={180}
/>
```

**Visual Features**:
- ✓ Green checkmarks for completed agents
- 🔄 Spinning icons for agents in progress
- ✗ Red X for failed agents
- Progress bars for overall and per-agent progress
- Collapsible details view

---

### **2. Agent Conversation Viewer**

**Component**: `AgentConversationViewer.tsx`

**What it shows**:
- Real-time agent-to-agent messages
- Message priority levels (1-10)
- Confidence scores (0-100%)
- Message types (fundamental, technical, risk, etc.)
- Timestamps for all communications

**Three View Modes**:

1. **Timeline View**: Chronological message feed
2. **By Agent View**: Messages grouped by agent
3. **High Priority View**: Only critical messages (priority ≥ 7)

**Statistics Displayed**:
- Total messages exchanged
- Number of active agents
- Average priority
- Average confidence
- High-priority message count

**How to use**:
```typescript
<AgentConversationViewer
  discussionLogs={analysisResult.discussion_logs}
  agentInsights={analysisResult.agent_insights}
/>
```

---

### **3. Swarm Health Metrics**

**Component**: `SwarmHealthMetrics.tsx`

**What it shows**:
- Agent contribution (17 agents, success rate)
- Communication stats (messages, state updates)
- Consensus strength (confidence levels)
- Warning alerts for failures

**Key Metrics**:
- Success rate progress bar
- Total agents / Contributed / Failed breakdown
- Total messages exchanged
- Average message priority
- Average confidence
- Consensus strength for:
  - Overall Action
  - Risk Level
  - Market Outlook

---

### **4. Position Analysis Panel**

**Component**: `PositionAnalysisPanel.tsx`

**What it shows**:
- Position-by-position breakdown
- 5 tabs per position:
  - Overview (metrics, Greeks)
  - Agent Insights (all 17 agents' analysis)
  - Stock Report (comprehensive)
  - Recommendations (replacements)
  - Risks & Opportunities

---

## 🖥️ **BACKEND MONITORING**

### **1. Swarm Logger**

**Module**: `src/api/swarm_monitoring.py`

**Enhanced logging with emojis and structure**:

```python
from src.api.swarm_monitoring import SwarmLogger

logger = SwarmLogger(__name__)
logger.set_analysis_id(analysis_id)

# Log major steps
logger.step("Importing positions from CSV")

# Log agent activity
logger.agent_start("fundamental_analyst_1")
logger.agent_complete("fundamental_analyst_1", duration=2.5)
logger.agent_error("risk_manager_1", "API timeout")

# Log metrics
logger.metric("positions_imported", 25)
logger.success("Analysis complete!")
logger.warning("Low confidence detected")
```

**Output Example**:
```
[abc123] ▶ Importing positions from CSV
[abc123] 🤖 fundamental_analyst_1 starting analysis
[abc123] ✓ fundamental_analyst_1 complete (2.50s)
[abc123] ✗ risk_manager_1 failed: API timeout
[abc123] 📊 positions_imported: 25
[abc123] ✓ Analysis complete!
[abc123] ⚠ Low confidence detected
```

---

### **2. Analysis Monitor**

**Module**: `src/api/swarm_monitoring.py`

**Thread-safe singleton for tracking analyses**:

```python
from src/api.swarm_monitoring import monitor

# Start tracking
analysis_id = monitor.start_analysis({
    'file_name': 'positions.csv',
    'positions_count': 25
})

# Update progress
monitor.update_step(analysis_id, "Running agent analysis")
monitor.update_agent_progress(analysis_id, "fundamental_analyst_1", "analyzing")
monitor.update_agent_progress(analysis_id, "fundamental_analyst_1", "complete")

# Record metrics
monitor.add_metric(analysis_id, "consensus_confidence", 0.85)
monitor.add_warning(analysis_id, "Agent timeout detected")
monitor.add_error(analysis_id, "Failed to fetch market data")

# Complete
monitor.complete_analysis(analysis_id, status="completed", result_summary={
    'agents_contributed': 17,
    'consensus_reached': True
})
```

---

## 🔌 **DIAGNOSTIC ENDPOINTS**

### **Base URL**: `http://localhost:8000/api/monitoring`

### **1. System Health**

```bash
GET /api/monitoring/health
```

**Returns**:
```json
{
  "active_analyses": 2,
  "completed_analyses": 15,
  "total_agents_tracked": 17,
  "healthy_agents": 16,
  "health_percentage": 94.1,
  "timestamp": "2025-01-17T10:30:00Z"
}
```

---

### **2. Active Analyses**

```bash
GET /api/monitoring/analyses/active
```

**Returns**:
```json
{
  "count": 2,
  "analyses": [
    {
      "id": "abc123",
      "status": "in_progress",
      "current_step": "Running agent analysis",
      "start_time": 1705489800.0,
      "agent_progress": {
        "fundamental_analyst_1": {
          "status": "complete",
          "timestamp": "2025-01-17T10:30:15Z"
        },
        "risk_manager_1": {
          "status": "analyzing",
          "progress": 0.65,
          "timestamp": "2025-01-17T10:30:20Z"
        }
      },
      "errors": [],
      "warnings": [],
      "metrics": {
        "positions_imported": 25
      }
    }
  ]
}
```

---

### **3. Specific Analysis Status**

```bash
GET /api/monitoring/analyses/{analysis_id}
```

**Returns**: Detailed status for a specific analysis

---

### **4. Agent Statistics**

```bash
GET /api/monitoring/agents/statistics
```

**Returns**:
```json
{
  "summary": {
    "total_agents": 17,
    "total_calls": 340,
    "total_successes": 325,
    "total_failures": 15,
    "overall_success_rate": 95.6
  },
  "agents": {
    "fundamental_analyst_1": {
      "total_calls": 20,
      "successful_calls": 19,
      "failed_calls": 1,
      "avg_time": 2.3,
      "last_error": "API timeout",
      "last_success": "2025-01-17T10:30:00Z"
    }
  }
}
```

---

### **5. Comprehensive Diagnostics**

```bash
GET /api/monitoring/diagnostics
```

**Returns**:
- System health
- Active analyses count
- Recent errors (last 10)
- Problematic agents (>10% failure rate)
- Agent statistics summary

**Perfect for troubleshooting!**

---

## 🔧 **TROUBLESHOOTING GUIDE**

### **Problem 1: Analysis Stuck / Not Progressing**

**Check**:
1. Active analyses: `GET /api/monitoring/analyses/active`
2. Agent progress: Look at `agent_progress` field
3. Identify stuck agent

**Solution**:
```bash
# Check which agent is stuck
curl http://localhost:8000/api/monitoring/analyses/{analysis_id}

# Check agent statistics
curl http://localhost:8000/api/monitoring/agents/{agent_id}/statistics

# Check backend logs
# Look for errors from the stuck agent
```

---

### **Problem 2: High Failure Rate**

**Check**:
```bash
# Get diagnostics
curl http://localhost:8000/api/monitoring/diagnostics

# Look at problematic_agents section
```

**Common Causes**:
- API key missing/invalid (Claude, OpenAI)
- LMStudio not running
- Network timeout
- Rate limiting

**Solution**:
- Check `.env` file for API keys
- Start LMStudio: `http://localhost:1234`
- Check agent statistics for specific errors

---

### **Problem 3: Low Consensus Confidence**

**Check**:
- Swarm Health Metrics component
- Agent Conversation Viewer (High Priority tab)
- Look for conflicting recommendations

**Solution**:
- Review agent insights
- Check if agents have divergent opinions
- May indicate market uncertainty (not a bug!)

---

### **Problem 4: Frontend Shows "Demo Data"**

**Check**:
1. Backend logs: `python -m uvicorn src.api.main:app --reload`
2. Browser console: Look for `[SwarmService]` logs
3. Network tab: Check API response

**Solution**:
- Verify backend is running
- Check API endpoint: `http://localhost:8000/api/swarm/analyze-csv`
- Verify response structure matches interface

---

## 📊 **PERFORMANCE MONITORING**

### **Expected Timings**

**Sequential Execution** (current):
- CSV Import: 1-2 seconds
- Agent Analysis: 3-5 minutes (17 agents × 10-20s each)
- Consensus: 1-2 seconds
- **Total**: 3-5 minutes

**Parallel Execution** (future):
- CSV Import: 1-2 seconds
- Agent Analysis: 20-30 seconds (parallel)
- Consensus: 1-2 seconds
- **Total**: 25-35 seconds

### **Monitor Performance**:

```bash
# Check agent average times
curl http://localhost:8000/api/monitoring/agents/statistics | jq '.agents[] | {agent: .agent_id, avg_time: .avg_time}'

# Check system health
curl http://localhost:8000/api/monitoring/health
```

---

## 🎯 **QUICK REFERENCE**

### **Frontend Components**
- `AnalysisProgressTracker` - Real-time progress
- `AgentConversationViewer` - Agent messages
- `SwarmHealthMetrics` - Health dashboard
- `PositionAnalysisPanel` - Position details

### **Backend Modules**
- `swarm_monitoring.py` - Monitoring infrastructure
- `monitoring_routes.py` - API endpoints
- `SwarmLogger` - Enhanced logging
- `AnalysisMonitor` - Progress tracking

### **API Endpoints**
- `/api/monitoring/health` - System health
- `/api/monitoring/analyses/active` - Active analyses
- `/api/monitoring/agents/statistics` - Agent stats
- `/api/monitoring/diagnostics` - Full diagnostics

### **Logging Emojis**
- ▶ Step
- 🤖 Agent activity
- ✓ Success
- ✗ Error
- ⚠ Warning
- 📊 Metric
- ℹ Info

---

## 📍 **WHERE TO FIND EVERYTHING**

**Frontend**:
- `frontend/src/components/AnalysisProgressTracker.tsx`
- `frontend/src/components/AgentConversationViewer.tsx`
- `frontend/src/components/SwarmHealthMetrics.tsx`
- `frontend/src/components/PositionAnalysisPanel.tsx`

**Backend**:
- `src/api/swarm_monitoring.py`
- `src/api/monitoring_routes.py`

**Documentation**:
- `MONITORING_AND_DIAGNOSTICS_GUIDE.md` (this file)
- `SWARM_ANALYSIS_PAGE_FIX_COMPLETE.md`

---

**You now have comprehensive monitoring and diagnostic capabilities for the 17-agent swarm system!** 🚀

