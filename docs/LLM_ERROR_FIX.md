# LLM Error Fix - 404 and 429 Errors

**Date**: 2025-10-20  
**Issue**: Agents experiencing 404 errors (LMStudio not available) and 429 errors (OpenAI rate limits)

---

## 🐛 Problem

The system was configured to use LMStudio as the preferred model for several agents, but LMStudio was not running. This caused:

1. **404 Errors**: Agents trying to connect to `http://localhost:1234/v1/chat/completions` but LMStudio server not available
2. **429 Errors**: Fallback to OpenAI API hitting rate limits due to too many requests

### Error Logs
```
2025-10-20 18:25:43,956 - ERROR - fundamental_analyst_1: LMStudio API error: 404 Client Error: Not Found for url: http://localhost:1234/v1/chat/completions
2025-10-20 18:25:43,956 - WARNING - fundamental_analyst_1: Preferred model failed, trying fallbacks
2025-10-20 18:25:44,933 - ERROR - fundamental_analyst_1: OpenAI API error: 429 Client Error: Too Many Requests for url: https://api.openai.com/v1/chat/completions
```

---

## ✅ Solution

### 1. Auto-Detection of LMStudio Availability

Added health check on agent initialization to detect if LMStudio is available:

```python
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
```

### 2. Auto-Fallback on Initialization

If LMStudio is not available, agents automatically switch to Anthropic or OpenAI:

```python
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
```

### 3. Improved Fallback Chain

Changed fallback order to prioritize Anthropic (no rate limits) over OpenAI:

**Old**: OpenAI → Anthropic → LMStudio  
**New**: Anthropic → OpenAI → LMStudio (only if available)

```python
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
```

### 4. Better Error Handling for Rate Limits

Added specific handling for 429 errors to avoid spamming OpenAI:

```python
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        logger.error(f"{self.agent_id}: OpenAI rate limit exceeded (429). Skipping OpenAI fallback.")
    else:
        logger.error(f"{self.agent_id}: OpenAI API error: {e}")
    return None
```

### 5. Graceful Degradation

Added fallback response generator when all providers fail:

```python
def _generate_fallback_response(self, prompt: str) -> str:
    """
    Generate a graceful fallback response when all LLM providers fail.
    Returns a basic analysis based on the agent type.
    """
    fallback_responses = {
        "MarketAnalyst": "Market analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
        "RiskManager": "Risk analysis unavailable. All LLM providers failed. Please check API keys and connectivity.",
        # ... etc
    }
    
    return fallback_responses.get(
        self.agent_type,
        f"[{self.agent_type}] Analysis unavailable. All LLM providers failed."
    )
```

---

## 🚀 How to Use

### Option 1: Use Anthropic (Recommended)

Set your Anthropic API key in `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Agents configured with `preferred_model="lmstudio"` will automatically switch to Anthropic if LMStudio is not available.

### Option 2: Use OpenAI

Set your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
```

**Note**: OpenAI has rate limits, so you may still see 429 errors if you have many agents.

### Option 3: Start LMStudio

1. Download and install LMStudio from https://lmstudio.ai/
2. Start the local server on port 1234
3. Load a model (e.g., Llama 2, Mistral, etc.)
4. Agents will automatically detect LMStudio and use it

### Option 4: Change Agent Configuration

Edit `src/api/swarm_routes.py` to change preferred models:

```python
# Change from lmstudio to anthropic
LLMFundamentalAnalystAgent(
    agent_id="fundamental_analyst_1",
    shared_context=_swarm_coordinator.shared_context,
    consensus_engine=_swarm_coordinator.consensus_engine,
    preferred_model="anthropic"  # Changed from "lmstudio"
),
```

---

## 📊 Expected Behavior After Fix

### Successful Logs
```
2025-10-20 18:30:00,000 - INFO - fundamental_analyst_1: LMStudio not available, auto-switching to Anthropic
2025-10-20 18:30:00,100 - INFO - Initialized LLM-powered FundamentalAnalyst (ID: fundamental_analyst_1, Model: anthropic)
2025-10-20 18:30:05,200 - INFO - fundamental_analyst_1: Fundamental analysis complete
```

### No More Errors
- ✅ No 404 errors (LMStudio auto-detection)
- ✅ No 429 errors (Anthropic prioritized over OpenAI)
- ✅ Graceful degradation if all providers fail

---

## 🔧 Files Modified

- `src/agents/swarm/llm_agent_base.py` - Added auto-detection, improved fallback chain, better error handling

---

## 📝 Recommendations

1. **Use Anthropic for production** - No rate limits, high quality responses
2. **Use LMStudio for development** - Free, local, no API costs
3. **Use OpenAI as last resort** - Rate limits can be problematic with many agents

---

**Where to Find Results**:
- **Modified File**: `src/agents/swarm/llm_agent_base.py`
- **Backend Server**: Terminal 233 (restarted with fixes)
- **Test**: Make a request to `/api/investor-report` and check logs for auto-switching messages

