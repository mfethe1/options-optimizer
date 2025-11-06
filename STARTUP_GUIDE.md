# Options Analysis System - Startup Guide

## Quick Start (Ports: Backend 9001, Frontend 4001)

### Terminal 1 - Backend
```bash
cd E:\Projects\Options_probability
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 9001 --reload
```

**Wait for:** `INFO: Application startup complete.`

### Terminal 2 - Frontend
```bash
cd E:\Projects\Options_probability\frontend
npm run dev
```

**Access at:** http://localhost:4001

---

## Connection Architecture

```
Browser (localhost:4001)
    ↓
Vite Dev Server (port 4001)
    ↓ proxy /api/* → http://127.0.0.1:9001/*
    ↓ proxy /ws/*  → ws://127.0.0.1:9001/*
    ↓
FastAPI Backend (port 9001)
```

### API Calls
- Frontend: `fetch('/api/unified/forecast/all')`
- Vite proxies to: `http://127.0.0.1:9001/unified/forecast/all`

### WebSocket Calls
- Frontend: `new WebSocket('ws://localhost:4001/ws/agent-stream/user123')`
- Vite proxies to: `ws://127.0.0.1:9001/ws/agent-stream/user123`

---

## Configuration Files

### Frontend Environment (`.env.development`)
- `VITE_API_BASE_URL=http://localhost:9001`
- `VITE_WS_BASE_URL=ws://localhost:9001`
- `VITE_PORT=4001`
- `VITE_API_LOGGING=true`

### Vite Config (`vite.config.ts`)
- Server port: **4001**
- Proxy target: **http://127.0.0.1:9001**
- WebSocket proxy: Enabled for `/ws/*`
- Proxy logging: Enabled for debugging

### API Config (`src/config/api.config.ts`)
- Dynamic URL generation based on environment
- Auto-detect protocol (ws/wss, http/https)
- Built-in retry logic and logging

---

## Verification Checklist

### Backend Health
```bash
curl http://localhost:9001/health
# Should return: {"status": "healthy", ...}
```

### Frontend Health
```bash
curl http://localhost:4001
# Should return: HTML page with React app
```

### API Proxy Test
```bash
curl -X POST "http://localhost:4001/api/unified/forecast/all?symbol=SPY&time_range=1D"
# Should return: JSON with predictions and timeline data
```

### Browser Console
Navigate to http://localhost:4001 and check console for:
```
[API Config] Initialized: { apiBaseURL: '', wsBaseURL: 'ws://localhost:4001', ... }
```

---

## Troubleshooting

### Port Already in Use
**Frontend:**
```bash
# Find and kill process on 4001
netstat -ano | findstr :4001
taskkill //F //PID <PID>
```

**Backend:**
```bash
# Find and kill process on 9001
netstat -ano | findstr :9001
taskkill //F //PID <PID>
```

### Backend Stuck at "Waiting for application startup"
This is expected - wait 10-15 seconds for it to say "Application startup complete"

### 404 Not Found Errors
1. Check backend is running: `curl http://localhost:9001/health`
2. Check backend logs for "Unified analysis routes registered successfully"
3. Verify Vite proxy is working: Check browser Network tab for proxied requests

### WebSocket Connection Fails
1. Verify backend WebSocket endpoint: `curl http://localhost:9001/ws`
2. Check browser console for WebSocket errors
3. Ensure Vite proxy is configured for `/ws/*` paths

---

## Key Endpoints

### Backend (Direct - Port 9001)
- Health: `GET /health`
- Unified Forecast: `POST /unified/forecast/all?symbol=SPY&time_range=1D`
- Models Status: `GET /unified/models/status`
- Agent Stream WS: `ws://localhost:9001/ws/agent-stream/{user_id}`

### Frontend (Proxied - Port 4001)
- Home: `http://localhost:4001/`
- Health: `GET /api/health` → proxied to backend
- Unified Forecast: `POST /api/unified/forecast/all` → proxied to backend
- Agent Stream WS: `ws://localhost:4001/ws/agent-stream/{user_id}` → proxied to backend

---

## Data Flow Example

**UnifiedAnalysis Page Load:**

1. User navigates to `http://localhost:4001/`
2. React component mounts
3. `loadAllPredictions()` called
4. Fetches: `/api/unified/forecast/all?symbol=SPY&time_range=1D`
5. Vite proxy forwards to: `http://127.0.0.1:9001/unified/forecast/all?symbol=SPY&time_range=1D`
6. Backend:
   - Calls yfinance for live SPY data (5-min intervals for 1D)
   - Generates predictions from 5 ML models
   - Returns JSON with timeline + overlays
7. Frontend receives data and renders chart

**Data Includes:**
- Actual prices (OHLCV from yfinance)
- 5 Model predictions (Epidemic VIX, GNN, Mamba, PINN, Ensemble)
- Confidence scores
- Upper/lower bounds for range predictions

---

## Success Indicators

✅ Backend logs show: `INFO: Application startup complete.`
✅ Backend logs show: `Unified analysis routes registered successfully`
✅ Frontend shows: `VITE v5.4.20 ready in XXXms`
✅ Frontend shows: `Local: http://localhost:4001/`
✅ Browser console shows: `[API Config] Initialized`
✅ API test returns JSON (not 404/500)

---

## Notes

- **Backend startup**: ~10-15 seconds (TensorFlow loading)
- **Frontend startup**: ~1-2 seconds
- **Hot reload**: Both servers support auto-reload on file changes
- **Caching**: Backend caches market data for 60 seconds
- **Rate limiting**: 10 requests/minute for analysis endpoints

---

**Last Updated:** 2025-11-05
**System Version:** 0.4.0
**Ports:** Backend 9001, Frontend 4001
