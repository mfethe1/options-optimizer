# Deployment Guide - Options Optimizer Platform

Complete guide to deploy the world-class options analysis platform with 5 competitive advantages.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Backend Deployment](#backend-deployment)
4. [Frontend Deployment](#frontend-deployment)
5. [Production Deployment](#production-deployment)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.9+** (backend)
- **Node.js 18+** (frontend)
- **npm or yarn** (package manager)
- **Git** (version control)

### Optional (for production)

- **Docker & Docker Compose**
- **Nginx** (reverse proxy)
- **PostgreSQL** (database)
- **Redis** (caching)

---

## Environment Setup

### 1. Clone Repository

```bash
cd /home/user/options-optimizer
```

### 2. Create Environment Files

#### Backend (.env)

Create `.env` in the root directory:

```bash
# API Configuration
API_PORT=8000
API_HOST=0.0.0.0

# AI Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vision Analysis (optional - defaults to Anthropic)
VISION_PROVIDER=anthropic  # or "openai"

# Database (optional - uses SQLite by default)
DATABASE_URL=sqlite:///./options_optimizer.db

# Redis (optional - for production caching)
REDIS_URL=redis://localhost:6379

# Sentry (optional - for error tracking)
SENTRY_DSN=your_sentry_dsn_here

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

#### Frontend (.env)

Create `.env` in the `frontend/` directory:

```bash
VITE_API_URL=http://localhost:8000
```

---

## Backend Deployment

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Backend Server

#### Development Mode

```bash
# From project root
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode

```bash
# With Gunicorn (recommended for production)
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### 3. Verify Backend

Open browser to:
- **API Docs**: http://localhost:8000/docs
- **Root endpoint**: http://localhost:8000/

You should see:
```json
{
  "status": "ok",
  "service": "investor-report",
  "version": "0.4.0",
  "competitive_advantages": [
    "Natural language trading interface",
    "AI-powered chart image analysis",
    "Real-time anomaly detection",
    "Deep sentiment with influencer weighting",
    "Autonomous paper trading with multi-agent consensus"
  ]
}
```

---

## Frontend Deployment

### 1. Install Node Dependencies

```bash
cd frontend

# Using npm
npm install

# Or using yarn
yarn install
```

### 2. Start Frontend Development Server

```bash
# Using npm
npm run dev

# Or using yarn
yarn dev
```

The frontend will be available at: **http://localhost:5173**

### 3. Build for Production

```bash
# Using npm
npm run build

# Or using yarn
yarn build
```

Production build will be in `frontend/dist/`

### 4. Preview Production Build

```bash
# Using npm
npm run preview

# Or using yarn
yarn preview
```

---

## Production Deployment

### Option 1: Docker Deployment (Recommended)

#### 1. Create Dockerfile for Backend

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

#### 2. Create Dockerfile for Frontend

Create `frontend/Dockerfile`:

```dockerfile
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Build application
COPY . .
RUN npm run build

# Production stage with Nginx
FROM nginx:alpine

# Copy built files
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

#### 3. Create nginx.conf

Create `frontend/nginx.conf`:

```nginx
server {
    listen 80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html;

    # Frontend routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 4. Create docker-compose.yml

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/options_optimizer
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: options_optimizer
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  postgres_data:
```

#### 5. Deploy with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Option 2: Manual Production Deployment

#### Backend on Server

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv nginx

# 2. Clone and setup
cd /opt
sudo git clone <your-repo> options-optimizer
cd options-optimizer
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Create systemd service
sudo nano /etc/systemd/system/options-optimizer.service
```

Add:
```ini
[Unit]
Description=Options Optimizer API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/options-optimizer
Environment="PATH=/opt/options-optimizer/venv/bin"
ExecStart=/opt/options-optimizer/venv/bin/gunicorn src.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

```bash
# 4. Start service
sudo systemctl daemon-reload
sudo systemctl enable options-optimizer
sudo systemctl start options-optimizer
sudo systemctl status options-optimizer
```

#### Frontend on Server

```bash
# 1. Build locally
cd frontend
npm run build

# 2. Copy build to server
scp -r dist/ user@server:/var/www/options-optimizer

# 3. Configure Nginx
sudo nano /etc/nginx/sites-available/options-optimizer
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/options-optimizer/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

```bash
# 4. Enable and restart Nginx
sudo ln -s /etc/nginx/sites-available/options-optimizer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Testing

### Backend Tests

```bash
# Unit tests
pytest tests/

# API tests
pytest tests/api/

# Coverage report
pytest --cov=src tests/
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Run tests with UI
npm run test:ui

# Coverage report
npm run test:coverage
```

### End-to-End Testing

#### 1. Test Conversational Trading

```bash
curl -X POST "http://localhost:8000/api/conversation/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is theta decay?",
    "user_id": "test_user"
  }'
```

#### 2. Test Chart Analysis

```bash
curl -X POST "http://localhost:8000/api/vision/analyze-chart" \
  -F "image=@test_chart.png" \
  -F "analysis_type=comprehensive"
```

#### 3. Test Anomaly Detection

```bash
curl -X POST "http://localhost:8000/api/anomalies/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "NVDA"
  }'
```

#### 4. Test Sentiment Analysis

```bash
curl -X POST "http://localhost:8000/api/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "sources": ["twitter", "reddit"]
  }'
```

#### 5. Test Paper Trading

```bash
curl -X GET "http://localhost:8000/api/paper-trading/portfolio/test_user"
```

---

## Troubleshooting

### Backend Issues

#### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

#### Missing API Keys

If you see errors about missing API keys:

1. Check `.env` file exists in project root
2. Verify keys are set correctly:
   ```bash
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```
3. Restart backend server

#### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Frontend Issues

#### Build Errors

```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
npm run dev
```

#### API Connection Errors

1. Check `VITE_API_URL` in `frontend/.env`
2. Verify backend is running: `curl http://localhost:8000/health`
3. Check CORS settings in backend `.env`

#### TypeScript Errors

```bash
# Regenerate types
npm run build

# Check TypeScript version
npx tsc --version
```

### Docker Issues

#### Container Won't Start

```bash
# View logs
docker-compose logs backend
docker-compose logs frontend

# Rebuild
docker-compose down
docker-compose up -d --build
```

#### Database Connection Error

```bash
# Check database is running
docker-compose ps

# Access database
docker-compose exec db psql -U postgres -d options_optimizer
```

---

## Performance Optimization

### Backend

1. **Enable Redis Caching**:
   ```bash
   # Install Redis
   pip install redis

   # Set in .env
   REDIS_URL=redis://localhost:6379
   ```

2. **Increase Workers**:
   ```bash
   # In production
   --workers 8  # 2-4x CPU cores
   ```

3. **Database Connection Pooling**:
   ```python
   # In database.py
   DATABASE_POOL_SIZE=20
   DATABASE_MAX_OVERFLOW=40
   ```

### Frontend

1. **Enable Compression** (Nginx):
   ```nginx
   gzip on;
   gzip_types text/plain text/css application/json application/javascript;
   ```

2. **CDN for Static Assets**:
   - Upload `dist/assets/` to CDN
   - Update asset paths in build config

3. **Lazy Loading Routes**:
   ```typescript
   // In App.tsx
   const ConversationalTradingPage = lazy(() => import('./pages/ConversationalTradingPage'));
   ```

---

## Monitoring

### Backend Metrics

Access Prometheus metrics:
```
http://localhost:8000/metrics
```

### Health Checks

```bash
# Simple health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Logs

```bash
# Backend logs (systemd)
sudo journalctl -u options-optimizer -f

# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

---

## Security Checklist

- [ ] Change default database passwords
- [ ] Set strong JWT secrets
- [ ] Enable HTTPS (SSL certificates)
- [ ] Configure firewall rules
- [ ] Set up rate limiting
- [ ] Enable Sentry error tracking
- [ ] Rotate API keys regularly
- [ ] Set up backup strategy

---

## Backup Strategy

### Database Backup

```bash
# PostgreSQL
docker-compose exec db pg_dump -U postgres options_optimizer > backup_$(date +%Y%m%d).sql

# Restore
cat backup_20251030.sql | docker-compose exec -T db psql -U postgres options_optimizer
```

### Application Backup

```bash
# Backup code and config
tar -czf options_optimizer_backup_$(date +%Y%m%d).tar.gz \
  --exclude=venv \
  --exclude=node_modules \
  --exclude=.git \
  /opt/options-optimizer
```

---

## Support

For issues or questions:
- **API Documentation**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/mfethe1/options-optimizer/issues
- **API Enhancements**: See `API_ENHANCEMENTS.md`

---

**Last Updated**: 2025-10-30
**Platform Version**: 0.4.0
**Status**: Production Ready ✅
