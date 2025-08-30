# Production Deployment Guide

Complete guide for deploying Aerospace MCP in production environments with security, scalability, and reliability best practices.

## 📋 Table of Contents

- [Deployment Overview](#deployment-overview)
- [Environment Configuration](#environment-configuration)
- [Deployment Strategies](#deployment-strategies)
- [Security Best Practices](#security-best-practices)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling & Performance](#scaling--performance)
- [Backup & Disaster Recovery](#backup--disaster-recovery)
- [Maintenance & Updates](#maintenance--updates)
- [Troubleshooting](#troubleshooting)

## 🏗️ Deployment Overview

### Architecture Options

```mermaid
graph TB
    subgraph "Option 1: Single Server"
        LB1[Load Balancer]
        Server1[Aerospace MCP Server]
        DB1[(SQLite/File)]
        LB1 --> Server1
        Server1 --> DB1
    end
    
    subgraph "Option 2: Multi-Server"
        LB2[Load Balancer]
        Server2A[Server A]
        Server2B[Server B]
        Server2C[Server N]
        Cache[Redis Cache]
        DB2[(PostgreSQL)]
        
        LB2 --> Server2A
        LB2 --> Server2B
        LB2 --> Server2C
        
        Server2A --> Cache
        Server2B --> Cache
        Server2C --> Cache
        
        Server2A --> DB2
        Server2B --> DB2
        Server2C --> DB2
    end
    
    subgraph "Option 3: Containerized"
        K8S[Kubernetes Cluster]
        Pods[Multiple Pods]
        PersistentStorage[Persistent Storage]
        
        K8S --> Pods
        Pods --> PersistentStorage
    end
```

### Deployment Checklist

**Pre-deployment:**
- [ ] Environment variables configured
- [ ] Dependencies installed and tested
- [ ] SSL certificates obtained
- [ ] Monitoring setup completed
- [ ] Backup strategy implemented
- [ ] Security review completed
- [ ] Load testing performed

**Deployment:**
- [ ] Application deployed
- [ ] Health checks passing
- [ ] HTTPS configured
- [ ] Rate limiting enabled
- [ ] Authentication implemented
- [ ] Logging configured
- [ ] Monitoring active

**Post-deployment:**
- [ ] Smoke tests passed
- [ ] Performance metrics baseline
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Team notified

## ⚙️ Environment Configuration

### Environment Variables

Create a comprehensive `.env` file for production:

```bash
# Application Configuration
APP_NAME=aerospace-mcp
APP_VERSION=0.1.0
APP_ENV=production
DEBUG=false

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
WORKER_CLASS=uvicorn.workers.UvicornWorker
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50

# Security
SECRET_KEY=your-super-secret-key-change-this
API_KEY_HEADER=X-API-Key
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Database (if using external database)
DATABASE_URL=postgresql://user:password@localhost:5432/aerospace_mcp
REDIS_URL=redis://localhost:6379/0

# OpenAP Configuration
OPENAP_CACHE_SIZE=1000
OPENAP_TIMEOUT=30

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10

# Monitoring
LOG_LEVEL=info
LOG_FORMAT=json
METRICS_ENABLED=true
METRICS_PORT=9090

# External Services
WEATHER_API_KEY=your-weather-api-key
NOTAM_API_URL=https://api.example.com/notam

# TLS Configuration
SSL_CERT_FILE=/etc/ssl/certs/aerospace-mcp.crt
SSL_KEY_FILE=/etc/ssl/private/aerospace-mcp.key
```

### Configuration Management

**Using python-decouple:**

```python
# config.py
from decouple import config
from typing import List

class Settings:
    # Application
    APP_NAME: str = config("APP_NAME", default="aerospace-mcp")
    APP_ENV: str = config("APP_ENV", default="development")
    DEBUG: bool = config("DEBUG", default=False, cast=bool)
    
    # Server
    HOST: str = config("HOST", default="127.0.0.1")
    PORT: int = config("PORT", default=8000, cast=int)
    WORKERS: int = config("WORKERS", default=1, cast=int)
    
    # Security
    SECRET_KEY: str = config("SECRET_KEY")
    ALLOWED_HOSTS: List[str] = config("ALLOWED_HOSTS", default="", cast=lambda v: [i.strip() for i in v.split(',')] if v else [])
    
    # Database
    DATABASE_URL: str = config("DATABASE_URL", default="sqlite:///./aerospace.db")
    REDIS_URL: str = config("REDIS_URL", default="redis://localhost:6379/0")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = config("RATE_LIMIT_ENABLED", default=True, cast=bool)
    RATE_LIMIT_PER_MINUTE: int = config("RATE_LIMIT_PER_MINUTE", default=60, cast=int)

settings = Settings()
```

## 🚀 Deployment Strategies

### 1. Docker Deployment

**Dockerfile (Production):**

```dockerfile
# Multi-stage build for smaller production image
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast dependency management
RUN pip install uv

# Copy dependency files
WORKDIR /app
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv export --no-dev > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY main.py ./
COPY aerospace_mcp/ ./aerospace_mcp/
COPY app/ ./app/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use Gunicorn for production
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--preload", "--log-level", "info"]
```

**docker-compose.yml (Production):**

```yaml
version: '3.8'

services:
  aerospace-mcp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://aerospace:${DB_PASSWORD}@db:5432/aerospace_mcp
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80" 
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - aerospace-mcp
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=aerospace_mcp
      - POSTGRES_USER=aerospace
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aerospace"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
```

### 2. Kubernetes Deployment

**k8s-deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aerospace-mcp
  labels:
    app: aerospace-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aerospace-mcp
  template:
    metadata:
      labels:
        app: aerospace-mcp
    spec:
      containers:
      - name: aerospace-mcp
        image: aerospace-mcp:latest
        ports:
        - containerPort: 8000
        env:
        - name: APP_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aerospace-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: aerospace-secrets
              key: secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: aerospace-config
---
apiVersion: v1
kind: Service
metadata:
  name: aerospace-mcp-service
spec:
  selector:
    app: aerospace-mcp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aerospace-mcp-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.aerospace-mcp.com
    secretName: aerospace-tls
  rules:
  - host: api.aerospace-mcp.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aerospace-mcp-service
            port:
              number: 80
```

### 3. Traditional Server Deployment

**systemd service file (/etc/systemd/system/aerospace-mcp.service):**

```ini
[Unit]
Description=Aerospace MCP API Server
After=network.target
Requires=network.target

[Service]
Type=exec
User=aerospace
Group=aerospace
WorkingDirectory=/opt/aerospace-mcp
Environment=PATH=/opt/aerospace-mcp/.venv/bin
EnvironmentFile=/opt/aerospace-mcp/.env
ExecStart=/opt/aerospace-mcp/.venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aerospace-mcp

[Install]
WantedBy=multi-user.target
```

**Nginx configuration (/etc/nginx/sites-available/aerospace-mcp):**

```nginx
upstream aerospace_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=plan:10m rate=2r/s;

server {
    listen 80;
    server_name api.aerospace-mcp.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.aerospace-mcp.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/aerospace-mcp.crt;
    ssl_certificate_key /etc/ssl/private/aerospace-mcp.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 24h;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/atom+xml;

    # Rate limiting
    location /health {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://aerospace_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /plan {
        limit_req zone=plan burst=5 nodelay;
        proxy_pass http://aerospace_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
    }

    location / {
        limit_req zone=api burst=10 nodelay;
        proxy_pass http://aerospace_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Logging
    access_log /var/log/nginx/aerospace-mcp.access.log;
    error_log /var/log/nginx/aerospace-mcp.error.log;
}
```

## 🔒 Security Best Practices

### Authentication & Authorization

**API Key Authentication:**

```python
# security.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import hashlib
import hmac
import time

security = HTTPBearer()

class APIKeyAuth:
    def __init__(self, valid_keys: dict):
        self.valid_keys = valid_keys  # {key_id: key_secret}
    
    def verify_api_key(self, credentials: HTTPAuthorizationCredentials = Security(security)):
        """Verify API key from Authorization header."""
        try:
            # Expected format: "Bearer key_id:signature:timestamp"
            token = credentials.credentials
            parts = token.split(':')
            
            if len(parts) != 3:
                raise HTTPException(status_code=401, detail="Invalid token format")
            
            key_id, signature, timestamp = parts
            
            # Check if key exists
            if key_id not in self.valid_keys:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Check timestamp (prevent replay attacks)
            current_time = int(time.time())
            request_time = int(timestamp)
            if abs(current_time - request_time) > 300:  # 5 minutes
                raise HTTPException(status_code=401, detail="Token expired")
            
            # Verify signature
            secret = self.valid_keys[key_id]
            expected_sig = hmac.new(
                secret.encode(),
                f"{key_id}:{timestamp}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_sig):
                raise HTTPException(status_code=401, detail="Invalid signature")
            
            return {"key_id": key_id, "timestamp": timestamp}
            
        except Exception as e:
            raise HTTPException(status_code=401, detail="Authentication failed")

# Usage in main.py
from fastapi import Depends
from config import settings

api_auth = APIKeyAuth(settings.API_KEYS)

@app.post("/plan", dependencies=[Depends(api_auth.verify_api_key)])
def plan_flight(req: PlanRequest):
    # Protected endpoint
    pass
```

**JWT Authentication:**

```python
# jwt_auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

security = HTTPBearer()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Input Validation & Sanitization

```python
# Enhanced request models with strict validation
from pydantic import BaseModel, Field, validator
import re

class SecurePlanRequest(BaseModel):
    depart_city: str = Field(..., min_length=2, max_length=100, regex=r'^[a-zA-Z\s\-\.]+$')
    arrive_city: str = Field(..., min_length=2, max_length=100, regex=r'^[a-zA-Z\s\-\.]+$')
    depart_country: Optional[str] = Field(None, regex=r'^[A-Z]{2}$')
    arrive_country: Optional[str] = Field(None, regex=r'^[A-Z]{2}$')
    ac_type: str = Field(..., regex=r'^[A-Z0-9]{3,4}$')  # ICAO format
    cruise_alt_ft: int = Field(35000, ge=8000, le=45000)
    mass_kg: Optional[float] = Field(None, gt=1000, lt=1000000)  # Reasonable aircraft mass range
    route_step_km: float = Field(25.0, gt=1.0, le=1000.0)
    
    @validator('depart_city', 'arrive_city')
    def validate_city_names(cls, v):
        # Additional sanitization
        v = v.strip()
        if len(v) < 2:
            raise ValueError('City name too short')
        # Remove any potential injection attempts
        if any(char in v for char in ['<', '>', '"', "'"]):
            raise ValueError('Invalid characters in city name')
        return v
```

### Rate Limiting

```python
# rate_limiter.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import redis

# Redis-backed rate limiter for distributed deployments
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_client_id(request: Request):
    """Get client identifier for rate limiting."""
    # Try API key first, then IP
    auth_header = request.headers.get("authorization")
    if auth_header:
        # Extract key ID from token
        try:
            token = auth_header.replace("Bearer ", "")
            key_id = token.split(':')[0]
            return f"api_key:{key_id}"
        except:
            pass
    
    # Fall back to IP address
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    
    return f"ip:{request.client.host}"

limiter = Limiter(key_func=get_client_id, storage_uri="redis://localhost:6379")

# Different limits for different endpoints
@app.get("/health")
@limiter.limit("100/minute")
def health_check(request: Request):
    pass

@app.post("/plan")
@limiter.limit("10/minute")  # More restrictive for compute-intensive operations
def plan_flight(request: Request, req: PlanRequest):
    pass
```

### SSL/TLS Configuration

**Let's Encrypt with Certbot:**

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.aerospace-mcp.com

# Test automatic renewal
sudo certbot renew --dry-run

# Set up automatic renewal cron job
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

## 📊 Monitoring & Observability

### Structured Logging

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger
from config import settings

def setup_logging():
    """Configure structured JSON logging."""
    
    # Create custom formatter
    logHandler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logHandler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.addHandler(logHandler)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    return logger

# Usage in main.py
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Aerospace MCP server", extra={
        "event": "startup",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV
    })
    yield
    # Shutdown
    logger.info("Shutting down Aerospace MCP server", extra={
        "event": "shutdown"
    })

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info("HTTP request processed", extra={
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time,
        "user_agent": request.headers.get("user-agent"),
        "remote_addr": request.client.host
    })
    
    return response
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
FLIGHT_PLANS_TOTAL = Counter('flight_plans_total', 'Total flight plans generated', ['aircraft_type', 'status'])
OPENAP_CALCULATIONS = Histogram('openap_calculation_seconds', 'OpenAP calculation time')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### Health Checks

```python
# health.py
from fastapi import HTTPException
import psutil
import time
from typing import Dict, Any

async def detailed_health_check() -> Dict[str, Any]:
    """Comprehensive health check."""
    start_time = time.time()
    
    health_data = {
        "status": "ok",
        "timestamp": int(time.time()),
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "checks": {}
    }
    
    # Check OpenAP availability
    try:
        import openap
        health_data["checks"]["openap"] = {"status": "ok", "version": openap.__version__}
    except ImportError:
        health_data["checks"]["openap"] = {"status": "unavailable", "error": "Module not installed"}
    
    # Check database connectivity
    try:
        # Test database connection
        # db_check = await test_database_connection()
        health_data["checks"]["database"] = {"status": "ok", "response_time": 0.001}
    except Exception as e:
        health_data["checks"]["database"] = {"status": "error", "error": str(e)}
        health_data["status"] = "degraded"
    
    # Check Redis connectivity
    try:
        # Test Redis connection
        # redis_check = await test_redis_connection()
        health_data["checks"]["redis"] = {"status": "ok", "response_time": 0.001}
    except Exception as e:
        health_data["checks"]["redis"] = {"status": "error", "error": str(e)}
        health_data["status"] = "degraded"
    
    # System metrics
    health_data["system"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }
    
    # Response time
    health_data["response_time"] = time.time() - start_time
    
    # Overall status determination
    if any(check.get("status") == "error" for check in health_data["checks"].values()):
        health_data["status"] = "error"
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check endpoint."""
    return await detailed_health_check()
```

### Alerting Configuration

**Prometheus alerts (alerts.yml):**

```yaml
groups:
  - name: aerospace-mcp
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time"
          description: "95th percentile response time is {{ $value }}s"

      - alert: ServiceDown
        expr: up{job="aerospace-mcp"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Aerospace MCP service is not responding"

      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes / 1024 / 1024) > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB"
```

## ⚡ Scaling & Performance

### Horizontal Scaling

**Load Balancer Configuration (HAProxy):**

```haproxy
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

frontend aerospace_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/aerospace-mcp.pem
    redirect scheme https if !{ ssl_fc }
    default_backend aerospace_backend

backend aerospace_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server app1 10.0.1.10:8000 check inter 30s rise 2 fall 3
    server app2 10.0.1.11:8000 check inter 30s rise 2 fall 3
    server app3 10.0.1.12:8000 check inter 30s rise 2 fall 3
```

### Caching Strategy

```python
# caching.py
import redis
import json
import hashlib
from typing import Any, Optional
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_response(ttl: int = 3600):
    """Decorator to cache API responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            
            return result
        return wrapper
    return decorator

def generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function name and arguments."""
    key_data = {
        'func': func_name,
        'args': args,
        'kwargs': kwargs
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"

# Usage
@cache_response(ttl=1800)  # Cache for 30 minutes
async def cached_airport_search(city: str, country: Optional[str] = None):
    """Cached airport search."""
    return _find_city_airports(city, country)
```

### Database Optimization

```sql
-- PostgreSQL schema for external database
CREATE TABLE airports (
    id SERIAL PRIMARY KEY,
    iata VARCHAR(3) UNIQUE NOT NULL,
    icao VARCHAR(4) UNIQUE,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    country VARCHAR(2) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    timezone VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookups
CREATE INDEX idx_airports_city ON airports (city);
CREATE INDEX idx_airports_country ON airports (country);
CREATE INDEX idx_airports_city_country ON airports (city, country);
CREATE INDEX idx_airports_coordinates ON airports (latitude, longitude);

-- Full-text search index
CREATE INDEX idx_airports_search ON airports USING gin(to_tsvector('english', name || ' ' || city));
```

## 💾 Backup & Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily backup script

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/aerospace-mcp"
LOG_FILE="$BACKUP_DIR/backup_$DATE.log"

mkdir -p $BACKUP_DIR

echo "Starting backup at $(date)" >> $LOG_FILE

# Backup database
if [ "$DATABASE_URL" != "" ]; then
    pg_dump $DATABASE_URL | gzip > $BACKUP_DIR/database_$DATE.sql.gz
    echo "Database backup completed" >> $LOG_FILE
fi

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/aerospace-mcp/.env /opt/aerospace-mcp/config/
echo "Configuration backup completed" >> $LOG_FILE

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz /opt/aerospace-mcp/logs/
echo "Logs backup completed" >> $LOG_FILE

# Upload to S3 (optional)
if command -v aws &> /dev/null; then
    aws s3 sync $BACKUP_DIR s3://aerospace-mcp-backups/$(date +%Y/%m/%d)/
    echo "S3 upload completed" >> $LOG_FILE
fi

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.log" -mtime +30 -delete

echo "Backup completed at $(date)" >> $LOG_FILE
```

### Disaster Recovery Plan

**Recovery Runbook:**

1. **Assessment**
   - Determine scope of failure
   - Check backup integrity
   - Notify stakeholders

2. **Infrastructure Recovery**
   ```bash
   # Restore from backup
   cd /opt/aerospace-mcp
   
   # Restore database
   gunzip -c /backups/database_latest.sql.gz | psql $DATABASE_URL
   
   # Restore configuration
   tar -xzf /backups/config_latest.tar.gz -C /
   
   # Restart services
   sudo systemctl restart aerospace-mcp
   sudo systemctl restart nginx
   ```

3. **Verification**
   - Test health endpoints
   - Verify core functionality
   - Monitor error rates
   - Update status page

## 🔧 Maintenance & Updates

### Zero-Downtime Deployment

```bash
#!/bin/bash
# deploy.sh - Blue-green deployment script

NEW_VERSION=$1
CURRENT_PORT=8000
NEW_PORT=8001

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "Deploying version $NEW_VERSION"

# Build new version
docker build -t aerospace-mcp:$NEW_VERSION .

# Start new container on different port
docker run -d -p $NEW_PORT:8000 --name aerospace-mcp-new aerospace-mcp:$NEW_VERSION

# Wait for health check
sleep 30
while ! curl -f http://localhost:$NEW_PORT/health; do
    echo "Waiting for new version to start..."
    sleep 5
done

# Update load balancer to point to new version
# (This depends on your load balancer configuration)
update_load_balancer $NEW_PORT

# Wait for connections to drain
sleep 60

# Stop old container
docker stop aerospace-mcp-current
docker rm aerospace-mcp-current

# Rename containers
docker rename aerospace-mcp-new aerospace-mcp-current

echo "Deployment completed successfully"
```

### Update Procedures

**Regular Updates:**
```bash
# Weekly maintenance window
sudo systemctl stop aerospace-mcp

# Update application
cd /opt/aerospace-mcp
git pull origin main
source .venv/bin/activate
pip install --upgrade -r requirements.txt

# Update system packages
sudo apt update && sudo apt upgrade -y

# Restart services
sudo systemctl start aerospace-mcp
sudo systemctl restart nginx

# Verify deployment
curl -f http://localhost/health
```

## 🐛 Troubleshooting

### Common Production Issues

#### High CPU Usage

```bash
# Investigate high CPU
top -p $(pgrep -f aerospace-mcp)
htop

# Check for infinite loops or inefficient queries
strace -p $(pgrep -f aerospace-mcp)

# Analyze with profiling
py-spy top -p $(pgrep -f aerospace-mcp)
```

#### Memory Leaks

```bash
# Monitor memory usage
ps aux | grep aerospace-mcp
cat /proc/$(pgrep -f aerospace-mcp)/status | grep VmRSS

# Use memory profiler
memory_profiler -p $(pgrep -f aerospace-mcp)
```

#### Database Connection Issues

```bash
# Check connection pool
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity WHERE datname='aerospace_mcp';"

# Monitor slow queries
psql $DATABASE_URL -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### Emergency Procedures

**Service Recovery:**
```bash
# Quick service restart
sudo systemctl restart aerospace-mcp

# Force kill if unresponsive
sudo pkill -f aerospace-mcp
sudo systemctl start aerospace-mcp

# Rollback to previous version
docker run -d -p 8000:8000 aerospace-mcp:previous-stable

# Check logs for errors
journalctl -u aerospace-mcp -f
tail -f /var/log/aerospace-mcp/app.log
```

**Performance Issues:**
```bash
# Temporary rate limiting
iptables -A INPUT -p tcp --dport 8000 -m limit --limit 10/min -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP

# Scale up quickly (if using containers)
docker-compose up --scale aerospace-mcp=3
```

This deployment guide provides comprehensive coverage for running Aerospace MCP in production environments. Adapt the configurations based on your specific infrastructure requirements and compliance needs.