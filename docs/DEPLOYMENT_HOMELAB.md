# Aerospace MCP Deployment Guide

This guide provides comprehensive instructions for deploying the Aerospace MCP server using Docker, Docker Compose, and Kubernetes in home network environments.

## Quick Start

### Docker Compose (Recommended for Home Labs)

1. **Clone and configure**:
   ```bash
   cd aerospace-mcp
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

2. **Build and run**:
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment**:
   ```bash
   curl http://localhost:8080/health
   ```

### Docker Only

```bash
# Build the image
docker build -t aerospace-mcp .

# Run with environment variables
docker run -d \
  --name aerospace-mcp \
  -p 8080:8080 \
  -e AEROSPACE_MCP_MODE=http \
  -e AEROSPACE_MCP_LOG_LEVEL=info \
  --restart unless-stopped \
  aerospace-mcp
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AEROSPACE_MCP_MODE` | `http` | Server mode: `http` or `mcp` |
| `AEROSPACE_MCP_HOST` | `0.0.0.0` | Bind address |
| `AEROSPACE_MCP_PORT` | `8080` | Listen port |
| `AEROSPACE_MCP_LOG_LEVEL` | `info` | Log level: debug, info, warning, error |
| `AEROSPACE_MCP_ENV` | `production` | Environment: development, production |

See `.env.example` for complete configuration options.

### Modes of Operation

- **HTTP Mode** (`AEROSPACE_MCP_MODE=http`): REST API server for web integration
- **MCP Mode** (`AEROSPACE_MCP_MODE=mcp`): MCP protocol server for Claude Desktop

## Home Network Integration

### Port Configuration

By default, the service runs on port 8080. Adjust in your environment:

```env
# .env file
AEROSPACE_PORT=8080  # External port
```

### Reverse Proxy Integration

#### Traefik

Add labels to docker-compose.yml:

```yaml
services:
  aerospace-mcp:
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.aerospace-mcp.rule=Host(`aerospace-mcp.local`)"
      - "traefik.http.services.aerospace-mcp.loadbalancer.server.port=8080"
```

#### Nginx Proxy Manager

1. Add new proxy host
2. Domain: `aerospace-mcp.local`
3. Forward to: `<docker-host-ip>:8080`

#### Caddy

```caddy
aerospace-mcp.local {
    reverse_proxy localhost:8080
}
```

### Local DNS Setup

Add to your local DNS or `/etc/hosts`:
```
192.168.1.100 aerospace-mcp.local
```

## Security Hardening

### Container Security Features

- ✅ Non-root user (UID 1000)
- ✅ Read-only root filesystem
- ✅ No new privileges
- ✅ Minimal base image (python:3.11-slim)
- ✅ Security context configured
- ✅ Resource limits enforced

### Network Security

1. **Internal Network**: Service runs in isolated Docker network
2. **Firewall**: Only expose necessary ports
3. **HTTPS**: Use reverse proxy for SSL termination

### Secrets Management

For sensitive configuration:

```yaml
# docker-compose.yml
services:
  aerospace-mcp:
    secrets:
      - weather_api_key
    environment:
      - WEATHER_API_KEY_FILE=/run/secrets/weather_api_key

secrets:
  weather_api_key:
    file: ./secrets/weather_api_key.txt
```

## Resource Management

### Home Server Specifications

Minimum requirements:
- **CPU**: 1 core, 500MHz
- **RAM**: 256MB
- **Storage**: 100MB

Recommended for production:
- **CPU**: 2 cores, 1GHz
- **RAM**: 512MB
- **Storage**: 1GB (for logs and caching)

### Resource Limits

Configure in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      memory: 512M
      cpus: '0.5'
    reservations:
      memory: 256M
      cpus: '0.25'
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (k3s, k0s, or full cluster)
- kubectl configured
- Ingress controller (nginx, traefik)

### Deploy to Kubernetes

1. **Create namespace** (optional):
   ```bash
   kubectl create namespace aerospace-mcp
   ```

2. **Apply configurations**:
   ```bash
   kubectl apply -f kubernetes/configmap.yaml
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   kubectl apply -f kubernetes/ingress.yaml
   ```

3. **Verify deployment**:
   ```bash
   kubectl get pods -l app=aerospace-mcp
   kubectl get services aerospace-mcp-service
   ```

### Ingress Configuration

Edit `kubernetes/ingress.yaml` for your setup:

```yaml
spec:
  rules:
  - host: aerospace-mcp.yourdomain.com  # Your domain
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

## Monitoring and Troubleshooting

### Health Checks

```bash
# Docker Compose
docker-compose ps
docker-compose logs aerospace-mcp

# Docker
docker ps
docker logs aerospace-mcp

# Kubernetes
kubectl get pods -l app=aerospace-mcp
kubectl logs -l app=aerospace-mcp
```

### API Endpoints

- **Health Check**: `GET /health`
- **Airport Search**: `GET /airports/by_city?city=San%20Jose`
- **Flight Planning**: `POST /plan`

### Performance Monitoring

Monitor resource usage:

```bash
# Docker stats
docker stats aerospace-mcp

# Kubernetes resource usage
kubectl top pods -l app=aerospace-mcp
```

### Common Issues

#### Port Conflicts
```bash
# Check port usage
ss -tulpn | grep 8080
# or
netstat -tulpn | grep 8080
```

#### Memory Issues
```bash
# Increase memory limits
docker-compose down
# Edit docker-compose.yml memory limits
docker-compose up -d
```

#### OpenAP Dependencies
```bash
# Check if OpenAP is working
curl http://localhost:8080/health | jq '.openap'
```

## Backup and Recovery

### Data Backup

```bash
# Backup logs and configuration
docker run --rm \
  -v aerospace-mcp_logs:/data \
  -v $(pwd)/backup:/backup \
  busybox tar czf /backup/aerospace-mcp-$(date +%Y%m%d).tar.gz /data
```

### Container Updates

```bash
# Pull latest image
docker-compose pull

# Recreate containers
docker-compose up -d --force-recreate
```

## Integration Examples

### Home Assistant Integration

```yaml
# configuration.yaml
rest:
  - resource: "http://aerospace-mcp.local:8080/health"
    sensor:
      - name: "Aerospace MCP Status"
        value_template: "{{ value_json.status }}"
```

### Grafana Dashboard

Monitor with Prometheus metrics (future feature):

```yaml
# docker-compose.yml
services:
  aerospace-mcp:
    environment:
      - ENABLE_METRICS=true
    ports:
      - "8080:8080"
      - "9090:9090"  # Metrics endpoint
```

## Advanced Configuration

### Multi-Instance Deployment

For high availability:

```yaml
# docker-compose.yml
services:
  aerospace-mcp-1:
    # ... configuration
    container_name: aerospace-mcp-1

  aerospace-mcp-2:
    # ... configuration
    container_name: aerospace-mcp-2
    ports:
      - "8081:8080"
```

### External Database Integration

Future feature for persistent flight plans:

```yaml
services:
  aerospace-mcp:
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/aerospace
    depends_on:
      - postgres

  postgres:
    image: postgres:15-alpine
    # ... postgres configuration
```

## Support and Troubleshooting

### Log Analysis

```bash
# View recent logs
docker-compose logs --tail=50 -f aerospace-mcp

# Search for errors
docker-compose logs aerospace-mcp 2>&1 | grep ERROR
```

### Debug Mode

```bash
# Enable debug logging
docker-compose down
echo "AEROSPACE_MCP_LOG_LEVEL=debug" >> .env
docker-compose up -d
```

For additional support, check the application logs and ensure all dependencies are properly installed.
