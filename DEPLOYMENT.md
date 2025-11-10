# Deployment Guide

This document provides comprehensive deployment instructions for the Browser-Use REST API.

## Table of Contents

1. [Quick Start with Docker](#quick-start-with-docker)
2. [Local Development Setup](#local-development-setup)
3. [GitHub Actions CI/CD](#github-actions-cicd)
4. [Production Deployment](#production-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Troubleshooting](#troubleshooting)

## Quick Start with Docker

The fastest way to get started:

```bash
# Clone the repository
git clone <your-repo-url>
cd browser-use-api

# Create environment file
cp .env.example .env
# Edit .env with your API keys (optional for browser-use-llm)

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Test the API
curl http://localhost:5000/health
```

Access the API at `http://localhost:5000` and documentation at `http://localhost:5000/docs`

## Local Development Setup

For development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Install Chromium for Playwright
python -m playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the server
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

## GitHub Actions CI/CD

### Automatic Docker Builds

The project includes a GitHub Actions workflow that automatically:
- Builds Docker images on every push to main/master
- Tags images with semantic versions from git tags
- Publishes to Docker Container Registry
- Supports multi-architecture (amd64, arm64)

### Setup Steps

1. **Configure Docker Hub Secrets in GitHub**:
   - Go to Settings → Secrets and variables → Actions
   - Add `DOCKERHUB_USERNAME` with your Docker Hub username
   - Add `DOCKERHUB_TOKEN` with your Docker Hub access token
   - (Get token from: https://hub.docker.com/settings/security)

2. **Automatic Builds**: Images are built automatically on:
   - Push to main/master branch
   - Git tags (v1.0.0, v1.2.3, etc.)
   - Pull requests (build only)

3. **Use Pre-built Images**:
   ```bash
   docker pull YOUR_DOCKERHUB_USERNAME/browser-use-api:latest
   docker run -d -p 5000:5000 YOUR_DOCKERHUB_USERNAME/browser-use-api:latest
   ```

### Version Tagging

Create a release with semantic versioning:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This creates images tagged as:
- `v1.0.0` (exact version)
- `v1.0` (major.minor)
- `v1` (major)
- `latest` (if default branch)

## Production Deployment

### Docker Compose with Resource Limits

Create a production `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  browser-use-api:
    image: docker.com/YOUR_USERNAME/browser-use-api:v1.0.0
    container_name: browser-use-api
    ports:
      - "5000:5000"
    environment:
      - BROWSER_USE_API_KEY=${BROWSER_USE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Deploy:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: browser-use-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: browser-use-api
  template:
    metadata:
      labels:
        app: browser-use-api
    spec:
      containers:
      - name: browser-use-api
        image: docker.com/reqeique/browser-use-api:v1.0.0
        ports:
        - containerPort: 5000
        env:
        - name: BROWSER_USE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: browser-use-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: browser-use-api
spec:
  selector:
    app: browser-use-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

## Environment Configuration

### Required Environment Variables

- `BROWSER_USE_API_KEY` - For browser-use-llm model (get free key at cloud.browser-use.com)
- `OPENAI_API_KEY` - For GPT models (optional)
- `ANTHROPIC_API_KEY` - For Claude models (optional)
- `GOOGLE_API_KEY` - For Gemini models (optional)

### Docker Environment Variables

**Option 1: .env file**
```bash
cp .env.example .env
# Edit .env
docker-compose up -d
```

**Option 2: Command line**
```bash
docker run -e BROWSER_USE_API_KEY=your-key -p 5000:5000 browser-use-api
```

**Option 3: Docker secrets** (production)
```bash
echo "your-api-key" | docker secret create browser_use_key -
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker-compose logs browser-use-api
```

Check health:
```bash
docker inspect --format='{{.State.Health.Status}}' browser-use-api
```

### Chromium Issues

The Docker image includes all Chromium dependencies. If you encounter issues:

```bash
# Rebuild image
docker-compose build --no-cache

# Verify Chromium installation
docker-compose exec browser-use-api python -m playwright install chromium
```

### Memory Issues

Increase Docker memory limits in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

### Port Already in Use

Change the port mapping in docker-compose.yml:

```yaml
ports:
  - "8080:5000"  # Use port 8080 instead of 5000
```

### API Key Issues

Verify environment variables are loaded:

```bash
docker-compose exec browser-use-api env | grep API_KEY
```

### Build Errors

If GitHub Actions build fails:

1. Check workflow logs in GitHub Actions tab
2. Verify Dockerfile syntax
3. Check if base image is accessible
4. Ensure GITHUB_TOKEN has necessary permissions

## Monitoring

### Health Checks

Monitor container health:

```bash
# Docker health status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
curl http://localhost:5000/health
```

### Logs

View real-time logs:

```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs -f browser-use-api

# Last 100 lines
docker-compose logs --tail=100
```

### Metrics

Monitor container resources:

```bash
docker stats browser-use-api
```

## Scaling

### Horizontal Scaling

Deploy multiple replicas:

```yaml
# docker-compose.yml
services:
  browser-use-api:
    # ... configuration
    deploy:
      replicas: 3
```

### Load Balancing

Use nginx or traefik as a reverse proxy:

```nginx
upstream browser_use_api {
    server localhost:5000;
    server localhost:5001;
    server localhost:5002;
}

server {
    listen 80;
    location / {
        proxy_pass http://browser_use_api;
    }
}
```

## Security Best Practices

1. **Never commit .env files** - Use .gitignore
2. **Use specific version tags** - Not `latest` in production
3. **Implement rate limiting** - Use nginx or API gateway
4. **Add authentication** - JWT or API keys
5. **Use HTTPS** - TLS/SSL certificates
6. **Scan images** - Use vulnerability scanners
7. **Limit container permissions** - Run as non-root user
8. **Network isolation** - Use Docker networks

## Support

For issues and questions:
- GitHub Issues: [Your Repo Issues]
- Documentation: README.md
- Browser-Use Docs: https://docs.browser-use.com
