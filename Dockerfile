# Multi-stage build for aerospace-mcp
# Stage 1: Build dependencies with UV
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies to a virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies only
RUN uv pip install --no-cache-dir -e .

# Stage 2: Runtime image with minimal footprint
FROM python:3.11-slim as runtime

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory and copy application code
WORKDIR /app
COPY --chown=appuser:appuser . .

# Create directories for logs and data with proper permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port 8080 (configurable via environment)
EXPOSE 8080

# Health check command
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables for configuration
ENV AEROSPACE_MCP_HOST=0.0.0.0
ENV AEROSPACE_MCP_PORT=8080
ENV AEROSPACE_MCP_MODE=http
ENV AEROSPACE_MCP_LOG_LEVEL=info

# Default command - can be overridden for MCP mode
CMD ["sh", "-c", "if [ \"$AEROSPACE_MCP_MODE\" = \"mcp\" ]; then aerospace-mcp; else aerospace-mcp-http; fi"]

# Labels for metadata
LABEL maintainer="Aaron <aaron@example.com>"
LABEL description="Aerospace MCP server for flight planning operations"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/username/aerospace-mcp"
LABEL org.opencontainers.image.description="MCP server for aerospace and flight planning operations"
LABEL org.opencontainers.image.licenses="MIT"