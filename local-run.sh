#!/bin/bash

# Chess Data Fetcher - Local Development Script
# For Mac without admin access

set -e

echo "🏠 Setting up local development environment..."

# Check if Docker Desktop is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker Desktop is not running. Please start Docker Desktop first."
    exit 1
fi

# Create directory structure if not exists
mkdir -p data Figures

# Build the Docker image
echo "🔨 Building Docker image..."
docker-compose build

# Test with small sample first
echo "🧪 Starting test run with limited concurrency..."
CONCURRENT_REQUESTS=10 docker-compose up

echo "✅ Local development setup complete!"

# Helper functions
cat << 'EOF' > local-monitor.sh
#!/bin/bash
echo "=== Local Chess Data Fetcher Status ==="
docker-compose ps
echo ""
echo "=== Data Directory Contents ==="
ls -la data/
echo ""
echo "=== Container Logs (last 20 lines) ==="
docker-compose logs --tail=20
EOF

chmod +x local-monitor.sh

echo ""
echo "📋 Available commands:"
echo "   docker-compose up -d     # Run in background"
echo "   docker-compose logs -f   # Follow logs"
echo "   docker-compose stop      # Stop container"
echo "   docker-compose down      # Stop and remove container"
echo "   ./local-monitor.sh       # Check status"
echo ""
echo "🎯 For production run, set CONCURRENT_REQUESTS=100 or higher"