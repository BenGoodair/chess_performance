#!/bin/bash

# Chess Data Fetcher - AWS Deployment Script
# Run this on your AWS EC2 instance

set -e

echo "🚀 Starting Chess Data Fetcher deployment..."

# Update system
sudo yum update -y
sudo yum install -y docker git

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create directory structure
mkdir -p ~/chess-analysis/{data,Figures,code}
cd ~/chess-analysis

# Copy your files here (you'll need to upload them via SCP or git)
echo "📁 Please upload your files to ~/chess-analysis/"
echo "   - code/chess_data_async.py"
echo "   - code/requirements.txt" 
echo "   - code/Dockerfile"
echo "   - docker-compose.yml"

# Set permissions
chmod -R 755 data Figures

echo "✅ Setup complete! Run the following commands to start:"
echo "   cd ~/chess-analysis"
echo "   docker-compose up -d"
echo "   docker-compose logs -f  # to monitor progress"

# Monitoring commands
cat << 'EOF' > monitor.sh
#!/bin/bash
echo "=== Chess Data Fetcher Status ==="
docker-compose ps
echo ""
echo "=== Disk Usage ==="
df -h ~/chess-analysis/data/
echo ""
echo "=== Latest Log Entries ==="
tail -20 ~/chess-analysis/data/chess_analysis.log 2>/dev/null || echo "No log file yet"
echo ""
echo "=== CSV File Size ==="
ls -lh ~/chess-analysis/data/*.csv 2>/dev/null || echo "No CSV files yet"
EOF

chmod +x monitor.sh

echo "📊 Run ./monitor.sh to check progress"