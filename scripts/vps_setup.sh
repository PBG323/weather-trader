#!/bin/bash
#
# Weather Trader VPS Setup Script
# Run this on a fresh Ubuntu 22.04 VPS
#
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_REPO/main/scripts/vps_setup.sh | bash
#

set -e

echo "=========================================="
echo "  Weather Trader VPS Setup"
echo "=========================================="

# Update system
echo "[1/7] Updating system..."
apt update && apt upgrade -y

# Install dependencies
echo "[2/7] Installing Python and dependencies..."
apt install -y python3-pip python3-venv git ufw

# Clone repo
echo "[3/7] Cloning repository..."
cd /root
if [ -d "weather-trader" ]; then
    cd weather-trader
    git pull
else
    git clone https://github.com/YOUR_USERNAME/weather-trader.git
    cd weather-trader
fi

# Create virtual environment
echo "[4/7] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create systemd service
echo "[5/7] Creating auto-start service..."
cat > /etc/systemd/system/weather-trader.service << 'EOF'
[Unit]
Description=Weather Trader Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/weather-trader
Environment=PATH=/root/weather-trader/venv/bin:/usr/bin
ExecStart=/root/weather-trader/venv/bin/streamlit run weather_trader/dashboard.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable weather-trader

# Configure firewall
echo "[6/7] Configuring firewall..."
ufw allow 22
ufw allow 8501
ufw --force enable

echo "[7/7] Setup complete!"
echo ""
echo "=========================================="
echo "  NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Upload your .env file from your local machine:"
echo "   scp C:\\Users\\perry\\weather_trader\\.env root@$(curl -s ifconfig.me):/root/weather-trader/"
echo ""
echo "2. Upload your Kalshi private key:"
echo "   scp C:\\Users\\perry\\kalshi_private_key.pem root@$(curl -s ifconfig.me):/root/weather-trader/"
echo ""
echo "3. Update the private key path in .env:"
echo "   nano /root/weather-trader/.env"
echo "   Change KALSHI_PRIVATE_KEY_PATH=/root/weather-trader/kalshi_private_key.pem"
echo ""
echo "4. Start the service:"
echo "   systemctl start weather-trader"
echo ""
echo "5. Access dashboard at:"
echo "   http://$(curl -s ifconfig.me):8501"
echo ""
echo "=========================================="
