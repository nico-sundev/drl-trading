#!/bin/bash
# Training deployment script

set -e

echo "🚀 Setting up training environment..."

# 1. Clone the repository
REPO_URL=${REPO_URL:-https://github.com/your-org/ai-trading.git}
git clone "$REPO_URL"
cd ai-trading

# 2. Set environment for training mode
export DEPLOYMENT_MODE=training
export DRL_TRADING_CONFIG_PATH=$(pwd)/drl-trading-strategy-example/config/applicationConfig.json

# 3. Install framework and dependencies
cd drl-trading-core
pip install -e .

# 4. Install implementation
cd ../drl-trading-strategy-example
pip install -e .

# 5. Run training
echo "🎯 Starting training..."
python main.py

echo "✅ Training completed!"
