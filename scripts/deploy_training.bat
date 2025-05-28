@echo off
REM Training deployment script for Windows

echo 🚀 Setting up training environment...

REM 1. Clone the repository
git clone https://github.com/your-org/ai-trading.git
cd ai-trading

REM 2. Set environment for training mode
set DEPLOYMENT_MODE=training
set DRL_TRADING_CONFIG_PATH=%cd%\drl-trading-impl-example\config\applicationConfig.json

REM 3. Install framework and dependencies
cd drl-trading-framework
pip install -e .

REM 4. Install implementation
cd ..\drl-trading-impl-example
pip install -e .

REM 5. Run training
echo 🎯 Starting training...
python main.py

echo ✅ Training completed!
