"""Demo script showing different deployment modes using drl-trading-common."""

import os
import threading
import time

from drl_trading_common.messaging import DeploymentMode
from drl_trading_common.messaging.factories import TradingMessageBusFactory


def demo_training_mode():
    """Demo training mode with in-memory transport."""
    print("=== TRAINING MODE DEMO ===")

    # Create message bus for training mode
    message_bus = TradingMessageBusFactory.create_message_bus(DeploymentMode.TRAINING)
    message_bus.start()

    # Set up a simple pipeline
    def feature_processor(data):
        print(f"🔧 Processing features for {data['symbol']}: {data['data']}")
        # Simulate feature computation
        features = {"rsi": 65.4, "ma_20": 1.2345}
        message_bus.publish_features_computed(
            data["symbol"], data["timeframe"], features
        )

    def signal_generator(data):
        print(f"🧠 Generating signal for {data['symbol']}: {data['features']}")
        # Simulate ML inference
        signal = {"action": "BUY", "confidence": 0.85, "timestamp": int(time.time())}
        message_bus.publish_trading_signal(data["symbol"], signal)

    def trade_executor(data):
        print(f"💰 Executing trade for {data['symbol']}: {data['signal']}")
        # Simulate trade execution
        execution_result = {"order_id": "12345", "status": "FILLED", "price": 1.2350}
        message_bus.publish_trade_executed(data["symbol"], execution_result)

    # Subscribe to events
    message_bus.subscribe_to_market_data("EURUSD", "H1", feature_processor)
    message_bus.subscribe_to_features("EURUSD", "H1", signal_generator)
    message_bus.subscribe_to_trading_signals("EURUSD", trade_executor)

    # Simulate market data arrival
    print("📊 Publishing market data...")
    message_bus.publish_market_data(
        "EURUSD",
        "H1",
        {
            "open": 1.2340,
            "high": 1.2355,
            "low": 1.2330,
            "close": 1.2350,
            "volume": 1000,
        },
    )

    time.sleep(1)  # Let events process
    message_bus.stop()
    print("✅ Training mode demo completed\n")


def demo_production_mode_simulation():
    """Demo production mode with in-memory transport (simulating distributed)."""
    print("=== PRODUCTION MODE SIMULATION ===")

    # For demo purposes, we'll use training mode but simulate distributed components
    message_bus = TradingMessageBusFactory.create_message_bus(DeploymentMode.TRAINING)
    message_bus.start()

    # Simulate different services running independently

    def data_ingestion_service():
        """Simulate data ingestion service."""
        print("🌐 Data Ingestion Service: Starting...")
        for i in range(3):
            time.sleep(0.5)
            message_bus.publish_market_data(
                "EURUSD",
                "H1",
                {
                    "open": 1.2340 + i * 0.001,
                    "high": 1.2355 + i * 0.001,
                    "low": 1.2330 + i * 0.001,
                    "close": 1.2350 + i * 0.001,
                    "volume": 1000 + i * 100,
                    "batch": i + 1,
                },
            )
        print("🌐 Data Ingestion Service: Completed")

    def inference_service():
        """Simulate ML inference service."""
        print("🧠 Inference Service: Starting...")

        def handle_inference_request(payload):
            symbol = payload["symbol"]
            features = payload["features"]
            print(f"🧠 Processing inference for {symbol}")
            # Simulate ML computation
            time.sleep(0.1)
            return {
                "signal": {
                    "action": "BUY" if features.get("rsi", 50) > 60 else "HOLD",
                    "confidence": 0.85,
                    "timestamp": int(time.time()),
                }
            }

        message_bus.handle_inference_requests("EURUSD", handle_inference_request)
        print("🧠 Inference Service: Ready")

    def feature_engineering_service():
        """Simulate feature engineering service."""
        print("⚙️ Feature Engineering Service: Starting...")

        def process_market_data(data):
            print(
                f"⚙️ Computing features for {data['symbol']} batch {data['data'].get('batch', 'N/A')}"
            )
            features = {
                "rsi": 65.4 + data["data"].get("batch", 0) * 2,
                "ma_20": data["data"]["close"],
                "computed_at": int(time.time()),
            }

            # Use RPC to get trading signal
            result = message_bus.request_inference(data["symbol"], features, timeout=2)
            if "error" not in result:
                message_bus.publish_trading_signal(data["symbol"], result["signal"])

        message_bus.subscribe_to_market_data("EURUSD", "H1", process_market_data)
        print("⚙️ Feature Engineering Service: Ready")

    def execution_service():
        """Simulate trade execution service."""
        print("💼 Execution Service: Starting...")

        def execute_signal(data):
            signal = data["signal"]
            if signal["action"] != "HOLD":
                print(f"💼 Executing {signal['action']} order for {data['symbol']}")
                execution_result = {
                    "order_id": f"ORD_{int(time.time())}",
                    "status": "FILLED",
                    "price": 1.2350,
                    "timestamp": int(time.time()),
                }
                message_bus.publish_trade_executed(data["symbol"], execution_result)

        message_bus.subscribe_to_trading_signals("EURUSD", execute_signal)
        print("💼 Execution Service: Ready")

    # Start services in separate threads (simulating distributed deployment)
    threads = []

    # Start inference service first
    inference_thread = threading.Thread(target=inference_service)
    inference_thread.start()
    time.sleep(0.1)  # Give it time to register handlers

    # Start other services
    for service in [feature_engineering_service, execution_service]:
        thread = threading.Thread(target=service)
        thread.start()
        threads.append(thread)
        time.sleep(0.1)

    # Start data ingestion (this will trigger the pipeline)
    data_thread = threading.Thread(target=data_ingestion_service)
    data_thread.start()
    threads.append(data_thread)

    # Wait for all services to complete
    for thread in threads:
        thread.join()

    inference_thread.join()

    time.sleep(1)  # Let final events process
    message_bus.stop()
    print("✅ Production mode simulation completed\n")


def demo_stage_switching() -> None:
    """Demo switching between modes via environment variable."""
    print("=== DEPLOYMENT MODE SWITCHING DEMO ===")

    # Test training stage
    os.environ["STAGE"] = "training"
    bus1 = TradingMessageBusFactory.create_message_bus(DeploymentMode.TRAINING)
    print(f"📋 Created bus for training stage: {type(bus1.transport).__name__}")

    # Test production stage (will use in-memory since RabbitMQ not available)
    os.environ["STAGE"] = "prod"
    try:
        bus2 = TradingMessageBusFactory.create_message_bus(DeploymentMode.PRODUCTION)
        print(f"📋 Created bus for production mode: {type(bus2.transport).__name__}")
    except Exception as e:
        print(f"📋 Production mode failed (expected): {e}")
        print("📋 In real deployment, RabbitMQ would be available")

    # Cleanup
    if "STAGE" in os.environ:
        del os.environ["STAGE"]

    print("✅ Mode switching demo completed\n")


if __name__ == "__main__":
    print("🚀 DRL Trading Common - Messaging Demo\n")
    print("This demo shows how the drl-trading-common library")
    print("provides pluggable messaging for all trading services.\n")

    demo_training_mode()
    demo_production_mode_simulation()
    demo_stage_switching()

    print("🎉 All demos completed!")
    print("\n💡 Key Benefits of drl-trading-common:")
    print("   ✓ Shared messaging infrastructure across all services")
    print("   ✓ Same API works in both training and production")
    print("   ✓ Training: Fast in-memory communication")
    print("   ✓ Production: Reliable message queues with idempotence")
    print("   ✓ Easy to switch modes via environment variable")
    print("   ✓ Components are loosely coupled and independently scalable")
    print("   ✓ Centralized messaging logic - DRY principle")
    print("\n📦 Import this library in any service:")
    print(
        "   from drl_trading_common.messaging import TradingMessageBus, DeploymentMode"
    )
