import numpy as np
import pandas as pd
from talipp.indicators import RSI


def create_sample_ohlcv_data(num_rows=20):
    """Create sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible results

    # Generate sample price data with some realistic movement
    base_price = 100.0
    price_changes = np.random.normal(0, 2, num_rows)
    closes = [base_price]

    for change in price_changes[1:]:
        new_close = max(closes[-1] + change, 10)  # Ensure price doesn't go below 10
        closes.append(new_close)

    # Generate OHLC from close prices with some spread
    data = []
    for i, close in enumerate(closes):
        spread = abs(np.random.normal(0, 1))
        high = close + spread
        low = close - spread
        # Open is previous close (or base price for first candle)
        open_price = base_price if i == 0 else closes[i-1]
        volume = np.random.randint(1000, 10000)

        data.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })

    return pd.DataFrame(data)

def test_rsi_batch_vs_streaming():
    """Test RSI calculation in batch vs streaming mode"""
    print("=== RSI Batch vs Streaming Test ===\n")

    # Create sample data
    df = create_sample_ohlcv_data(20)
    print("Sample OHLCV Data:")
    print(df.to_string(index=False))
    print()

    # RSI period
    rsi_period = 14
      # 1. BATCH PROCESSING - Calculate RSI for entire dataset at once
    print("1. BATCH PROCESSING:")
    print("-" * 50)

    # RSI works with close prices, not OHLCV objects
    close_prices = df['close'].tolist()

    # Initialize RSI with all close prices at once
    rsi_batch = RSI(period=rsi_period)
    rsi_batch.add(close_prices)

    print(f"RSI({rsi_period}) batch results:")
    rsi_batch_values = [round(val, 4) if val is not None else None for val in rsi_batch]
    print(f"All values: {rsi_batch_values}")
    print(f"Latest RSI value (batch): {rsi_batch_values[-1]}")
    print()

    # 2. STREAMING PROCESSING - Add values incrementally
    print("2. STREAMING PROCESSING:")
    print("-" * 50)
      # Initialize empty RSI indicator
    rsi_streaming = RSI(period=rsi_period)

    print("Adding values incrementally:")
    for i, row in df.iterrows():
        # Add close price directly to streaming RSI
        rsi_streaming.add(row['close'])

        current_rsi = rsi_streaming[-1] if len(rsi_streaming) > 0 else None
        current_rsi_rounded = round(current_rsi, 4) if current_rsi is not None else None

        print(f"Row {i+1:2d}: Close={row['close']:6.2f} -> RSI={current_rsi_rounded}")

    print()
    rsi_streaming_values = [round(val, 4) if val is not None else None for val in rsi_streaming]
    print(f"RSI({rsi_period}) streaming results:")
    print(f"All values: {rsi_streaming_values}")
    print(f"Latest RSI value (streaming): {rsi_streaming_values[-1]}")
    print()

    # 3. COMPARISON
    print("3. COMPARISON:")
    print("-" * 50)

    batch_final = rsi_batch_values[-1]
    streaming_final = rsi_streaming_values[-1]

    print(f"Batch final RSI:     {batch_final}")
    print(f"Streaming final RSI: {streaming_final}")

    if batch_final == streaming_final:
        print("✅ SUCCESS: Batch and streaming RSI values match exactly!")
    else:
        print("❌ ERROR: Batch and streaming RSI values do not match!")
        if batch_final is not None and streaming_final is not None:
            diff = abs(batch_final - streaming_final)
            print(f"Difference: {diff}")

    # Compare all values
    print("\nFull comparison:")
    all_match = True
    for i, (batch_val, stream_val) in enumerate(zip(rsi_batch_values, rsi_streaming_values)):
        match_status = "✅" if batch_val == stream_val else "❌"
        print(f"Index {i+1:2d}: Batch={batch_val}, Streaming={stream_val} {match_status}")
        if batch_val != stream_val:
            all_match = False

    print(f"\nOverall result: {'✅ All values match!' if all_match else '❌ Some values differ!'}")

    return all_match

if __name__ == "__main__":
    # Install talipp if needed
    try:
        import talipp
        print(f"Using talipp version: {talipp.__version__ if hasattr(talipp, '__version__') else 'unknown'}")
        print()
    except ImportError:
        print("talipp not found. Please install it with: pip install talipp")
        exit(1)

    # Run the test
    test_rsi_batch_vs_streaming()
