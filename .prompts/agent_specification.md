
# Agent Specification for Proprietary Trading Strategy under FTMO Constraints

## Overview

This document defines the intended behavior, constraints, and reward structure for a Deep Reinforcement Learning (DRL) agent operating in a trading environment that simulates the conditions of a proprietary trading firm, specifically modeled after FTMO.

The agent is expected to autonomously discover profitable trading strategies while **strictly adhering to risk management constraints**, including **daily drawdown** and **max total drawdown** limits. The reward structure is carefully designed to reflect performance across a variety of asset classes without bias toward any specific market.

---

## Core Objectives

1. **Maximize normalized PnL**: The agent should enter and exit trades that maximize profit while controlling for volatility.
2. **Comply with FTMO-style risk limits**:
   - Daily Drawdown: Do not exceed the allowed intraday loss.
   - Maximum Drawdown: Ensure equity never drops more than the allowed threshold from peak.
3. **Avoid hardcoded trading strategies**: The agent should learn from raw features without being explicitly biased toward scalping, trend-following, or any other human-defined style.
4. **Encourage statistically significant trading**: The strategy should include **sufficient trade frequency** to allow meaningful performance evaluation.

---

## Reward Function Design

### Description

The reward function is defined to reflect **normalized realized PnL**, conditioned on the agent being in a trade:

- If **not in trade**, reward is **zero**.
- If **in a trade**, reward is:

```python
reward = realized_pnl / (atr + 1e-8)
```

### Notes

- **ATR (Average True Range)** is used to normalize for volatility, making the reward invariant to asset class (e.g., crypto vs. forex).
- This encourages the agent to seek **high-efficiency trades**, not just high-return trades.
- **Fees, slippage, and leverage effects** are incorporated into the `realized_pnl`.

---

## Observation Space

The environment includes a **multi-timeframe, multi-feature observation space**, currently exceeding 100 features, including:

- Price data (OHLC)
- Momentum indicators (RSI, MACD, etc.)
- Volatility measures (ATR)
- Market structure
- Trade metadata (position size, leverage, etc.)

The agent is expected to learn feature importance on its own.

---

## Risk Management Constraints

### Daily Drawdown (FTMO-style)

- Tracked per trading day.
- If current equity drops below a threshold from the **start-of-day equity**, the episode **terminates** and a large **negative reward** is applied.

### Maximum Total Drawdown

- Tracked from peak equity.
- If current equity drops beyond this max drawdown limit, the episode **terminates** with **penalty**.

---

## Gym Environment Integration

### FTMOConstraintWrapper

A Gym wrapper (`FTMOConstraintWrapper`) will enforce risk management constraints by:

- Tracking `start_of_day_equity`, `max_equity`, and `current_equity`
- Terminating episodes when FTMO constraints are violated
- Returning informative `info["terminated_reason"]` flags to help the agent distinguish between:
  - Natural episode ends
  - Constraint violations

### Example Structure

```python
obs, reward, terminated, truncated, info = env.step(action)

if info.get("terminated_reason") == "daily_drawdown":
    # Episode ended due to FTMO rule violation
```

---

## Trade Duration Consideration

- The agent is **not explicitly penalized for long trades**.
- However, excessive duration may lead to fewer trades per episode, impacting statistical significance.
- **No artificial "magic numbers"** (e.g., max candle length) are enforced.
- The agent is expected to learn optimal exit timing based on embedded indicators like **volatility** and **RSI**, already present in the observation space.

---

## Evaluation Metrics

During training and validation, the following will be logged:

- Number of trades per episode
- Average trade duration
- Win/loss ratio
- Constraint violation rate
- Cumulative return vs. drawdown profile

---

## Final Notes

This environment does **not** aim to teach a specific strategy such as scalping or trend-following. Instead, the agent will be trained to:

- Operate within strict professional risk parameters
- Discover profitable patterns across various asset types
- Prioritize robustness and statistical validity over subjective trading heuristics

The core shaping mechanisms are:
- The **reward function** based on PnL/ATR
- The **FTMO-style constraints** acting as non-negotiable episode termination events

This setup is intended to simulate the decision-making constraints of real proprietary traders and enforce discipline through environmental design.
