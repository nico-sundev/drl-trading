# Gym Trading Environment — Interactive UI Renderer Specification

## Objective
Build a modular and intuitive UI rendering module to visually inspect the agent's behavior, action decisions, and reward shaping logic over time. This tool is intended for debugging, visual strategy evaluation, and validating the reward engine's effectiveness.

---

## Core Features

### 1. OHLC Candlestick Chart (Main Area)
- Plot base timeframe candles with full OHLC data.
- Overlay action markers:
  - **Buy**: Green upward arrow at candle low
  - **Sell**: Red downward arrow at candle high
  - **Close**: Neutral dot or X at price midpoint
  - **Wait**: Optional grey vertical line
- On hover: Show full tooltip (timestamp, OHLC, volume, reward at that step)

---

### 2. Reward Timeline (Directly Below Main Chart)
- Aligned X-axis with candlestick chart
- Display as line or bar chart:
  - Green bars = positive reward
  - Red bars = negative reward
- Optional overlays:
  - Entry/hold/close opportunity scores
  - Drawdown penalty annotations
- Tooltip breakdown:
Reward = +0.42
+0.72 (entry)
-0.30 (drawdown penalty)


---

### 3. Action Info Sidebar (Right Panel or Popover)
- At current step (based on mouse cursor) or latest candle:
- **Action**: WAIT / ENTRY / HOLD / CLOSE
- **Position size**
- **Leverage**
- **Percent closed**
- **Entry/hold/close opportunity score**
- **Drawdown info**
- **Reward components (decomposed)**

---

### 4. Playback Controls (Top or Bottom UI)
- Step-by-step view
- Adjustable playback speed (e.g., 100ms / 500ms / 1s)
- Pause / Play toggle
- Jump to timestamp or step index
- Option to loop over specific window (range selector)

---

### 5. Enhanced Visual Cues
- **Background shading** when drawdown exceeds 50% of limit
- **Color-coded markers** for critical conditions (e.g., entering at max risk)
- **Reward justification popup** for selected step

---

## File Requirements

### Input Data Format
JSON or structured log per episode:
```json
[
{
  "timestamp": "2023-01-01 09:00",
  "open": 100.0,
  "high": 101.0,
  "low": 99.5,
  "close": 100.8,
  "action": "ENTRY",
  "position_size": 0.3,
  "leverage": 2.0,
  "percent_closed": 0.0,
  "reward": 0.42,
  "entry_score": 2.1,
  "hold_score": null,
  "close_score": null,
  "drawdown_penalty": -0.1
},
...
]
```

### 6. Target Outcome
Enable fast, intuitive review of reward system effectiveness. The user should be able to:

- Spot reward inconsistencies
- Validate the alignment of actions with opportunity scores
- Observe drawdown stress zones and behavior shifts
- Trust that reward feedback is doing what it’s meant to
