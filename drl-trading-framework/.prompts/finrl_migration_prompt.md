
# 🧠 Task: Migrate Gym + Stable Baselines3 Setup to FinRL Framework

## 📌 Objective

Migrate the existing Deep Reinforcement Learning (DRL) project currently using **OpenAI Gym environments** and **Stable Baselines3** to use the **[FinRL](https://github.com/AI4Finance-Foundation/FinRL)** framework. Also, set up FinRL's training module using the project's current logic where applicable.

---

## ✅ Migration Steps

### 1. 🧼 Environment Replacement

- Replace the current Gym environment with a **FinRL-compatible custom trading environment**.
- If custom logic exists (e.g., custom reward, action space, or state representation), migrate and adapt it into FinRL’s `StockTradingEnv` or subclass it:
  - Preserve reward function logic.
  - Preserve feature selection and normalization pipeline.
  - Preserve portfolio management or action interpretation logic.

### 2. 🔄 DRL Agent API Migration

- Replace direct usage of `StableBaselines3` with FinRL’s modular training API:
  ```python
  from finrl.agents.stablebaselines3_models import DRLAgent
  agent = DRLAgent(env=env)
  model = agent.get_model("ppo")  # or any other algorithm
  trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=50000)
  ```

- Ensure reproducibility with deterministic seeds if needed.

### 3. 📈 Data & Preprocessing

- Ensure market data pipeline aligns with FinRL’s `data_processor` logic:
  - Use existing dataset or adapt to FinRL's `YahooDownloader`, `BinanceDownloader`, or a custom data source.
  - Confirm compatibility with `config.py` or define your own constants.

- Migrate technical indicator calculations to FinRL's `FeatureEngineer` or keep your custom ones if more complex.

---

## 🧪 Training Module Setup

### Structure Required:

```python
# train.py or finrl_train.py

from finrl.agents.stablebaselines3_models import DRLAgent
from finrl.env.env_stock_trading import StockTradingEnv
from finrl.config import config

# 1. Load and preprocess data
# 2. Instantiate FinRL environment
# 3. Wrap in DummyVecEnv
# 4. Train using DRLAgent class
```

### Example Template:
```python
env = StockTradingEnv(df=your_dataframe, turbulence_threshold=None, ...
                      initial_amount=1000000, ...)

env_train = DummyVecEnv([lambda: env])
agent = DRLAgent(env=env_train)

model = agent.get_model("ppo")
trained_model = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=100_000)
```

---

## ⚠️ Constraints

- ✅ Preserve all core DRL logic (reward, state, actions).
- ❌ Do not refactor agent behavior unless FinRL requires it.
- ✅ Use FinRL-compatible data pipeline but maintain original dataset if needed.
- ✅ Log model training performance and export the final trained model.

---

## 📁 Deliverables

- `finrl_train.py` — Training script using FinRL API.
- `env_custom.py` — Your migrated environment if custom.
- Updated `config.py` if needed (initial capital, state space, indicators).
- Trained model artifacts saved in a dedicated directory (`trained_models/`).
- Optional: A Jupyter notebook to test the trained model (`test_agent.ipynb`).

---

## 📌 Extras (Optional)

- Add early stopping if supported.
- Implement tensorboard logging for easy visualization.
- Add training wrapper script or Makefile for automation.

---

Let me know when the FinRL training module is complete and tested. If issues arise during environment adaptation, please highlight them clearly for review.
