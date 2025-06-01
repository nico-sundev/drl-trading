import numpy as np
import shap
from sklearn.feature_selection import VarianceThreshold


def evaluate_features_with_drl(feature_sets, env, agent, episodes=10):
    """
    Evaluate different feature sets using DRL agent performance.

    Args:
        feature_sets: List of feature column sets to evaluate
        env: DRL environment
        agent: DRL agent
        episodes: Number of episodes to evaluate each feature set

    Returns:
        List of feature names that performed best
    """
    best_reward = float("-inf")
    best_features = None

    for features in feature_sets:
        total_reward = 0
        env.reset()  # Reset environment with new feature set
        env.set_active_features(features)  # Assuming environment has this method

        for _ in range(episodes):
            done = False
            state = env.reset()
            episode_reward = 0

            while not done:
                action = agent.predict(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward

        avg_reward = total_reward / episodes
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_features = features

    return best_features


class DRLFeatureSelector:
    def __init__(self, env, agent, threshold=0.9):
        """
        Feature selection for DRL.
        :param env: DRL environment
        :param agent: DRL agent (e.g., PPO, DQN)
        :param threshold: Correlation threshold for filtering.
        """
        self.env = env
        self.agent = agent
        self.threshold = threshold

    def remove_low_variance(self, X):
        """Filter features with near-zero variance."""
        selector = VarianceThreshold(threshold=0.01)
        return X.loc[:, selector.fit(X).get_support()]

    def remove_correlated_features(self, X):
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]
        return X.drop(columns=to_drop)

    def evaluate_feature_subsets(self, X, num_trials=3):
        """Train agent with different feature subsets and return the best one."""
        feature_sets = [
            list(X.sample(frac=0.8, axis=1).columns) for _ in range(num_trials)
        ]
        best_features = evaluate_features_with_drl(feature_sets, self.env, self.agent)
        return X[best_features]

    def compute_shap_importance(self, X):
        """Use SHAP to analyze feature importance post-training."""
        explainer = shap.Explainer(self.agent.predict, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)

    def select_features(self, X, method="hybrid"):
        """Apply feature selection before training DRL."""
        if method == "filter":
            X_filtered = self.remove_low_variance(X)
            return self.remove_correlated_features(X_filtered)
        elif method == "wrapper":
            return self.evaluate_feature_subsets(X)
        elif method == "embedded":
            self.compute_shap_importance(X)
            return X  # Features are not removed but analyzed
        elif method == "hybrid":
            X_filtered = self.remove_low_variance(X)
            X_filtered = self.remove_correlated_features(X_filtered)
            return self.evaluate_feature_subsets(X_filtered)
        else:
            raise ValueError(
                "Invalid method. Choose 'filter', 'wrapper', 'embedded', or 'hybrid'."
            )
