import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import VarianceThreshold

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
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return X.drop(columns=to_drop)

    def evaluate_feature_subsets(self, X, num_trials=3):
        """Train agent with different feature subsets and return the best one."""
        feature_sets = [list(X.sample(frac=0.8, axis=1).columns) for _ in range(num_trials)]
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
            raise ValueError("Invalid method. Choose 'filter', 'wrapper', 'embedded', or 'hybrid'.")
