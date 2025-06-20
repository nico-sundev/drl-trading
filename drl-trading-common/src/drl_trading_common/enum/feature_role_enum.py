from enum import Enum


class FeatureRoleEnum(Enum):
    """
    Enum for the role or usage of a feature in the trading environment.
    """

    # Context-related features that are applied for reward engineering of the trading environment.
    REWARD_ENGINEERING = "reward_engineering"

    # Features that will be part of observation space of the trading environment.
    OBSERVATION_SPACE = "observation_space"
