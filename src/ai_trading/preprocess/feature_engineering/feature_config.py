class MACDConfig:
    """Configuration for the MACD indicator."""
    
    def __init__(self, fast=3, slow=6, signal=4):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def to_dict(self):
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal}


class RSIConfig:
    """Configuration for the RSI indicator."""
    
    def __init__(self, lengths=None):
        self.lengths = lengths or [14, 3]

    def to_dict(self):
        return {"lengths": self.lengths}


class ROCConfig:
    """Configuration for the ROC indicator."""
    
    def __init__(self, lengths=None):
        self.lengths = lengths or [14, 7, 3]

    def to_dict(self):
        return {"lengths": self.lengths}


class FeatureConfig:
    """Master configuration class for all features."""
    
    def __init__(self):
        self.macd = MACDConfig()
        self.rsi = RSIConfig()
        self.roc = ROCConfig()

    def to_dict(self):
        return {
            "MACD": self.macd.to_dict(),
            "RSI": self.rsi.to_dict(),
            "ROC": self.roc.to_dict(),
        }