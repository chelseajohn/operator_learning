from .data_loss import LpLoss, VectorNormLoss, LOSSES_CLASSES as DATA_LOSSES

LOSSES_CLASSES = {**DATA_LOSSES}

__all__ = ["LpLoss",
            "VectorNormLoss",
        ]