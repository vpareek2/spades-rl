"""Training-facing wrappers."""

from spades.wrappers.gymnasium_env import SpadesGymnasiumEnv
from spades.wrappers.pettingzoo_aec import SpadesAECEnv

__all__ = ["SpadesAECEnv", "SpadesGymnasiumEnv"]
