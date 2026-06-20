"""Spades+ Classic core environment."""

from spades.actions import ACTION_SPACE_SIZE
from spades.env import IllegalActionError, SpadesPlusEnv

__all__ = ["ACTION_SPACE_SIZE", "IllegalActionError", "SpadesPlusEnv"]


def main() -> None:
    env = SpadesPlusEnv()
    obs = env.reset(seed=0)
    print(f"Spades env ready. Current player: {env.current_player()}, obs keys: {len(obs)}")
