"""Plain terminal REPL for manually trying the Spades environment."""

from __future__ import annotations

import shlex

from spades.actions import BLIND_NIL_ACTION, NIL_ACTION
from spades.cards import card_name
from spades.config import SpadesPlusConfig
from spades.env import IllegalActionError, SpadesPlusEnv


HELP = """Commands:
  reset [seed]        Reset the match
  obs                 Show compact current observation
  hand                Show current player's hand
  legal               Show legal actions
  bid <1-13|nil|blind_nil>
  play <card_id>
  scores              Show team scores and bags
  trick               Show current trick
  history             Show recent events
  help
  quit
"""


def _format_cards(cards: list[int]) -> str:
    return " ".join(f"{card}:{card_name(card)}" for card in cards)


def _print_turn(env: SpadesPlusEnv) -> None:
    print(f"phase={env.phase.value} player={env.current_player()} legal={env.legal_actions()}")


def _handle_command(env: SpadesPlusEnv, line: str) -> bool:
    parts = shlex.split(line)
    if not parts:
        return True

    command = parts[0].lower()
    if command in {"quit", "exit"}:
        return False
    if command == "help":
        print(HELP)
    elif command == "reset":
        seed = int(parts[1]) if len(parts) > 1 else None
        env.reset(seed=seed)
        print(f"reset seed={seed}")
    elif command == "obs":
        obs = env.observe()
        print(
            f"phase={obs['phase']} current_rel={obs['current_player_rel']} "
            f"dealer_rel={obs['dealer_rel']} leader_rel={obs['leader_rel']} "
            f"trick={obs['trick_number']} turn={obs['turn_index_within_trick']}"
        )
    elif command == "hand":
        hand = env.get_debug_state()["hands"][env.current_player()]
        print(_format_cards(hand))
    elif command == "legal":
        print(env.legal_actions())
    elif command == "bid":
        if len(parts) != 2:
            raise ValueError("Usage: bid <1-13|nil|blind_nil>")
        value = parts[1].lower()
        if value == "nil":
            action = NIL_ACTION
        elif value in {"blind_nil", "blind", "bn"}:
            action = BLIND_NIL_ACTION
        else:
            bid = int(value)
            if not 1 <= bid <= 13:
                raise ValueError("Normal bid must be in [1, 13]")
            action = bid + 51
        _, rewards, terminated, _, info = env.step(action)
        print(f"bid action={action} rewards={rewards.tolist()} terminated={terminated}")
        if info.get("hand_completed"):
            print(f"hand score delta={info['team_score_delta'].tolist()}")
    elif command == "play":
        if len(parts) != 2:
            raise ValueError("Usage: play <card_id>")
        action = int(parts[1])
        _, rewards, terminated, _, info = env.step(action)
        print(f"played {action}:{card_name(action)} rewards={rewards.tolist()} terminated={terminated}")
        if info.get("trick_completed"):
            print(f"trick winner={info['trick_winner']} cards={info['trick_cards']}")
        if info.get("hand_completed"):
            print(f"hand score delta={info['team_score_delta'].tolist()}")
    elif command == "scores":
        state = env.get_debug_state()
        print(f"scores={state['team_scores'].tolist()} bags={state['team_bags'].tolist()}")
    elif command == "trick":
        trick = env.get_debug_state()["current_trick"]
        print([(play.player, play.card_id, card_name(play.card_id)) for play in trick])
    elif command == "history":
        for event in env.get_debug_state()["history"][-10:]:
            print(event)
    else:
        raise ValueError(f"Unknown command: {command}")

    _print_turn(env)
    return True


def main() -> None:
    env = SpadesPlusEnv(SpadesPlusConfig(debug=True))
    env.reset(seed=0)
    print("Spades REPL. Type help for commands.")
    _print_turn(env)
    while True:
        try:
            line = input("spades> ")
        except EOFError:
            print()
            break
        try:
            if not _handle_command(env, line):
                break
        except (IllegalActionError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    main()
