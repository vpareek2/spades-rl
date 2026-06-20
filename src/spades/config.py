"""Environment configuration."""

from dataclasses import dataclass, field


@dataclass
class RewardShapingConfig:
    enabled: bool = False
    trick_win_reward: float = 0.0
    trick_loss_penalty: float = 0.0
    made_contract_reward: float = 0.0
    failed_contract_penalty: float = 0.0
    nil_success_reward: float = 0.0
    nil_failure_penalty: float = 0.0
    blind_nil_success_reward: float = 0.0
    blind_nil_failure_penalty: float = 0.0
    bag_penalty_risk: float = 0.0
    bag_gained_penalty: float = 0.0
    set_opponent_reward: float = 0.0


@dataclass
class SpadesPlusConfig:
    num_players: int = 4
    teams: tuple[tuple[int, int], tuple[int, int]] = ((0, 2), (1, 3))
    target_score: int = 250

    nil_enabled: bool = True
    nil_bonus: int = 100
    nil_penalty: int = -100

    blind_nil_enabled: bool = True
    blind_nil_bonus: int = 200
    blind_nil_penalty: int = -200
    blind_nil_policy: str = "always"
    blind_nil_pass_count: int = 0

    bags_enabled: bool = True
    bag_threshold: int = 10
    bag_penalty: int = -100

    spades_must_be_broken_to_lead: bool = True
    allow_lead_spades_if_only_spades: bool = True

    dealer_rotates: bool = True
    first_dealer: int | None = None

    seat_relative_observation: bool = True

    reward_mode: str = "team_score_delta"
    terminal_win_reward_enabled: bool = False
    terminal_win_reward: float = 1.0
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)

    illegal_action_mode: str = "raise"
    debug: bool = False

    def __post_init__(self) -> None:
        if self.num_players != 4:
            raise ValueError("SpadesPlusEnv supports exactly 4 players")
        if self.teams != ((0, 2), (1, 3)):
            raise ValueError("v1 supports fixed Spades teams: ((0, 2), (1, 3))")
        if self.blind_nil_policy not in {"always", "disabled"}:
            raise ValueError("v1 supports blind_nil_policy values 'always' and 'disabled'")
        if self.illegal_action_mode != "raise":
            raise ValueError("v1 supports only illegal_action_mode='raise'")
        if self.first_dealer is not None and self.first_dealer not in range(4):
            raise ValueError("first_dealer must be None or an integer in [0, 3]")
