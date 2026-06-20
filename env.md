Below is a tightened v2 spec you can hand off directly. It incorporates the implementation issues: no bid ambiguity, one fixed action space, seat-relative observations, explicit phase transitions, and configurable reward shaping.

---

# Spades+ Classic Environment Specification v2

Implement a turn-based multi-agent Spades environment matching the user’s observed rules from the **Spades+** app. The environment should support RL/self-play training with partial observability, legal-action masking, deterministic scoring, reproducible seeding, and clean wrappers for PettingZoo/PufferLib-style training.

The first implementation priority is **correctness and testability**, not optimization.

---

## 1. Game Overview

This is a 4-player partnership trick-taking card game.

```python
NUM_PLAYERS = 4
TEAMS = ((0, 2), (1, 3))
TARGET_SCORE = 250
```

Players sit clockwise:

```text
0 -> 1 -> 2 -> 3 -> 0
```

Partners sit opposite each other:

```text
Player 0 partners with Player 2
Player 1 partners with Player 3
```

A match consists of repeated hands until a team reaches the target score.

The environment is a **team game**, but contracts are scored **individually**.

That means:

```text
Each player bids for themselves.
Each player is scored based on their own bid and own tricks.
Team score delta = sum of both partners’ individual score deltas.
```

Do **not** combine partner bids into one team contract.

---

## 2. Core Rules Confirmed from Spades+

Use these as the default rules:

```python
target_score = 250

nil_enabled = True
nil_bonus = 100
nil_penalty = -100

blind_nil_enabled = True
blind_nil_bonus = 200
blind_nil_penalty = -200
blind_nil_always_allowed = True
blind_nil_pass_count = 0

bags_enabled = True
bag_threshold = 10
bag_penalty = -100

total_table_bid_restriction = None  # table bid can be anything

dealer_rotates = True
first_bidder = "left_of_dealer"
first_trick_leader = "left_of_dealer"

spades_must_be_broken_to_lead = True
allow_lead_spades_if_only_spades = True

game_ends_when_team_reaches_target = True
if_both_teams_reach_target_same_hand = "higher_score_wins"
if_tied_at_or_above_target = "continue"
```

---

## 3. Card Model

Use a standard 52-card deck.

Recommended suit encoding:

```python
CLUBS = 0
DIAMONDS = 1
HEARTS = 2
SPADES = 3
```

Recommended rank encoding:

```python
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
```

Rank order is ascending from `2` to `A`.

Recommended card ID encoding:

```python
card_id = suit * 13 + rank_index
```

Examples:

```text
0  = 2 of clubs
12 = A of clubs
13 = 2 of diamonds
25 = A of diamonds
26 = 2 of hearts
38 = A of hearts
39 = 2 of spades
51 = A of spades
```

Required helper functions:

```python
def card_suit(card_id: int) -> int:
    return card_id // 13

def card_rank(card_id: int) -> int:
    return card_id % 13

def is_spade(card_id: int) -> bool:
    return card_suit(card_id) == SPADES

def card_name(card_id: int) -> str:
    ...
```

Spades are always trump.

---

## 4. Game / Hand Flow

A match is made up of repeated hands.

Each hand has this flow:

```text
1. Determine dealer.
2. Shuffle and deal 13 cards to each player.
3. Bidding phase.
4. Playing phase: 13 tricks.
5. Score hand.
6. Apply bag penalties.
7. Update total team scores.
8. Check terminal condition.
9. Rotate dealer.
```

Dealer rotates clockwise after each hand:

```python
dealer = (dealer + 1) % 4
```

First bidder is the player left of dealer:

```python
first_bidder = (dealer + 1) % 4
```

First trick leader is also the player left of dealer:

```python
first_leader = (dealer + 1) % 4
```

Important implementation detail:

After all four bids are complete, reset:

```python
current_player = first_leader
leader = first_leader
phase = PLAYING
```

Do **not** let bidding order accidentally determine the first player of the play phase.

---

## 5. Dealing

Each hand deals 13 cards to each player.

It is acceptable to simulate dealing as one shuffled deck split into four hands:

```python
deck = np.arange(52)
rng.shuffle(deck)

hands[0] = deck[0:13]
hands[1] = deck[13:26]
hands[2] = deck[26:39]
hands[3] = deck[39:52]
```

The app may visually deal one-by-one, but for simulation this does not matter as long as:

```text
- every player gets 13 cards
- no card is duplicated
- all 52 cards are used
- seeded deals are reproducible
```

Recommended: sort hands internally for deterministic display/debugging, but preserve hidden information rules.

---

## 6. Bidding Rules

Each player makes one bid during the bidding phase.

Allowed bid types:

```text
Normal bid: 1 through 13
Nil
Blind Nil
```

Do **not** implement a separate normal bid of `0`.

Treat `0 tricks` as `NIL`, not as a normal numeric bid.

Recommended bid representation:

```python
from dataclasses import dataclass
from enum import Enum

class BidKind(Enum):
    NORMAL = "normal"
    NIL = "nil"
    BLIND_NIL = "blind_nil"

@dataclass(frozen=True)
class Bid:
    kind: BidKind
    value: int = 0
```

Validation:

```python
NORMAL: value in [1, 13]
NIL: value == 0
BLIND_NIL: value == 0
```

Bidding order:

```python
bid_order = [
    (dealer + 1) % 4,
    (dealer + 2) % 4,
    (dealer + 3) % 4,
    dealer,
]
```

After four bids, transition to play phase.

---

## 7. Fixed Action Space

Use one fixed discrete action space for both bidding and card play.

Recommended action space:

```python
Discrete(67)
```

Action mapping:

```python
# Card play actions
0..51      -> play card_id 0..51

# Bid actions
52..64     -> normal bids 1..13
65         -> NIL
66         -> BLIND_NIL
```

Helper functions:

```python
CARD_ACTION_START = 0
CARD_ACTION_END = 51

BID_ACTION_START = 52
NORMAL_BID_ACTION_START = 52
NORMAL_BID_ACTION_END = 64

NIL_ACTION = 65
BLIND_NIL_ACTION = 66

ACTION_SPACE_SIZE = 67
```

Normal bid conversion:

```python
def action_to_normal_bid(action: int) -> int:
    assert 52 <= action <= 64
    return action - 51
```

Examples:

```text
52 -> bid 1
53 -> bid 2
...
64 -> bid 13
65 -> NIL
66 -> BLIND_NIL
```

During bidding, only actions `52..66` can be legal.

During play, only actions `0..51` can be legal, and only if the corresponding card is in the acting player’s hand and legal under trick-taking rules.

The action mask must always have shape:

```python
mask.shape == (67,)
```

with `1` or `True` for legal actions and `0` or `False` for illegal actions.

---

## 8. Phases

Use explicit phases.

```python
from enum import Enum

class Phase(Enum):
    BIDDING = "bidding"
    PLAYING = "playing"
    HAND_OVER = "hand_over"
    GAME_OVER = "game_over"
```

Recommended internal state fields:

```python
phase: Phase
dealer: int
current_player: int
leader: int
bid_count: int
trick_number: int
turn_index_within_trick: int
spades_broken: bool
```

Expected phase transitions:

```text
reset match -> BIDDING
4 bids complete -> PLAYING
13 tricks complete -> HAND_OVER
score hand -> either BIDDING for next hand or GAME_OVER
```

For RL APIs, `HAND_OVER` can be an internal transient phase. It may immediately score and transition to the next hand or game over inside `step`.

---

## 9. Legal Bidding Actions

During bidding:

```python
legal_actions = list(range(52, 67))
```

That means:

```text
52..64 = normal bids 1..13
65 = NIL
66 = BLIND_NIL
```

Because Spades+ allows blind nil always, no score-based restriction is needed.

If future variants require restricting blind nil, add config:

```python
blind_nil_policy: Literal["always", "behind_100", "behind_200", "disabled"] = "always"
```

For this Spades+ env, default:

```python
blind_nil_policy = "always"
```

---

## 10. Legal Card Play Rules

Each trick has four card plays.

The leader plays first. Other players act clockwise.

Players must follow suit if possible.

If a player cannot follow suit, they may play any card.

Spades are trump.

The highest spade wins if any spades are played. Otherwise, the highest card of the led suit wins.

### Spades-Broken Rule

Spades cannot be led until broken.

Spades are broken when a player plays a spade on a trick where spades were **not** the led suit.

Exception:

```text
If the leader has only spades left, they may lead spades even if spades have not been broken.
```

### Legal Card Function

```python
def legal_card_ids(state, player: int) -> list[int]:
    hand = state.hands[player]

    # Leading a trick
    if len(state.current_trick) == 0:
        if state.spades_broken:
            return list(hand)

        non_spades = [c for c in hand if not is_spade(c)]

        if len(non_spades) > 0:
            return non_spades

        # Player has only spades.
        return list(hand)

    # Following a trick
    led_card = state.current_trick[0].card_id
    led_suit = card_suit(led_card)

    follow_suit_cards = [c for c in hand if card_suit(c) == led_suit]

    if len(follow_suit_cards) > 0:
        return follow_suit_cards

    return list(hand)
```

### Legal Action Mask During Play

```python
mask = np.zeros(67, dtype=np.bool_)

for card_id in legal_card_ids(state, current_player):
    mask[card_id] = True
```

All bid actions must be masked out during play.

---

## 11. Trick Resolution

Represent the current trick as ordered plays:

```python
@dataclass
class TrickPlay:
    player: int
    card_id: int

current_trick: list[TrickPlay]
```

The first card determines the led suit:

```python
led_suit = card_suit(current_trick[0].card_id)
```

Winner logic:

```python
def resolve_trick_winner(current_trick: list[TrickPlay]) -> int:
    spade_plays = [
        play for play in current_trick
        if is_spade(play.card_id)
    ]

    if spade_plays:
        winning_play = max(spade_plays, key=lambda play: card_rank(play.card_id))
        return winning_play.player

    led_suit = card_suit(current_trick[0].card_id)

    led_suit_plays = [
        play for play in current_trick
        if card_suit(play.card_id) == led_suit
    ]

    winning_play = max(led_suit_plays, key=lambda play: card_rank(play.card_id))
    return winning_play.player
```

After resolving a trick:

```python
winner = resolve_trick_winner(current_trick)
player_tricks[winner] += 1
leader = winner
current_player = winner
current_trick = []
trick_number += 1
turn_index_within_trick = 0
```

If fewer than 13 tricks have been completed, the winner leads the next trick.

If 13 tricks have been completed, score the hand.

---

## 12. Spades Broken Update

After every card play, update `spades_broken`.

Spades are broken when a spade is played while spades are not the led suit.

Implementation detail:

```python
def update_spades_broken_after_play(state, played_card: int):
    if state.spades_broken:
        return

    if not is_spade(played_card):
        return

    # If this is the first card of the trick, it is a lead.
    # Leading spades when only spades are held should not be considered
    # "breaking" by off-suit trumping; however, after this point spades
    # can effectively be considered broken for future tricks.
    if len(state.current_trick) == 0:
        state.spades_broken = True
        return

    led_suit = card_suit(state.current_trick[0].card_id)

    if led_suit != SPADES:
        state.spades_broken = True
```

Note: In practice, if a player legally leads spades because they only have spades left, the game can safely set `spades_broken = True` afterward. There is no downside for future legality.

Be careful about call order: if `current_trick` already includes the newly played card, detect whether this play was the first card before appending, or pass `was_lead` into the function.

---

## 13. Scoring

Track these per hand:

```python
player_bids: list[Bid]           # length 4
player_tricks: np.ndarray        # shape (4,)
player_score_delta: np.ndarray   # shape (4,)
bags_gained_by_player: np.ndarray # shape (4,)
```

Track these per match:

```python
team_scores: np.ndarray  # shape (2,)
team_bags: np.ndarray    # shape (2,)
```

Team mapping:

```python
def player_team(player: int) -> int:
    return 0 if player in (0, 2) else 1
```

### Normal Bid Scoring

For a normal bid `b` and tricks taken `t`:

```python
if t >= b:
    player_score = 10 * b + (t - b)
    bags_gained = t - b
else:
    player_score = -10 * b
    bags_gained = 0
```

Examples:

```text
Bid 4, take 4 -> +40, 0 bags
Bid 4, take 6 -> +42, 2 bags
Bid 4, take 3 -> -40, 0 bags
```

### Nil Scoring

For `NIL`:

```python
if t == 0:
    player_score = +100
    bags_gained = 0
else:
    player_score = -100
    bags_gained = t
```

Failed nil tricks become bags for the team.

Failed nil tricks do **not** help the partner make their individual contract.

Example:

```text
Player 0 bids NIL and takes 1 trick.
Player 2, their partner, bids 4 and takes 3 tricks.

Player 0 score = -100
Player 0 bags = 1

Player 2 score = -40
Player 2 bags = 0

Team 0 score delta = -140
Team 0 bags gained = 1
```

### Blind Nil Scoring

For `BLIND_NIL`:

```python
if t == 0:
    player_score = +200
    bags_gained = 0
else:
    player_score = -200
    bags_gained = t
```

Blind nil has no card passing.

Failed blind nil tricks become bags for the team.

### Team Aggregation

After scoring all four players:

```python
team_score_delta = np.zeros(2, dtype=np.int32)
team_bags_gained = np.zeros(2, dtype=np.int32)

for p in range(4):
    team = player_team(p)
    team_score_delta[team] += player_score_delta[p]
    team_bags_gained[team] += bags_gained_by_player[p]
```

Then update bags and apply penalties:

```python
for team in range(2):
    team_bags[team] += team_bags_gained[team]

    while team_bags[team] >= 10:
        team_score_delta[team] -= 100
        team_bags[team] -= 10
```

Then update total score:

```python
team_scores += team_score_delta
```

---

## 14. Game End Logic

After each hand is scored:

```python
reached = [
    team_scores[0] >= target_score,
    team_scores[1] >= target_score,
]
```

Cases:

```python
if not any(reached):
    continue_to_next_hand()

elif reached[0] and not reached[1]:
    winner = 0
    phase = GAME_OVER

elif reached[1] and not reached[0]:
    winner = 1
    phase = GAME_OVER

else:
    # Both teams reached target in same hand.
    if team_scores[0] > team_scores[1]:
        winner = 0
        phase = GAME_OVER
    elif team_scores[1] > team_scores[0]:
        winner = 1
        phase = GAME_OVER
    else:
        # Tied at or above target.
        # Continue until tie breaks.
        continue_to_next_hand()
```

---

## 15. Observation Design

The environment is imperfect-information.

The acting player may observe:

```text
- their own hand
- public bids
- public tricks taken
- current trick
- previously played cards
- team scores
- team bags
- dealer
- leader
- current player
- spades_broken
```

The acting player must **not** observe:

```text
- partner hand
- opponent hands
- hidden card ownership
```

---

## 16. Seat-Relative Observation Encoding

For shared-policy self-play, observations should be seat-relative by default.

From the acting player’s perspective:

```text
relative seat 0 = self
relative seat 1 = left opponent
relative seat 2 = partner
relative seat 3 = right opponent
```

Given acting player `p`, convert absolute player ID to relative seat:

```python
def abs_to_rel(abs_player: int, acting_player: int) -> int:
    return (abs_player - acting_player) % 4
```

Convert relative seat back to absolute player ID:

```python
def rel_to_abs(rel_seat: int, acting_player: int) -> int:
    return (acting_player + rel_seat) % 4
```

Examples for acting player `2`:

```text
abs 2 -> rel 0/self
abs 3 -> rel 1/left opponent
abs 0 -> rel 2/partner
abs 1 -> rel 3/right opponent
```

Use this config flag:

```python
seat_relative_observation: bool = True
```

Default should be:

```python
seat_relative_observation = True
```

For debugging, allow absolute observation:

```python
seat_relative_observation = False
```

But RL agents should normally use seat-relative observations.

---

## 17. Recommended Observation Dict

Return observations for the current acting player.

Recommended structure:

```python
obs = {
    # Phase / turn state
    "phase": int,
    "current_player_rel": int,   # should usually be 0 for acting player
    "dealer_rel": int,
    "leader_rel": int,
    "trick_number": int,         # 0..12
    "turn_index_within_trick": int, # 0..3
    "hand_number": int,

    # Scores from acting player's perspective
    "team_scores": np.array([own_team_score, opp_team_score], dtype=np.int32),
    "team_bags": np.array([own_team_bags, opp_team_bags], dtype=np.int32),

    # Per-seat public state, ordered by relative seat 0..3
    "bid_kind": np.array([kind0, kind1, kind2, kind3], dtype=np.int8),
    "bid_value": np.array([value0, value1, value2, value3], dtype=np.int8),
    "tricks_taken": np.array([t0, t1, t2, t3], dtype=np.int8),

    # Cards
    "own_hand_mask": np.ndarray shape (52,), bool,
    "played_cards_mask": np.ndarray shape (52,), bool,

    # Current trick, order-based
    "current_trick_cards_by_order": np.ndarray shape (4,), int16,
    "current_trick_seats_by_order": np.ndarray shape (4,), int8,

    # Current trick, seat-based
    "current_trick_card_by_rel_seat": np.ndarray shape (4,), int16,

    # Rules
    "spades_broken": bool,

    # Legal actions
    "action_mask": np.ndarray shape (67,), bool,
}
```

Use `-1` for empty card/player slots:

```python
current_trick_cards_by_order = [-1, -1, -1, -1]
current_trick_seats_by_order = [-1, -1, -1, -1]
current_trick_card_by_rel_seat = [-1, -1, -1, -1]
```

Bid kind encoding:

```python
BID_UNKNOWN = 0
BID_NORMAL = 1
BID_NIL = 2
BID_BLIND_NIL = 3
```

Before a player has bid:

```python
bid_kind = BID_UNKNOWN
bid_value = 0
```

For normal bid:

```python
bid_kind = BID_NORMAL
bid_value = 1..13
```

For nil:

```python
bid_kind = BID_NIL
bid_value = 0
```

For blind nil:

```python
bid_kind = BID_BLIND_NIL
bid_value = 0
```

---

## 18. Public History

The env should internally track enough history to support recurrent/transformer policies later.

Recommended history event schema:

```python
@dataclass
class Event:
    type: Literal["bid", "play", "trick_end", "hand_end"]
    player: int | None = None
    action: int | None = None
    card_id: int | None = None
    bid: Bid | None = None
    trick_winner: int | None = None
    hand_index: int | None = None
    trick_index: int | None = None
```

This does not need to be part of the default observation, but should be available in `info` or debug mode.

---

## 19. Rewards

The official environment reward should be sparse and score-based by default.

Default reward mode:

```python
reward_mode = "team_score_delta"
```

At normal intermediate actions:

```python
reward[p] = 0
```

At end of hand:

```python
reward[p] = team_score_delta[player_team(p)]
```

Because this is a team game, partners receive the same team reward.

Example:

```text
Team 0 score delta = +42
Team 1 score delta = -30

Players 0 and 2 receive +42.
Players 1 and 3 receive -30.
```

### Optional Terminal Win Reward

Config:

```python
terminal_win_reward_enabled: bool = False
terminal_win_reward: float = 1.0
```

If enabled at game end:

```python
if player_team(p) == winning_team:
    reward[p] += terminal_win_reward
else:
    reward[p] -= terminal_win_reward
```

For pure score optimization, keep this disabled.

For win-rate-focused training, it may be useful.

---

## 20. Configurable Reward Shaping

Reward shaping should be implemented as a config flag, disabled by default.

```python
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
```

Default:

```python
reward_shaping.enabled = False
```

The official score and all raw scoring details should always be returned in `info`, regardless of reward shaping.

Do not bake shaping into the game rules.

When shaping is enabled, compute:

```python
reward = official_reward + shaped_reward
```

Keep both available separately:

```python
info = {
    "official_reward": ...,
    "shaped_reward": ...,
    "total_reward": ...,
}
```

This allows training experiments without corrupting the canonical environment.

---

## 21. Step API

Recommended core API:

```python
class SpadesPlusEnv:
    def reset(self, seed: int | None = None) -> dict:
        ...

    def current_player(self) -> int:
        ...

    def observe(self, player: int | None = None) -> dict:
        ...

    def legal_actions(self) -> list[int]:
        ...

    def action_mask(self) -> np.ndarray:
        ...

    def step(self, action: int):
        ...
```

Recommended `step` return:

```python
obs, rewards, terminated, truncated, info = env.step(action)
```

Where:

```python
obs
```

is the observation for the next acting player, unless the game is over.

```python
rewards
```

should be shape `(4,)`, one reward per absolute player.

```python
terminated
```

is `True` if the match is over.

```python
truncated
```

is `False` unless an external time/hand limit is added.

```python
info
```

contains scoring/debug details.

For PettingZoo AEC compatibility, wrap this core API rather than making the core env depend on PettingZoo internals.

---

## 22. Illegal Actions

In debug/development mode:

```python
illegal_action_mode = "raise"
```

Illegal actions should raise a clear exception.

Example:

```python
raise IllegalActionError(
    f"Player {player} attempted action {action}, "
    f"but legal actions are {legal_actions}"
)
```

For RL training mode, optionally support:

```python
illegal_action_mode = "mask_only"
```

But the policy should always receive a valid action mask. Do not silently accept illegal actions.

Recommended config:

```python
illegal_action_mode: Literal["raise", "penalize", "ignore"] = "raise"
```

For v1, only `"raise"` is required.

---

## 23. Info Dict

At every step, return useful debug info.

At minimum:

```python
info = {
    "phase": phase,
    "current_player": current_player,
    "legal_actions": legal_actions,
}
```

At trick end, include:

```python
info.update({
    "trick_completed": True,
    "trick_winner": winner,
    "trick_cards": [(player, card_id), ...],
    "player_tricks": player_tricks.copy(),
})
```

At hand end, include:

```python
info.update({
    "hand_completed": True,
    "player_bids": player_bids,
    "player_tricks": player_tricks.copy(),
    "player_score_delta": player_score_delta.copy(),
    "bags_gained_by_player": bags_gained_by_player.copy(),
    "team_score_delta": team_score_delta.copy(),
    "team_scores": team_scores.copy(),
    "team_bags": team_bags.copy(),
    "made_contract": made_contract.copy(),
    "nil_success": nil_success.copy(),
})
```

At game end, include:

```python
info.update({
    "game_completed": True,
    "winning_team": winning_team,
    "final_team_scores": team_scores.copy(),
    "final_team_bags": team_bags.copy(),
})
```

---

## 24. Full-State Debug Access

The environment should support a debug-only full state.

This is useful for tests, search, or oracle bots.

```python
def get_debug_state(self) -> dict:
    return {
        "hands": copy_of_all_hands,
        "deck": ...,
        "dealer": ...,
        "current_player": ...,
        "team_scores": ...,
        "team_bags": ...,
        ...
    }
```

Important:

```text
Normal agents must not receive this state.
RL observations must remain partial-observation only.
```

---

## 25. Determinism and Seeding

The env must support reproducible simulation.

```python
env.reset(seed=123)
```

should produce the same initial dealer, shuffled deck, and dealt hands every time.

Use a centralized RNG:

```python
self.rng = np.random.default_rng(seed)
```

Do not use global randomness.

Bad:

```python
np.random.shuffle(deck)
random.shuffle(deck)
```

Good:

```python
self.rng.shuffle(deck)
```

If the same seed and same action sequence are replayed, the full match trajectory should be identical.

---

## 26. Config Object

Recommended top-level config:

```python
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
```

If `first_dealer is None`, choose randomly at match reset.

For deterministic tests, allow explicit first dealer:

```python
config.first_dealer = 3
```

---

## 27. PettingZoo / PufferLib Compatibility

The core env should be library-agnostic.

Then provide wrappers.

Suggested structure:

```text
spades_plus/
  __init__.py
  cards.py
  config.py
  state.py
  rules.py
  scoring.py
  observations.py
  env.py
  wrappers/
    pettingzoo_aec.py
    pufferlib_env.py
  bots.py
  tests/
```

The core env should expose:

```python
Discrete(67)
```

and action masks of shape:

```python
(67,)
```

This is important for PPO/PufferLib-style training.

For PettingZoo AEC:

```text
agent_0 -> agent_1 -> agent_2 -> agent_3 -> ...
```

but the actual acting order should be determined by the Spades rules:

```text
bidding starts left of dealer
play starts left of dealer
trick winner leads next trick
```

The PettingZoo wrapper should map:

```python
agent_name = f"player_{player_id}"
```

Rewards should be assigned to all four players at hand end.

---

## 28. Baseline Bots

Implement simple bots for smoke tests.

```python
class RandomLegalBot:
    def act(obs, legal_actions):
        return random legal action

class LowestLegalCardBot:
    def act(obs, legal_actions):
        during bidding: choose a simple legal bid
        during play: play lowest legal card

class HighestLegalCardBot:
    def act(obs, legal_actions):
        during bidding: choose a simple legal bid
        during play: play highest legal card

class SimpleBidBot:
    def estimate_bid(hand):
        rough heuristic based on high cards and spades
```

These bots are not meant to be strong. They are meant to validate full-game execution before RL training.

---

## 29. Required Unit Tests

Write tests before training.

### Card Tests

```text
1. card_suit works for all 52 cards.
2. card_rank works for all 52 cards.
3. is_spade works for card IDs 39..51.
4. card IDs are unique.
```

### Deal Tests

```text
1. Each player receives 13 cards.
2. No duplicate cards.
3. All 52 cards are dealt.
4. Same seed produces same deal.
5. Different seeds usually produce different deals.
```

### Bidding Tests

```text
1. First bidder is left of dealer.
2. Bidding order is clockwise.
3. Normal bid actions 52..64 map to bids 1..13.
4. Action 65 maps to NIL.
5. Action 66 maps to BLIND_NIL.
6. Normal bid 0 is not available.
7. After four bids, phase switches to PLAYING.
8. After bidding, current_player resets to first trick leader, left of dealer.
```

### Legal Play Tests

```text
1. Leader can play any non-spade before spades are broken.
2. Leader cannot lead spades before broken while holding non-spades.
3. Leader can lead spades before broken if holding only spades.
4. Player must follow suit when possible.
5. Player may play any card when void in led suit.
6. Playing a spade off-suit breaks spades.
7. After spades are broken, spades can be led.
```

### Trick Resolution Tests

```text
1. Highest led suit wins when no spades are played.
2. Highest spade wins when any spades are played.
3. Off-suit non-spade cannot win.
4. Trick winner receives one trick.
5. Trick winner leads next trick.
6. After four cards, current trick resets.
```

### Scoring Tests

```text
1. Normal bid exactly made.
2. Normal bid overmade gives bags.
3. Normal bid failed gives negative bid value.
4. Nil success gives +100.
5. Nil failure gives -100.
6. Nil failure tricks become bags.
7. Blind nil success gives +200.
8. Blind nil failure gives -200.
9. Blind nil failure tricks become bags.
10. Partner bids are scored individually.
11. Team score delta is sum of partner score deltas.
12. 10 bags causes -100 and removes 10 bags.
13. Multiple bag penalties are handled.
```

### Game End Tests

```text
1. Game continues below 250.
2. Game ends when one team reaches 250.
3. If both teams reach 250, higher score wins.
4. If both teams are tied at or above 250, game continues.
5. Dealer rotates after each non-terminal hand.
```

### Observation Tests

```text
1. Acting player sees own hand.
2. Acting player does not see partner hand.
3. Acting player does not see opponent hands.
4. Played cards are visible.
5. Bids are visible after made.
6. Seat-relative encoding maps self to relative seat 0.
7. Partner maps to relative seat 2.
8. Action mask shape is always 67.
9. During bidding, only bid actions are legal.
10. During play, only legal card actions are legal.
```

---

## 30. Canonical Scoring Examples

### Example 1: Failed Nil and Failed Partner Bid

```text
Team 0: players 0 and 2

Player 0 bids NIL and takes 1 trick.
Player 2 bids 4 and takes 3 tricks.
Team 0 had 0 bags before the hand.
```

Scoring:

```python
player_0_score = -100
player_0_bags = 1

player_2_score = -40
player_2_bags = 0

team_0_delta = -140
team_0_bags = 1
```

### Example 2: Normal Overbid and Blind Nil Success

```text
Team 0: players 0 and 2

Player 0 bids 3 and takes 5 tricks.
Player 2 bids BLIND_NIL and takes 0 tricks.
Team 0 had 0 bags before the hand.
```

Scoring:

```python
player_0_score = 10 * 3 + (5 - 3) = 32
player_0_bags = 2

player_2_score = 200
player_2_bags = 0

team_0_delta = 232
team_0_bags = 2
```

### Example 3: Bag Penalty

```text
Team 1 starts hand with 8 bags.

Player 1 bids 2 and takes 4 tricks.
Player 3 bids 3 and takes 4 tricks.
```

Player scoring:

```python
player_1_score = 22
player_1_bags = 2

player_3_score = 31
player_3_bags = 1
```

Before penalty:

```python
team_1_delta = 53
team_1_bags = 8 + 2 + 1 = 11
```

Apply bag penalty:

```python
team_1_delta = 53 - 100 = -47
team_1_bags = 1
```

Final:

```python
team_1_score_delta = -47
team_1_bags = 1
```

---

## 31. Non-Goals for v1

Do not implement these initially:

```text
- UI automation
- playing through the real Spades+ app
- vision/card recognition
- chat or partner communication
- solo mode
- mirror mode
- whiz mode
- card passing
- table customization
- social features
- matchmaking
- reneging penalties
```

Illegal moves should be impossible through the env, so reneging penalties are unnecessary in v1.

---

## 32. Implementation Priority

Recommended implementation order:

```text
1. cards.py
2. config.py
3. state.py
4. rules.py legal action logic
5. trick resolution
6. scoring.py
7. env.py hand flow
8. observation encoding
9. action masks
10. deterministic seeding
11. unit tests
12. simple bots
13. PettingZoo wrapper
14. PufferLib wrapper
```

Do not start RL training until the rule/scoring tests are passing.

The v1 goal is:

```text
A correct, reproducible, partially observable, seat-relative,
fixed-action-space Spades+ Classic simulator suitable for self-play RL.
```

