import json

from spades.eval import (
    BotActor,
    evaluate_duplicate,
    generate_deal,
    make_env_for_deal,
)
from spades.bots import ConservativeBidBot, HighestLegalCardBot, LowestLegalCardBot
from spades.state import Phase


def test_duplicate_deal_reproducibility_and_orientation_setup():
    deal = generate_deal(17)
    same_deal = generate_deal(17)

    assert deal == same_deal

    env0 = make_env_for_deal(deal, nil=True, blind_nil=True)
    env1 = make_env_for_deal(deal, nil=True, blind_nil=True)

    assert env0.get_debug_state()["dealer"] == env1.get_debug_state()["dealer"] == deal.dealer
    assert env0.get_debug_state()["hands"] == env1.get_debug_state()["hands"] == deal.hands


def test_bot_actor_respects_filtered_bidding_actions():
    deal = generate_deal(23)
    env = make_env_for_deal(deal, nil=False, blind_nil=False)
    actor = BotActor("bot:conservative", ConservativeBidBot())

    assert env.phase == Phase.BIDDING
    action = actor.act(env.observe(), [52, 53, 54, 55, 56], env.rng)

    assert 52 <= action <= 56


def test_duplicate_eval_same_policy_has_zero_paired_margin():
    actor_a = BotActor("bot:lowest", LowestLegalCardBot())
    actor_b = BotActor("bot:lowest", LowestLegalCardBot())

    metrics = evaluate_duplicate(
        actor_a,
        actor_b,
        hands=4,
        seed=11,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        reward_scale=0.01,
    )

    assert metrics["duplicate_margin_mean"] == 0.0
    assert metrics["illegal_actions"] == {"A": 0, "B": 0}


def test_duplicate_eval_swapping_policies_negates_margin():
    lowest = BotActor("bot:lowest", LowestLegalCardBot())
    highest = BotActor("bot:highest", HighestLegalCardBot())

    ab = evaluate_duplicate(
        lowest,
        highest,
        hands=4,
        seed=31,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        reward_scale=0.01,
    )
    ba = evaluate_duplicate(
        highest,
        lowest,
        hands=4,
        seed=31,
        max_normal_bid=13,
        nil=False,
        blind_nil=False,
        reward_scale=0.01,
    )

    assert ab["duplicate_margin_mean"] == -ba["duplicate_margin_mean"]


def test_duplicate_eval_cli_writes_json(tmp_path, monkeypatch):
    output = tmp_path / "duplicate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "spades-eval-duplicate",
            "--policy-a",
            "bot:lowest",
            "--policy-b",
            "bot:highest",
            "--hands",
            "2",
            "--seed",
            "5",
            "--no-nil",
            "--no-blind-nil",
            "--output",
            str(output),
        ],
    )

    from spades.eval import duplicate_main

    duplicate_main()

    metrics = json.loads(output.read_text())
    assert metrics["hands"] == 2
    assert metrics["policy_a"] == "bot:lowest"
    assert metrics["policy_b"] == "bot:highest"
    assert "duplicate_margin_mean" in metrics
