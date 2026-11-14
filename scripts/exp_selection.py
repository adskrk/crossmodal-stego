"""Proof-of-concept experiment: does choosing the carriers lower the pooled warden's suspicion?

Strategies, all evaluated on the same held-out episodes (unseen photos and
unseen prompts):
  none        first cover, first prompt, C = 0 (no selection)
  random      uniform random choice
  heuristic   busiest cover (highest HILL texture), highest-entropy prompt, C = 0
  grid-A      try every (cover, prompt, C) against warden A, keep the best
              (expensive: K*M*|C| embeddings + detector runs per message)
  ppo         policy trained with PPO on training photos/prompts, reward from
              warden A; picks from cheap features in one forward pass

Reported for warden A (the training reward) and warden B (never seen in
training): mean pooled suspicion and the fraction of pairs flagged at a 5%
false-alarm rate, with 95% bootstrap intervals, plus carriers evaluated per message.

With --ensemble (needs the style-detector scores from add_style_detector.py),
the policies are also trained against an ensemble of wardens (A, C, D: worst
case) that shares no detector with warden B, and everything is evaluated under
A, B and the ensemble. Results go to selection_ensemble_*.{csv,json}.

Usage:  python scripts/exp_selection.py [--ppo-steps 60000] [--ensemble]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cmstego.selection import carrier as P  # noqa: E402

TRAIN_PROMPTS = list(range(0, 8))
TEST_PROMPTS = list(range(8, 12))


def bootstrap(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--covers", default="data/covers")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--ppo-steps", type=int, default=60000)
    ap.add_argument("--text-tag", default="text_filter_", help="which text variant's measurements to use")
    ap.add_argument("--ensemble", action="store_true", help="also train against the warden ensemble (A, C, D)")
    args = ap.parse_args()
    t0 = time.time()
    T = P.build_tables(args.results, args.covers, text_tag=args.text_tag)
    print(f"tables built: {len(T.covers)} covers, {len(T.text)} held-out stego-texts ({time.time()-t0:.0f}s)")
    if args.ensemble and not all(w in T.wardens for w in P.ENSEMBLE):
        sys.exit("--ensemble needs style-detector scores: run scripts/add_style_detector.py first")

    # ---- PPO ----
    from stable_baselines3 import PPO
    env = P.make_env(T, "train", TRAIN_PROMPTS, seed=1)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=128, learning_rate=3e-4, ent_coef=0.01,
                gamma=0.0, seed=1, verbose=0)
    t1 = time.time()
    model.learn(total_timesteps=args.ppo_steps)
    ppo_train_s = time.time() - t1
    print(f"PPO trained for {args.ppo_steps} steps in {ppo_train_s:.0f}s")

    def strat_ppo(T_, ep, rng):
        a, _ = model.predict(P.observation(T_, ep), deterministic=True)
        return int(a[0]), int(a[1]), int(a[2]), 1

    # PPO v2: scoring policy (see carrier.score_action_to_choice), same budget of steps.
    env2 = P.make_env(T, "train", TRAIN_PROMPTS, seed=1, action="score")
    model2 = PPO("MlpPolicy", env2, n_steps=512, batch_size=128, learning_rate=3e-4, ent_coef=0.0,
                 gamma=0.0, seed=1, verbose=0)
    t1 = time.time()
    model2.learn(total_timesteps=args.ppo_steps)
    ppo2_train_s = time.time() - t1
    print(f"PPO (scoring policy) trained for {args.ppo_steps} steps in {ppo2_train_s:.0f}s")

    def strat_ppo_score(T_, ep, rng):
        a, _ = model2.predict(P.observation(T_, ep), deterministic=True)
        return (*P.score_action_to_choice(T_, ep, a), 1)

    strategies = {"none": P.strat_fixed_first, "random": P.strat_random, "heuristic": P.strat_heuristic,
                  "grid-A": P.strat_grid, "ppo": strat_ppo, "ppo-score": strat_ppo_score}
    wardens = ("A", "B")
    ens_train_s = None
    prefix = "selection_"
    if args.ensemble:
        env3 = P.make_env(T, "train", TRAIN_PROMPTS, seed=1, warden="ENS", action="score")
        model3 = PPO("MlpPolicy", env3, n_steps=512, batch_size=128, learning_rate=3e-4, ent_coef=0.0,
                     gamma=0.0, seed=1, verbose=0)
        t1 = time.time()
        model3.learn(total_timesteps=args.ppo_steps)
        ens_train_s = time.time() - t1
        print(f"PPO (scoring policy) vs ensemble trained for {args.ppo_steps} steps in {ens_train_s:.0f}s")

        def strat_ppo_ens(T_, ep, rng):
            a, _ = model3.predict(P.observation(T_, ep), deterministic=True)
            return (*P.score_action_to_choice(T_, ep, a), 1)

        strategies["grid-ens"] = lambda T_, ep, rng: P.strat_grid(T_, ep, rng, w="ENS")
        strategies["ppo-ens"] = strat_ppo_ens
        wardens = ("A", "B", "ENS")
        prefix = "selection_ensemble_"

    rows = []
    for split in ("test", "test_B"):
        rng = np.random.default_rng(123)
        episodes = [P.make_episode(T, rng, split, TEST_PROMPTS) for _ in range(args.episodes)]
        for name, fn in strategies.items():
            srng = np.random.default_rng(7)
            for ep in episodes:
                k, m, c, cost = fn(T, ep, srng)
                for w in wardens:
                    s, f = P.evaluate(T, ep, k, m, c, w)
                    rows.append({"split": split, "strategy": name, "warden": w, "L": ep.L,
                                 "suspicion": s, "flagged": int(f), "evaluations": cost, "C": P.C_VALUES[c]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results, prefix + "episodes.csv"), index=False)

    summary = []
    for (split, name, w), g in df.groupby(["split", "strategy", "warden"], sort=False):
        ms, lo, hi = bootstrap(g["suspicion"])
        fr, flo, fhi = bootstrap(g["flagged"])
        summary.append({"split": split, "strategy": name, "warden": w,
                        "mean_suspicion": ms, "susp_ci": [lo, hi],
                        "flag_rate": fr, "flag_ci": [flo, fhi],
                        "evaluations_per_message": float(g["evaluations"].mean()),
                        "share_C>0": float((g["C"] > 0).mean())})
    out = {"text_variant": args.text_tag, "episodes_per_split": args.episodes, "ppo_steps": args.ppo_steps, "ppo_train_seconds": ppo_train_s, "ppo_score_train_seconds": ppo2_train_s,
           "message_bytes": P.MESSAGE_BYTES, "K_covers": 8, "M_prompts": 4, "C_values": P.C_VALUES,
           "wardens": {w: P.WARDENS[w] for w in (("A", "B") + (P.ENSEMBLE if args.ensemble else ()))},
           "ensemble": list(P.ENSEMBLE) if args.ensemble else None,
           "ppo_ensemble_train_seconds": ens_train_s, "summary": summary}
    with open(os.path.join(args.results, prefix + "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(pd.DataFrame(summary)[["split", "strategy", "warden", "mean_suspicion", "flag_rate",
                                 "evaluations_per_message", "share_C>0"]].round(3).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
