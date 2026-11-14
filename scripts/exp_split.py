"""Does choosing the split (text share of the lock-box package) lower suspicion?

Needs text measurements at several fragment sizes, e.g. (GPT-2, about 1.5 h):
    python scripts/exp_text.py --model gpt2 --variant filter --cs 0 --per-cell 4 --payload 64  --out results_gpt2/len64
    python scripts/exp_text.py --model gpt2 --variant filter --cs 0 --per-cell 4 --payload 160 --out results_gpt2/len160
    python scripts/add_style_detector.py --results results_gpt2/len64
    python scripts/add_style_detector.py --results results_gpt2/len160
then:
    python scripts/exp_split.py --results results_gpt2 --sizes 24:results_gpt2,64:results_gpt2/len64,160:results_gpt2/len160

Strategies (all with C = 0; cover and prompt chosen as in the heuristic unless the
strategy chooses them itself):
  min-text   smallest measured text size (the rule the system uses for long secrets)
  max-text   largest measured text size
  random     uniform random cover, prompt and size
  grid-A / grid-ens   try every (cover, prompt, size) against warden A / the ensemble
  ppo-A / ppo-ens     scoring policy (cover and prompt weights + a size knob) trained
                      with PPO against warden A / the ensemble
Evaluated under warden A, warden B (never used in training) and the ensemble.
Writes <results>/split_summary.json and split_episodes.csv.
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
from cmstego.selection import split as SP  # noqa: E402

TRAIN_PROMPTS = list(range(0, 8))
TEST_PROMPTS = list(range(8, 12))


def bootstrap(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_gpt2", help="folder with the image tables and the output")
    ap.add_argument("--covers", default="data/covers")
    ap.add_argument("--sizes", required=True, help="size:folder pairs, e.g. 24:results_gpt2,64:results_gpt2/len64")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--ppo-steps", type=int, default=60000)
    args = ap.parse_args()
    t0 = time.time()
    size_dirs = {int(a): b for a, b in (x.split(":", 1) for x in args.sizes.split(","))}
    base = P.build_tables(args.results, args.covers, text_tag="text_filter_")
    S = SP.build_split_tables(base, size_dirs)
    wardens = [w for w in ("A", "B") if w in S.wardens]
    ens = all(w in S.wardens for w in P.ENSEMBLE)
    if ens:
        wardens.append("ENS")
    print(f"sizes {S.sizes} (mean tokens {[round(S.tokens[s]) for s in S.sizes]}), wardens {wardens} "
          f"({time.time()-t0:.0f}s)")

    from stable_baselines3 import PPO
    policies, train_s = {}, {}
    for w in (["A", "ENS"] if ens else ["A"]):
        env = SP.make_env(S, "train", TRAIN_PROMPTS, seed=1, warden=w)
        model = PPO("MlpPolicy", env, n_steps=512, batch_size=128, learning_rate=3e-4, ent_coef=0.0,
                    gamma=0.0, seed=1, verbose=0)
        t1 = time.time()
        model.learn(total_timesteps=args.ppo_steps)
        train_s[w] = time.time() - t1
        policies[w] = model
        print(f"PPO vs {w} trained in {train_s[w]:.0f}s")

    def ppo(w):
        def f(S_, ep, rng):
            a, _ = policies[w].predict(SP.observation(S_, ep), deterministic=True)
            return (*SP.choice_from_scores(S_, ep, a), 1)
        return f

    strategies = {"min-text": SP.strat_min_text, "max-text": SP.strat_max_text, "random": SP.strat_random,
                  "grid-A": SP.strat_grid, "ppo-A": ppo("A")}
    if ens:
        strategies["grid-ens"] = lambda S_, ep, rng: SP.strat_grid(S_, ep, rng, w="ENS")
        strategies["ppo-ens"] = ppo("ENS")

    rows = []
    for split in ("test", "test_B"):
        rng = np.random.default_rng(123)
        episodes = [SP.make_episode(S, rng, split, TEST_PROMPTS) for _ in range(args.episodes)]
        for name, fn in strategies.items():
            srng = np.random.default_rng(7)
            for ep in episodes:
                k, m, s, cost = fn(S, ep, srng)
                for w in wardens:
                    v, flagged = SP.evaluate(S, ep, k, m, s, w)
                    rows.append({"split": split, "strategy": name, "warden": w, "L": ep.L, "size": S.sizes[s],
                                 "suspicion": v, "flagged": int(flagged), "evaluations": cost})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results, "split_episodes.csv"), index=False)
    summary = []
    for (split, name, w), g in df.groupby(["split", "strategy", "warden"], sort=False):
        ms, lo, hi = bootstrap(g["suspicion"])
        fr, flo, fhi = bootstrap(g["flagged"])
        sizes = {int(k): float(v) for k, v in g["size"].value_counts(normalize=True).items()}
        summary.append({"split": split, "strategy": name, "warden": w, "mean_suspicion": ms, "susp_ci": [lo, hi],
                        "flag_rate": fr, "flag_ci": [flo, fhi], "evaluations_per_message": float(g["evaluations"].mean()),
                        "size_share": sizes})
    out = {"sizes": S.sizes, "size_dirs": {str(k): v for k, v in size_dirs.items()},
           "mean_tokens": {str(k): v for k, v in S.tokens.items()}, "message_bytes": SP.SPLIT_MESSAGE_BYTES,
           "episodes_per_split": args.episodes, "ppo_steps": args.ppo_steps, "ppo_train_seconds": train_s,
           "wardens": wardens, "summary": summary}
    with open(os.path.join(args.results, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(pd.DataFrame(summary)[["split", "strategy", "warden", "mean_suspicion", "flag_rate",
                                 "evaluations_per_message"]].round(3).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
