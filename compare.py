"""Run layer_sharing.py, fedagent.py, and flake.py back-to-back with aligned
hyperparameters and print a 3-way side-by-side comparison.

All three scripts read the common ``FL_*`` family of environment variables
(``FL_BATCH_SIZE``, ``FL_EPOCHS_PER_ROUND``, ``FL_DIRICHLET_ALPHA``,
``FL_ROUNDS``) so the harness can align them. ``layer_sharing.py``'s
convergence-based early stopping is also disabled here so it runs the full
``FL_ROUNDS`` budget like the other two.

Each child writes a JSON summary to ``FL_RESULTS_JSON``; the harness reads them
back and prints the comparison table.

Frameworks
----------
* **layer_sharing**: peer-to-peer, **random layer stacking**, no global model,
  no KD. Reports the **average** test accuracy across local clients.
* **fedagent**: **centralized** server + Dueling Double-DQN client selection +
  KD against the broadcast global model. Reports the **single global model's**
  test accuracy.
* **flake**: peer-to-peer, **KD against a locally-aggregated peer "teacher"**
  (FedAvg of own + peers each round). Reports the **average** test accuracy
  across local clients.

Usage::

    python compare.py                 # default: 20 rounds, SimpleCNN on all
    python compare.py --rounds 50     # bigger budget
    python compare.py --skip-flake   # also: --skip-layer-sharing, --skip-fedagent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def _run_child(name: str, cmd: list[str], env_extra: dict, results_path: Path,
               log_path: Path) -> tuple[int, dict | None, float]:
    print(f"\n{'=' * 72}\n[{name}] launching: {' '.join(cmd)}\n{'=' * 72}")
    env = {**os.environ, **env_extra, "FL_RESULTS_JSON": str(results_path)}
    if results_path.exists():
        results_path.unlink()
    t0 = time.time()
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[{name}] exit={proc.returncode}, wall={elapsed:.1f}s, log={log_path.name}")
    result = None
    if results_path.exists():
        try:
            result = json.loads(results_path.read_text())
        except Exception as e:
            print(f"[{name}] failed to parse {results_path.name}: {e}")
    else:
        print(f"[{name}] WARNING: no JSON results at {results_path.name}")
        try:
            tail = log_path.read_text().splitlines()[-30:]
            print(f"[{name}] last lines of {log_path.name}:")
            for line in tail:
                print(f"    {line}")
        except Exception:
            pass
    return proc.returncode, result, elapsed


def _common_env(args) -> dict:
    return {
        "FL_BATCH_SIZE": str(args.batch_size),
        "FL_EPOCHS_PER_ROUND": str(args.epochs),
        "FL_DIRICHLET_ALPHA": str(args.alpha),
        "FL_ROUNDS": str(args.rounds),
        # Make layer_sharing run the full T rounds, matching the others.
        "FL_DISABLE_EARLY_STOP": "1",
    }


def _print_section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def _row_for(result: dict | None, kind: str) -> dict:
    """Normalize the very different summary shapes into one comparable row."""
    if result is None:
        return {"final_acc": None, "best_acc": None, "wall_s": None}
    if kind == "layer_sharing":
        return {
            "final_acc": result.get("avg_final_acc"),
            "best_acc": result.get("avg_best_acc"),
            "wall_s": result.get("total_time_s"),
            "metric": "avg over local clients",
        }
    if kind == "fedagent":
        return {
            "final_acc": result.get("final_acc"),
            "best_acc": result.get("best_acc"),
            "wall_s": result.get("total_time_s"),
            "metric": "single global model",
        }
    if kind == "flake":
        return {
            "final_acc": result.get("avg_final_acc"),
            "best_acc": result.get("avg_best_acc"),
            "wall_s": result.get("total_time_s"),
            "metric": "avg over local clients",
        }
    return {"final_acc": None, "best_acc": None, "wall_s": None}


def _fmt_acc(v) -> str:
    return f"{v:6.2f}%" if isinstance(v, (int, float)) else "  n/a "


def _fmt_secs(v) -> str:
    return f"{v:7.1f}s" if isinstance(v, (int, float)) else "   n/a "


def _print_comparison(ls: dict | None, fa: dict | None, fk: dict | None) -> None:
    print("\n" + "=" * 72)
    print("COMPARISON SUMMARY")
    print("=" * 72)

    sample = ls or fa or fk or {}
    _print_section("Aligned hyperparameters")
    print(f"  batch_size       : {sample.get('batch_size')}")
    print(f"  epochs_per_round : {sample.get('epochs_per_round')}")
    print(f"  dirichlet_alpha  : {sample.get('dirichlet_alpha')}")
    print(f"  rounds           : "
          f"{sample.get('max_rounds') or sample.get('rounds')}")

    rows = [
        ("layer_sharing", "decentralized P2P, random layer stacking, no KD", _row_for(ls, "layer_sharing")),
        ("fedagent",      "centralized, Double-DQN selection + KD (vs global)", _row_for(fa, "fedagent")),
        ("flake",        "decentralized P2P, KD vs locally-aggregated peer", _row_for(fk, "flake")),
    ]

    _print_section("Side-by-side (final acc | best acc | wall time | metric)")
    header = f"  {'framework':<14} {'final':>8} {'best':>8} {'wall':>9}   metric"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, _desc, row in rows:
        print(f"  {name:<14} "
              f"{_fmt_acc(row['final_acc'])} {_fmt_acc(row['best_acc'])} "
              f"{_fmt_secs(row['wall_s'])}   {row.get('metric', '-')}")

    _print_section("What each row means")
    for name, desc, _row in rows:
        print(f"  {name:<14} : {desc}")

    # Best of the bunch on final accuracy (handle None safely).
    valid = [(name, row) for name, _d, row in rows if isinstance(row["final_acc"], (int, float))]
    if len(valid) >= 2:
        winner_name, winner_row = max(valid, key=lambda kv: kv[1]["final_acc"])
        _print_section("Winner (final accuracy)")
        print(f"  {winner_name} @ {winner_row['final_acc']:.2f}%")

    print(
        "\nNote: layer_sharing and flake report the *average* test accuracy "
        "across all local clients\n(each keeps its own model). fedagent reports "
        "the *single* global model's test accuracy.\nAll three evaluate on the "
        "same CIFAR-10 test set with the same Dirichlet split."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run layer_sharing.py, fedagent.py, flake.py with aligned settings."
    )
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of FL rounds for all runs (default: 20)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Local epochs per round (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Local SGD batch size (default: 32)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID split (default: 0.5)")
    parser.add_argument("--ls-model", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help="layer_sharing --model choice (default: 1=SimpleCNN)")
    parser.add_argument("--fa-model", type=int, default=2, choices=[1, 2],
                        help="fedagent --model choice (default: 2=SimpleCNN)")
    parser.add_argument("--fk-model", type=int, default=2, choices=[1, 2],
                        help="flake --model choice (default: 2=SimpleCNN)")
    parser.add_argument("--skip-layer-sharing", action="store_true")
    parser.add_argument("--skip-fedagent", action="store_true")
    parser.add_argument("--skip-flake", action="store_true")
    args = parser.parse_args()

    inputs = {
        "layer_sharing": ROOT / "layer_sharing_input_localhost.txt",
        "fedagent": ROOT / "fedagent_input_localhost.txt",
        "flake": ROOT / "flake_input_localhost.txt",
    }
    skip = {
        "layer_sharing": args.skip_layer_sharing,
        "fedagent": args.skip_fedagent,
        "flake": args.skip_flake,
    }
    for name, path in inputs.items():
        if not skip[name] and not path.exists():
            print(f"Missing {path.name}; aborting.")
            return 1

    common_env = _common_env(args)

    print("Aligned settings for this comparison run:")
    for k, v in common_env.items():
        print(f"  {k} = {v}")
    print(f"  layer_sharing model = {args.ls_model}")
    print(f"  fedagent      model = {args.fa_model}")
    print(f"  flake        model = {args.fk_model}")

    ls_results: dict | None = None
    fa_results: dict | None = None
    fk_results: dict | None = None

    if not args.skip_layer_sharing:
        _, ls_results, _ = _run_child(
            "layer_sharing",
            [sys.executable, "layer_sharing.py", "--model", str(args.ls_model)],
            {**common_env, "LAYER_SHARING_INPUT": str(inputs["layer_sharing"])},
            ROOT / "compare_layer_sharing_results.json",
            ROOT / "compare_layer_sharing.log",
        )

    if not args.skip_fedagent:
        _, fa_results, _ = _run_child(
            "fedagent",
            [sys.executable, "fedagent.py", "--model", str(args.fa_model)],
            {**common_env, "FEDAGENT_INPUT": str(inputs["fedagent"])},
            ROOT / "compare_fedagent_results.json",
            ROOT / "compare_fedagent.log",
        )

    if not args.skip_flake:
        _, fk_results, _ = _run_child(
            "flake",
            [sys.executable, "flake.py", "--model", str(args.fk_model)],
            {**common_env, "flake_INPUT": str(inputs["flake"])},
            ROOT / "compare_flake_results.json",
            ROOT / "compare_flake.log",
        )

    _print_comparison(ls_results, fa_results, fk_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
