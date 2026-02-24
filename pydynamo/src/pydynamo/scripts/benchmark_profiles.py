#!/usr/bin/env python3
"""Benchmark small/medium/large alignment profiles with regression thresholds."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from ..core.align import align_one_particle


PROFILE_DEFAULTS = {
    "small": {"particles": 64, "box": 32},
    "medium": {"particles": 128, "box": 48},
    "large": {"particles": 256, "box": 64},
}


def _peak_rss_mb() -> float:
    v = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if v <= 0:
        return 0.0
    return v / 1024.0


def _run_profile(name: str, particles: int, box: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((box, box, box), dtype=np.float32)
    mask = np.ones((box, box, box), dtype=bool)

    start = time.time()
    for _ in range(int(particles)):
        part = rng.standard_normal((box, box, box), dtype=np.float32)
        align_one_particle(
            part,
            ref,
            mask=mask,
            cone_step=90.0,
            tdrot_step=180.0,
            tdrot_range=(0.0, 360.0),
            cone_range=(0.0, 180.0),
            inplane_step=180.0,
            inplane_range=(0.0, 360.0),
            shift_search=0,
            multigrid_levels=1,
            shift_mode="cube",
            subpixel=False,
            cc_mode="ncc",
            angle_sampling_mode="dynamo",
            old_angles=(0.0, 0.0, 0.0),
            device="cpu",
            device_id=None,
        )
    elapsed = max(1e-9, time.time() - start)
    throughput = float(particles) / elapsed
    return {
        "profile": name,
        "particles": int(particles),
        "box": int(box),
        "elapsed_s": float(elapsed),
        "throughput_particles_per_s": float(throughput),
        "peak_rss_mb": float(_peak_rss_mb()),
    }


def _parse_profiles(raw: str) -> list[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not names:
        return ["small", "medium", "large"]
    for n in names:
        if n not in PROFILE_DEFAULTS:
            raise ValueError(f"Unknown profile: {n}")
    return names


def _load_baseline(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["profile"]: r for r in data.get("results", [])}


def _check_regression(results: list[dict], baseline: dict, max_slowdown_ratio: float) -> list[str]:
    issues = []
    for r in results:
        name = r["profile"]
        base = baseline.get(name)
        if not base:
            continue
        cur_t = float(r.get("throughput_particles_per_s", 0.0))
        base_t = float(base.get("throughput_particles_per_s", 0.0))
        if base_t <= 0:
            continue
        min_allowed = base_t * (1.0 - max_slowdown_ratio)
        if cur_t < min_allowed:
            drop_ratio = (base_t - cur_t) / base_t
            issues.append(
                f"profile={name} throughput drop {drop_ratio:.1%} "
                f"(current={cur_t:.3f}, baseline={base_t:.3f}, allowed={max_slowdown_ratio:.1%})"
            )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pydynamo small/medium/large profiles")
    parser.add_argument("--profiles", default="small,medium,large", help="comma list: small,medium,large")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--output", default=None, help="optional JSON output file path")
    parser.add_argument("--baseline", default=None, help="baseline JSON output for regression check")
    parser.add_argument(
        "--max-slowdown-ratio",
        type=float,
        default=0.20,
        help="allowed relative throughput slowdown vs baseline (0.20 means 20%%)",
    )
    args = parser.parse_args()

    profile_names = _parse_profiles(args.profiles)
    results = []
    for idx, name in enumerate(profile_names):
        spec = PROFILE_DEFAULTS[name]
        results.append(_run_profile(name, spec["particles"], spec["box"], seed=args.seed + idx))

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "results": results,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

    if args.baseline:
        baseline = _load_baseline(Path(args.baseline))
        issues = _check_regression(results, baseline, float(args.max_slowdown_ratio))
        if issues:
            for i in issues:
                print(f"[PERF-REGRESSION] {i}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
