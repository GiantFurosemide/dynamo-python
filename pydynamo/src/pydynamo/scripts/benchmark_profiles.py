#!/usr/bin/env python3
"""
Benchmark small/medium/large alignment profiles with regression thresholds.

Also supports multi-process alignment (num_workers=2,4) and multi-worker
reconstruction (recon_workers=2) for capacity planning.
recon_mp2 measures transform+sum throughput with synthetic in-memory volumes;
it does not include MRC I/O.

Run: python -m pydynamo.scripts.benchmark_profiles --profiles small,medium,small_mp2,small_mp4
     python -m pydynamo.scripts.benchmark_profiles --profiles recon_mp2 --output bench.json
     60k-scale (for CI/regression): --profiles 60k_mp4,60k_mp8 (1000 particles, multi-worker).
Output: JSON with elapsed_s, throughput_particles_per_s, peak_rss_mb per profile.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..core.align import align_one_particle

# Used by multi-process alignment workers (set in child via initializer).
_bench_ref = None
_bench_mask = None

PROFILE_DEFAULTS = {
    "small": {"particles": 64, "box": 32},
    "medium": {"particles": 128, "box": 48},
    "large": {"particles": 256, "box": 64},
    "small_mp2": {"particles": 64, "box": 32, "num_workers": 2},
    "small_mp4": {"particles": 64, "box": 32, "num_workers": 4},
    "recon_mp2": {"particles": 128, "box": 32, "recon_workers": 2},
    # 60k-scale: 1000 particles + multi-worker for regression / capacity planning (jg_015)
    "60k_mp4": {"particles": 1000, "box": 48, "num_workers": 4},
    "60k_mp8": {"particles": 1000, "box": 48, "num_workers": 8},
}


def _peak_rss_mb() -> float:
    v = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if v <= 0:
        return 0.0
    return v / 1024.0


def _align_worker_init(ref_mask_tuple):
    """Set global ref/mask in worker for multi-process alignment."""
    global _bench_ref, _bench_mask
    _bench_ref, _bench_mask = ref_mask_tuple


def _align_worker_task(particle_seed_box):
    """Run one alignment in worker; particle_seed_box = (particle_array, seed, box)."""
    part, seed, box = particle_seed_box
    align_one_particle(
        part,
        _bench_ref,
        mask=_bench_mask,
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
    return None


def _run_profile(name: str, particles: int, box: int, seed: int, num_workers: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((box, box, box), dtype=np.float32)
    mask = np.ones((box, box, box), dtype=bool)

    if num_workers is None or num_workers <= 1:
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
    else:
        parts = [rng.standard_normal((box, box, box), dtype=np.float32) for _ in range(int(particles))]
        tasks = [(p, seed + i, box) for i, p in enumerate(parts)]
        start = time.time()
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_align_worker_init, initargs=((ref, mask),)) as ex:
            list(as_completed(ex.submit(_align_worker_task, t) for t in tasks))
        elapsed = max(1e-9, time.time() - start)

    throughput = float(particles) / elapsed
    out = {
        "profile": name,
        "particles": int(particles),
        "box": int(box),
        "elapsed_s": float(elapsed),
        "throughput_particles_per_s": float(throughput),
        "peak_rss_mb": float(_peak_rss_mb()),
    }
    if num_workers is not None and num_workers > 1:
        out["num_workers"] = num_workers
    return out


def _recon_chunk_worker(args):
    """Worker: generate volumes from seed and run apply_inverse_transform + sum."""
    from ..core.average import apply_inverse_transform

    start_idx, end_idx, box, base_seed, angles_chunk, shifts_chunk = args
    rng = np.random.default_rng(base_seed + start_idx)
    acc = np.zeros((box, box, box), dtype=np.float64)
    for i in range(end_idx - start_idx):
        vol = rng.standard_normal((box, box, box), dtype=np.float32)
        tr = apply_inverse_transform(
            vol,
            angles_chunk[i, 0], angles_chunk[i, 1], angles_chunk[i, 2],
            shifts_chunk[i, 0], shifts_chunk[i, 1], shifts_chunk[i, 2],
        )
        acc += tr
    return acc, end_idx - start_idx


def _run_recon_profile(name: str, particles: int, box: int, seed: int, recon_workers: int) -> dict:
    """
    Minimal reconstruction-style benchmark: transform and sum volumes with recon_workers.
    Uses synthetic in-memory volumes (no MRC I/O); reports throughput for transform+sum only.
    """
    rng = np.random.default_rng(seed)
    n = int(particles)
    angles = rng.uniform(0, 360, (n, 3)).astype(np.float64)
    shifts = rng.uniform(-2, 2, (n, 3)).astype(np.float64)

    chunk_size = max(1, (n + recon_workers - 1) // recon_workers)
    chunks = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        chunks.append((s, e, box, seed, angles[s:e], shifts[s:e]))

    start = time.time()
    with ProcessPoolExecutor(max_workers=recon_workers) as ex:
        results = list(ex.map(_recon_chunk_worker, chunks))
    elapsed = max(1e-9, time.time() - start)
    total = sum(r[1] for r in results)
    throughput = total / elapsed
    return {
        "profile": name,
        "particles": total,
        "box": int(box),
        "elapsed_s": float(elapsed),
        "throughput_particles_per_s": float(throughput),
        "peak_rss_mb": float(_peak_rss_mb()),
        "recon_workers": recon_workers,
    }


def _parse_profiles(raw: str) -> list[str]:
    names = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not names:
        return ["small", "medium", "large"]
    for n in names:
        if n not in PROFILE_DEFAULTS:
            raise ValueError(f"Unknown profile: {n}. Choose from: {list(PROFILE_DEFAULTS.keys())}")
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
        if spec.get("recon_workers"):
            results.append(_run_recon_profile(
                name,
                spec["particles"],
                spec["box"],
                seed=args.seed + idx,
                recon_workers=spec["recon_workers"],
            ))
        else:
            results.append(_run_profile(
                name,
                spec["particles"],
                spec["box"],
                seed=args.seed + idx,
                num_workers=spec.get("num_workers"),
            ))

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
