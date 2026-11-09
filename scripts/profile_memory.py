"""KV-cache memory profiler: paged allocation vs contiguous reservation.

Drives the real :class:`kvforge.cache.BlockManager` with a synthetic decode workload
(no model needed, runs in seconds) and reports, for a fixed KV memory budget:

* how many sequences fit concurrently under each allocation strategy,
* internal fragmentation (allocated-but-unused slots) per block size,
* prefix-cache sharing savings for a shared system prompt.

Example::

    python scripts/profile_memory.py --preset llama-8b --kv-gb 16 --dtype-bytes 2
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

from kvforge.cache.block_manager import BlockManager
from kvforge.model_loader import PRESETS


def bytes_per_token(preset: str, dtype_bytes: int) -> int:
    cfg = PRESETS[preset]
    assert cfg.head_dim is not None
    return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * dtype_bytes


def sample_lengths(n: int, rng: random.Random, max_len: int) -> list[tuple[int, int]]:
    """(prompt_len, output_len), long-tailed like chat traffic (ShareGPT-ish)."""
    out = []
    for _ in range(n):
        p = min(max_len // 2, max(4, int(rng.lognormvariate(5.0, 0.9))))
        o = min(max_len - p, max(4, int(rng.lognormvariate(5.2, 0.8))))
        out.append((p, o))
    return out


def simulate_paged(
    num_blocks: int, block_size: int, workload: list[tuple[int, int]], shared_prefix: int = 0,
    prefix_caching: bool = False,
) -> dict[str, float]:
    """Continuous admission until the pool is full; every running seq decodes 1 token/step."""
    bm = BlockManager(num_blocks, block_size, enable_prefix_caching=prefix_caching)
    pending = list(enumerate(workload))
    # seq id -> (current length, final length, workload index)
    running: dict[str, tuple[int, int, int]] = {}
    concurrency: list[int] = []
    frag: list[float] = []
    util: list[float] = []
    preemptions = 0
    system = list(range(shared_prefix))
    steps = 0
    while (pending or running) and steps < 50000:
        steps += 1
        while pending:
            i, (p, o) = pending[0]
            tokens = system + [10_000 + i * 7919 + j for j in range(p)]
            if not bm.can_allocate(tokens):
                break
            pending.pop(0)
            sid = f"s{i}"
            bm.allocate(sid, tokens)
            bm.mark_computed(sid, tokens, len(tokens))
            running[sid] = (len(tokens), len(tokens) + o, i)
        for sid in list(running):
            if sid not in running:
                continue
            n, target, i = running[sid]
            if n >= target:
                bm.free(sid)
                del running[sid]
                continue
            while not bm.can_append_slots(sid, n + 1, n) and len(running) > 1:
                # Recompute-preempt the newest sequence, as the real scheduler does.
                victim = next(reversed(running))
                if victim == sid:
                    break
                bm.free(victim)
                pending.insert(0, (running.pop(victim)[2], workload[int(victim[1:])]))
                preemptions += 1
            if not bm.can_append_slots(sid, n + 1, n):
                continue
            bm.append_slots(sid, n + 1, n)
            running[sid] = (n + 1, target, i)
        # Measure only while saturated (requests still queued) and sample to keep it cheap.
        if running and pending and steps % 10 == 0:
            s = bm.stats()
            concurrency.append(len(running))
            frag.append(s.internal_fragmentation)
            util.append(s.utilization)
    return {
        "mean_concurrency": statistics.mean(concurrency),
        "peak_concurrency": max(concurrency),
        "mean_internal_fragmentation": statistics.mean(frag),
        "mean_utilization": statistics.mean(util),
        "preemptions": preemptions,
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="llama-8b", choices=sorted(PRESETS))
    ap.add_argument("--kv-gb", type=float, default=16.0)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--num-requests", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bpt = bytes_per_token(args.preset, args.dtype_bytes)
    budget_tokens = int(args.kv_gb * 1024**3 // bpt)
    workload = sample_lengths(args.num_requests, rng, args.max_model_len)
    mean_final = statistics.mean(p + o for p, o in workload)

    report: dict[str, object] = {
        "preset": args.preset,
        "kv_bytes_per_token": bpt,
        "kv_budget_tokens": budget_tokens,
        "mean_final_seq_len": mean_final,
        "contiguous_max_model_len": {
            "max_concurrency": budget_tokens // args.max_model_len,
            "waste_fraction": 1 - mean_final / args.max_model_len,
        },
        "contiguous_prompt_plus_max_tokens": {
            # Static-batching servers reserve prompt + max_new_tokens; assume max_new = 2x mean.
            "max_concurrency": int(budget_tokens // (mean_final * 1.5)),
            "waste_fraction": 1 - 1 / 1.5,
        },
        "paged": {},
    }
    for bs in (8, 16, 32, 64):
        report["paged"][f"block_size_{bs}"] = simulate_paged(budget_tokens // bs, bs, workload)  # type: ignore[index]

    shared = 512
    no_share = simulate_paged(budget_tokens // 16, 16, workload, shared_prefix=shared)
    share = simulate_paged(budget_tokens // 16, 16, workload, shared_prefix=shared, prefix_caching=True)
    report["shared_system_prompt_512"] = {
        "no_prefix_caching_mean_concurrency": no_share["mean_concurrency"],
        "prefix_caching_mean_concurrency": share["mean_concurrency"],
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return
    c1 = report["contiguous_max_model_len"]
    c2 = report["contiguous_prompt_plus_max_tokens"]
    print(f"model={args.preset}  KV/token={bpt / 1024:.0f} KiB  budget={args.kv_gb} GiB "
          f"= {budget_tokens:,} tokens  mean final len={mean_final:.0f}\n")
    print(f"{'strategy':38s} {'concurrency':>12s} {'KV waste':>9s}")
    print(f"{'contiguous, reserve max_model_len':38s} {c1['max_concurrency']:>12d} {c1['waste_fraction']:>8.1%}")  # type: ignore[index]
    print(f"{'contiguous, reserve prompt+max_tokens':38s} {c2['max_concurrency']:>12d} {c2['waste_fraction']:>8.1%}")  # type: ignore[index]
    for name, r in report["paged"].items():  # type: ignore[union-attr]
        label = f"paged, {name.replace('_', ' ')}"
        print(f"{label:38s} {r['mean_concurrency']:>12.1f} {r['mean_internal_fragmentation']:>8.1%}")
    s = report["shared_system_prompt_512"]
    print(f"\nshared 512-token system prompt (bs=16): mean concurrency "
          f"{s['no_prefix_caching_mean_concurrency']:.1f} -> {s['prefix_caching_mean_concurrency']:.1f} "  # type: ignore[index]
          "with prefix caching")


if __name__ == "__main__":
    main()
