# This (and rest of this package) was vibe coded

"""In-process latency/throughput benchmark for the infercore scheduler.

The HTTP server is not streaming: it returns only when a request has finished,
so TTFT and TPOT cannot be seen from a client. This script drives the real
Scheduler directly (same thread-based loop the server uses) and wraps its
step() method to timestamp every token as the step that produced it ends.
The engine code itself is not modified.

    venv/bin/python benchmark/bench_engine.py
"""
import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kvcache2 import Pool  # noqa: E402
from scheduler.request import Request  # noqa: E402
from scheduler.scheduler import Scheduler  # noqa: E402

DEFAULT_SLOTS = 110
# Each request needs its prompt plus ~16 decode slots, and slots are only
# returned when it finishes. The pool has no admission control (running out
# crashes the scheduler thread), so scenarios must keep
# concurrency * (prompt + 16) <= pool slots.
DECODE_SLOT_MARGIN = 16

# (label, pool slots, concurrency, prompt tokens)
SCENARIOS = [
    ("load", DEFAULT_SLOTS, 1, 8),
    ("load", DEFAULT_SLOTS, 2, 8),
    ("load", DEFAULT_SLOTS, 3, 8),
    ("load", DEFAULT_SLOTS, 4, 8),
    ("prompt-length", DEFAULT_SLOTS, 1, 8),
    ("prompt-length", DEFAULT_SLOTS, 1, 32),
    ("prompt-length", DEFAULT_SLOTS, 1, 64),
    ("prompt-length", DEFAULT_SLOTS, 1, 90),
    ("big-pool", 2000, 1, 32),
    ("big-pool", 2000, 3, 32),
    ("big-pool", 2000, 6, 32),
    ("big-pool", 2000, 12, 32),
]

PASSAGE = (
    " The history of bread begins with grain that was ground between stones, "
    "mixed with water and baked on hot rocks near the fire. Over thousands of "
    "years, bakers learned that dough left in the open would rise, and that "
    "the resulting loaf was lighter and kept for longer. Mills, ovens and "
    "markets grew up around this simple food, and towns came to depend on "
    "the people who made it."
)


class Recorder:
    """Wraps scheduler.add and scheduler.step to timestamp tokens and steps."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.records = {}  # Request -> dict
        self.steps = []  # (duration, active requests, produced a first token)
        self._add = scheduler.add
        self._step = scheduler.step
        scheduler.add = self.add
        scheduler.step = self.step

    def add(self, request):
        self.records[request] = {
            "arrival": time.perf_counter(), "tokens": [], "done": None,
            "first_step_start": None, "seen": 0,
        }
        self._add(request)

    def step(self):
        sched = self.scheduler
        before = list(sched.requests)
        t0 = time.perf_counter()
        self._step()
        t1 = time.perf_counter()
        # requests that finished were removed from sched.requests, so look at
        # the pre-step list too
        active = before + [r for r in sched.requests if r not in before]
        n_active, first = 0, False
        for req in active:
            rec = self.records.get(req.request)
            if rec is None:
                continue
            changed = False
            if req.num_response_tokens > rec["seen"]:
                rec["seen"] = req.num_response_tokens
                rec["tokens"].append(t1)
                if len(rec["tokens"]) == 1:
                    rec["first_step_start"] = t0
                    first = True
                changed = True
            if rec["done"] is None and req.request.future.done():
                rec["done"] = t1
                changed = True
            n_active += changed
        if n_active:
            self.steps.append((t1 - t0, n_active, first))

    def reset(self):
        self.records = {}
        self.steps = []


def make_prompts(tokenizer, prompt_tokens, count):
    """Distinct prompts of exactly prompt_tokens tokens each."""
    ids = tokenizer(PASSAGE)["input_ids"]
    needed = prompt_tokens + count
    ids = (ids * (needed // len(ids) + 1))[:needed]
    prompts = [tokenizer.decode(ids[i:i + prompt_tokens]) for i in range(count)]
    for p in prompts:
        got = len(tokenizer(p)["input_ids"])
        if got != prompt_tokens:
            raise ValueError(f"prompt has {got} tokens, wanted {prompt_tokens}")
    return prompts


def enlarge_pool(scheduler, slots):
    pool = Pool(12, slots, 12, 64)
    scheduler.gpt2_kpool = pool
    for block in scheduler.model.transformer_blocks:
        block.attention.kpool = pool


def pct(values, scale=1000):
    v = np.array(values) * scale
    return {
        "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
    }


def run_scenario(sched, thread, rec, tokenizer, pool_slots, concurrency, prompt_tokens, num_requests, warmup):
    if concurrency * (prompt_tokens + DECODE_SLOT_MARGIN) > pool_slots:
        raise ValueError("scenario would overflow the KV pool")
    prompts = make_prompts(tokenizer, prompt_tokens, num_requests + warmup)

    def submit(prompt):
        req = Request(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        sched.add(req)
        req.future.result(timeout=120)
        return req

    for prompt in prompts[:warmup]:
        submit(prompt)
    rec.reset()

    pending = list(prompts[warmup:])
    lock = threading.Lock()
    errors = []

    def worker():
        while True:
            with lock:
                if not pending:
                    return
                prompt = pending.pop()
            try:
                submit(prompt)
            except Exception as e:  # scheduler crash shows up as a timeout
                errors.append(repr(e))
                return

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    pool = sched.gpt2_kpool
    leaked = len(pool.free_slots) != pool_slots
    result = {
        "pool_slots": pool_slots, "concurrency": concurrency,
        "prompt_tokens": prompt_tokens, "errors": errors,
        "scheduler_alive": thread.is_alive(), "slot_leak": bool(leaked),
    }
    done = [r for r in rec.records.values() if r["done"] and r["tokens"]]
    if not done:
        return result

    ttft = [r["tokens"][0] - r["arrival"] for r in done]
    queue = [r["first_step_start"] - r["arrival"] for r in done]
    e2e = [r["done"] - r["arrival"] for r in done]
    counts = [len(r["tokens"]) for r in done]
    tpot = [(r["tokens"][-1] - r["tokens"][0]) / (len(r["tokens"]) - 1)
            for r in done if len(r["tokens"]) > 1]
    itl = [d for r in done for d in np.diff(r["tokens"])]
    prefill_steps = [s[0] for s in rec.steps if s[2]]
    decode_steps = [s[0] for s in rec.steps if not s[2]]
    total_tokens = int(np.sum(counts))
    result.update({
        "requests": len(done),
        "ttft_ms": pct(ttft), "queue_wait_ms": pct(queue),
        "tpot_ms": pct(tpot) if tpot else None,
        "itl_ms": pct(itl) if itl else None,
        "e2e_ms": pct(e2e),
        "output_tokens_mean": float(np.mean(counts)),
        "tokens_per_sec": total_tokens / wall,
        "requests_per_sec": len(done) / wall,
        "mean_active_per_step": float(np.mean([s[1] for s in rec.steps])),
        "prefill_step_ms_mean": float(np.mean(prefill_steps) * 1000) if prefill_steps else None,
        "decode_step_ms_mean": float(np.mean(decode_steps) * 1000) if decode_steps else None,
        "wall_seconds": wall,
    })
    return result


def markdown_table(rows):
    lines = [
        "| Group | Pool | Clients | Prompt | TTFT p50 / p99 (ms) | Queue wait p50 (ms) "
        "| TPOT p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | Out tokens | Tok/s | Req/s "
        "| Active/step | Prefill step (ms) | Decode step (ms) |",
        "|" + "---|" * 15,
    ]
    for r in rows:
        if "ttft_ms" not in r:
            lines.append(f"| {r['group']} | {r['pool_slots']} | {r['concurrency']} | {r['prompt_tokens']} "
                         f"| FAILED: {r['errors'][:1]} |" + " |" * 10)
            continue
        lines.append(
            f"| {r['group']} | {r['pool_slots']} | {r['concurrency']} | {r['prompt_tokens']} "
            f"| {r['ttft_ms']['p50']:.0f} / {r['ttft_ms']['p99']:.0f} "
            f"| {r['queue_wait_ms']['p50']:.0f} "
            f"| {r['tpot_ms']['p50']:.1f} | {r['itl_ms']['p99']:.1f} "
            f"| {r['e2e_ms']['p50']:.0f} | {r['output_tokens_mean']:.1f} "
            f"| {r['tokens_per_sec']:.1f} | {r['requests_per_sec']:.2f} "
            f"| {r['mean_active_per_step']:.2f} "
            f"| {r['prefill_step_ms_mean']:.1f} | {r['decode_step_ms_mean']:.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24, help="measured requests per scenario")
    parser.add_argument("--warmup", type=int, default=2, help="discarded requests per scenario")
    parser.add_argument("--output", default=str(ROOT / "benchmark" / "results.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sched = Scheduler(tokenizer)
    thread = threading.Thread(target=sched.process, daemon=True)
    thread.start()
    rec = Recorder(sched)
    batch_size = sched.batch_size
    max_response = sched.max_response_tokens

    rows = []
    current_pool = DEFAULT_SLOTS
    for group, pool_slots, concurrency, prompt_tokens in SCENARIOS:
        if pool_slots != current_pool:
            enlarge_pool(sched, pool_slots)
            current_pool = pool_slots
        print(f"[{group}] pool={pool_slots} clients={concurrency} prompt={prompt_tokens}", flush=True)
        row = run_scenario(sched, thread, rec, tokenizer, pool_slots, concurrency,
                           prompt_tokens, args.requests, args.warmup)
        row["group"] = group
        rows.append(row)
        if not row["scheduler_alive"]:
            print("scheduler thread died; stopping", flush=True)
            break

    report = {
        "machine": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "device": "cpu", "scheduler_batch_size": batch_size,
        "max_response_tokens": max_response,
        "requests_per_scenario": args.requests, "warmup_per_scenario": args.warmup,
        "scenarios": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(markdown_table(rows))
    return 1 if any(r["errors"] or r["slot_leak"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
