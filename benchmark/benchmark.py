# This (and rest of this package) was vibe coded


"""Client-side latency benchmark for a streaming /v1/chat/completions server.

Assumes the server is already running. Every timestamp is taken here, when a
token reaches the client, so the numbers include queueing, HTTP and decoding.

    python benchmark/benchmark.py --url http://127.0.0.1:8000
"""
import argparse
import codecs
import json
import platform
import threading
import time

import httpx
import numpy as np
import torch
from transformers import AutoTokenizer

# (concurrency, prompt tokens). The first group varies load, the second varies
# prompt length. Sized so in-flight requests stay inside the default 1000 slot
# KV pool, because the server has no admission control: each request needs its
# prompt plus about 52 slots.
SCENARIOS = [
    (1, 32),
    (2, 32),
    (4, 32),
    (8, 32),
    (1, 8),
    (1, 128),
    (1, 256),
]

PASSAGE = (
    " The history of bread begins with grain that was ground between stones, "
    "mixed with water and baked on hot rocks near the fire. Over thousands of "
    "years, bakers learned that dough left in the open would rise, and that "
    "the resulting loaf was lighter and kept for longer. Mills, ovens and "
    "markets grew up around this simple food, and towns came to depend on "
    "the people who made it."
)
JSON_DECODER = json.JSONDecoder()


def make_prompts(tokenizer, prompt_tokens, count):
    """Builds distinct prompts of exactly prompt_tokens tokens each."""
    passage_ids = tokenizer(PASSAGE)["input_ids"]
    needed = prompt_tokens + count
    ids = (passage_ids * (needed // len(passage_ids) + 1))[:needed]
    prompts = [tokenizer.decode(ids[i:i + prompt_tokens]) for i in range(count)]
    for prompt in prompts:
        actual = len(tokenizer(prompt)["input_ids"])
        if actual != prompt_tokens:
            raise ValueError(f"prompt has {actual} tokens, wanted {prompt_tokens}")
    return prompts


def parse_chunks(buffer):
    """Pulls complete chunks out of buffer, returning (chunks, leftover text).

    Finds each "data: " payload by decoding the JSON rather than splitting on
    the blank line SSE puts between events, so it doesn't depend on that framing.
    """
    chunks = []
    while True:
        start = buffer.find("data: ")
        if start == -1:
            return chunks, buffer
        body = buffer[start + len("data: "):]
        if body.startswith("[DONE]"):
            buffer = body[len("[DONE]"):]
            continue
        try:
            chunk, end = JSON_DECODER.raw_decode(body)
        except json.JSONDecodeError:
            # the rest of this chunk hasn't arrived yet
            return chunks, buffer[start:]
        chunks.append(chunk)
        buffer = body[end:]


def is_token(chunk):
    # the opening chunk only announces the role, so it must not count as the
    # first token or TTFT would look close to zero
    delta = chunk["choices"][0]["delta"]
    return isinstance(delta, dict) and "content" in delta and "role" not in delta


def run_request(client, url, prompt):
    """Sends one request and returns its token arrival times."""
    body = {
        "model": "gpt2",
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    token_times = []
    start = time.perf_counter()
    with client.stream("POST", f"{url}/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        # iter_raw hands over each read as it lands, with no extra buffering
        for raw in response.iter_raw():
            now = time.perf_counter()
            buffer += decoder.decode(raw)
            chunks, buffer = parse_chunks(buffer)
            token_times.extend(now for chunk in chunks if is_token(chunk))
    end = time.perf_counter()
    if not token_times:
        raise ValueError("response contained no tokens")
    return {"start": start, "end": end, "token_times": token_times}


def run_scenario(url, tokenizer, concurrency, prompt_tokens, num_requests, warmup):
    prompts = make_prompts(tokenizer, prompt_tokens, num_requests + warmup)

    # warmup requests run one at a time and are thrown away
    with httpx.Client(timeout=300) as client:
        for prompt in prompts[:warmup]:
            run_request(client, url, prompt)

    # closed loop: each worker sends its next request when the last one ends,
    # so exactly `concurrency` requests are in flight
    pending = list(prompts[warmup:])
    results = []
    errors = []
    lock = threading.Lock()

    def worker():
        with httpx.Client(timeout=300) as client:
            while True:
                with lock:
                    if not pending:
                        return
                    prompt = pending.pop()
                try:
                    result = run_request(client, url, prompt)
                except Exception as error:
                    with lock:
                        errors.append(repr(error))
                    continue
                with lock:
                    results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return summarize(results, errors, concurrency, prompt_tokens)


def peak_overlap(results):
    """Most requests that were ever between first and last token at once.

    Shows how many streams really ran together, which can be lower than the
    number of requests the client had in flight.
    """
    events = []
    for result in results:
        events.append((result["token_times"][0], 1))
        events.append((result["token_times"][-1], -1))
    active = peak = 0
    # at equal times, process ends first so touching intervals don't overlap
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        active += change
        peak = max(peak, active)
    return peak


def percentiles(values):
    values = np.array(values) * 1000
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def summarize(results, errors, concurrency, prompt_tokens):
    summary = {
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "requests": len(results),
        "errors": errors,
    }
    if not results:
        return summary

    ttft, tpot, itl, e2e, counts = [], [], [], [], []
    for result in results:
        times = result["token_times"]
        ttft.append(times[0] - result["start"])
        e2e.append(times[-1] - result["start"])
        counts.append(len(times))
        if len(times) > 1:
            tpot.append((times[-1] - times[0]) / (len(times) - 1))
            itl.extend(np.diff(times).tolist())

    wall = max(r["end"] for r in results) - min(r["start"] for r in results)
    summary.update({
        "ttft_ms": percentiles(ttft),
        "tpot_ms": percentiles(tpot),
        "itl_ms": percentiles(itl),
        "e2e_ms": percentiles(e2e),
        "itl_samples": len(itl),
        "output_tokens_mean": float(np.mean(counts)),
        "output_tokens_total": int(np.sum(counts)),
        "tokens_per_sec": float(np.sum(counts) / wall),
        "requests_per_sec": len(results) / wall,
        "peak_overlap": peak_overlap(results),
        "wall_seconds": wall,
    })
    return summary


def markdown_table(scenarios):
    lines = [
        "| Clients | Prompt tokens | TTFT p50 / p99 (ms) | TPOT p50 (ms) "
        "| ITL p50 / p99 (ms) | E2E p50 / p99 (ms) | Output tokens (mean) "
        "| Tokens/sec | Peak streams overlapping | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scenarios:
        if not s["requests"]:
            lines.append(
                f"| {s['concurrency']} | {s['prompt_tokens']} | - | - | - | - "
                f"| - | - | - | {len(s['errors'])} |"
            )
            continue
        lines.append(
            f"| {s['concurrency']} | {s['prompt_tokens']} "
            f"| {s['ttft_ms']['p50']:.0f} / {s['ttft_ms']['p99']:.0f} "
            f"| {s['tpot_ms']['p50']:.1f} "
            f"| {s['itl_ms']['p50']:.1f} / {s['itl_ms']['p99']:.1f} "
            f"| {s['e2e_ms']['p50']:.0f} / {s['e2e_ms']['p99']:.0f} "
            f"| {s['output_tokens_mean']:.1f} "
            f"| {s['tokens_per_sec']:.1f} "
            f"| {s['peak_overlap']} "
            f"| {len(s['errors'])} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=40, help="measured requests per scenario")
    parser.add_argument("--warmup", type=int, default=3, help="discarded requests per scenario")
    parser.add_argument("--output", default="benchmark/results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    scenarios = []
    for concurrency, prompt_tokens in SCENARIOS:
        print(f"running clients={concurrency} prompt_tokens={prompt_tokens}", flush=True)
        scenarios.append(
            run_scenario(args.url, tokenizer, concurrency, prompt_tokens, args.requests, args.warmup)
        )

    report = {
        "machine": platform.machine(),
        "client_torch_threads": torch.get_num_threads(),
        "requests_per_scenario": args.requests,
        "warmup_per_scenario": args.warmup,
        "scenarios": scenarios,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(markdown_table(scenarios))

    failed = sum(len(s["errors"]) for s in scenarios)
    if failed:
        print(f"{failed} requests failed, see {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
