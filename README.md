# infercore

This is a small LLM inference engine I built in order to learn how serving models actually work under the hood. The only model it runs right now is GPT-2, and it runs only on CPU. Nonetheless, it has the architecture and many of the pieces you would find in a real inference engine like vLLM. 

These are the specific components:

- **Model** (`models/`): a from-scratch GPT-2 implementation, with a kv cache
- **KV cache** (`kvcache2/`): a pre-allocated pool of cache slots that all requests share. Each request grabs slots for its prompt and for the tokens it generates.
- **Scheduler** (`scheduler/`): a scheduler that uses continous batching. 
- **HTTP server** (`httpserver/`): a thin FastAPI wrapper around the scheduler, which runs in a background thread.
- **Benchmark** (`benchmark/`): measures latency and throughput of the engine. Results are below.

This project was just for me to learn about inference, it is not something to be deployed. Below are the benchmark numbers. 

## Benchmark

Setup: GPT-2 (124M), CPU only (8 threads) on an Apple M2 Max with 32 GB, PyTorch 2.14, Python 3.14. 

To reproduce, run `benchmark/bench_engine.py` with the project's venv Python. It takes about 75 seconds and writes the raw numbers to `benchmark/results.json`, which is where every number below comes from. 

### Results

| Group | KV slots | Clients | Prompt tokens | TTFT p50 / p99 (ms) | Queue wait p50 (ms) | TPOT p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | Tok/s | Req/s | Active/step | Prefill step (ms) | Decode step (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| load | 110 | 1 | 8 | 39 / 40 | 12 | 12.7 | 13.2 | 230 | 62.5 | 4.16 | 1.00 | 27.0 | 12.7 |
| load | 110 | 2 | 8 | 29 / 41 | 0 | 29.5 | 30.8 | 473 | 63.0 | 4.20 | 2.00 | 28.0 | 29.5 |
| load | 110 | 3 | 8 | 30 / 43 | 0 | 27.3 | 28.4 | 441 | 101.8 | 6.78 | 3.00 | 30.2 | 27.3 |
| load | 110 | 4 | 8 | 31 / 480 | 0 | 27.3 | 28.7 | 444 | 101.6 | 6.77 | 3.00 | 30.3 | 27.4 |
| prompt-length | 110 | 1 | 8 | 39 / 40 | 12 | 12.6 | 13.1 | 228 | 62.4 | 4.16 | 1.00 | 26.9 | 12.7 |
| prompt-length | 110 | 1 | 32 | 44 / 44 | 12 | 12.9 | 13.4 | 238 | 60.3 | 4.02 | 1.00 | 31.3 | 13.0 |
| prompt-length | 110 | 1 | 64 | 46 / 49 | 12 | 13.3 | 14.8 | 247 | 58.0 | 3.87 | 1.00 | 34.7 | 13.4 |
| prompt-length | 110 | 1 | 90 | 60 / 125 | 12 | 14.0 | 23.5 | 270 | 51.6 | 3.44 | 1.00 | 54.8 | 14.9 |
| big-pool | 2000 | 1 | 32 | 47 / 48 | 12 | 13.6 | 14.3 | 252 | 57.1 | 3.81 | 1.00 | 34.7 | 13.6 |
| big-pool | 2000 | 3 | 32 | 55 / 69 | 0 | 30.7 | 32.3 | 517 | 77.1 | 5.14 | 3.00 | 53.2 | 30.3 |
| big-pool | 2000 | 6 | 32 | 518 / 536 | 474 | 28.8 | 30.7 | 950 | 82.3 | 5.49 | 3.00 | 45.0 | 28.8 |
| big-pool | 2000 | 12 | 32 | 1473 / 1489 | 1429 | 28.9 | 30.3 | 1907 | 82.2 | 5.48 | 3.00 | 45.0 | 28.8 |

Every request produced exactly 15 tokens, since generation always hit the cap before an end-of-sequence token. The "big-pool" rows use a 2000-slot KV pool that the benchmark swaps in, because the default 110 slots can't hold more than a few requests at once.
