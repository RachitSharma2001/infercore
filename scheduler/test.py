import torch
from scheduler import Scheduler
from request import Request
from transformers import AutoTokenizer
import threading


def test_scheduler_step(tokenizer, prompts):
    scheduler = Scheduler(tokenizer)
    thread = threading.Thread(
        target=scheduler.process,
        daemon=True,
    )
    thread.start()

    requests = [Request(tokenizer(prompt, return_tensors="pt")['input_ids'][0]) for prompt in prompts]
    for req in requests:
        scheduler.add(req)

    for req in requests:
        res = req.future.result()
        print(res)

if __name__ == "__main__":
    print("in mian")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_scheduler_step(tokenizer, ["Where is Paris?", "What continent is France in?"])