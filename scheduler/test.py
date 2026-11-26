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
        print("Result: ", res)
        print(tokenizer.batch_decode(res))

if __name__ == "__main__":
    print("in mian")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_scheduler_step(tokenizer, ["What is pizza"])
    # test_scheduler_step(tokenizer, ["What continent is United States in?", "What is pizza?", "Who are you?", "what?", "how?", "Where am I?", "London is a city in", "Paris is a city in"])