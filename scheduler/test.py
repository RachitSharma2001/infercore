import torch
from scheduler import Scheduler
from transformers import AutoTokenizer

def test_scheduler_step(tokenizer, prompt):
    scheduler = Scheduler(tokenizer)
    token_ids = tokenizer(prompt, return_tensors="pt")
    scheduler.add(token_ids['input_ids'][0])

    while len(scheduler.requests) == 1:
        scheduler.step()  # currently just prints logits -- no return value to assert on yet

if __name__ == "__main__":
    print("in mian")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_scheduler_step(tokenizer, "Where is Paris?")