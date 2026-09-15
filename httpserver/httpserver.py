from fastapi import FastAPI, HTTPException
import uvicorn
from transformers import AutoTokenizer
from scheduler.scheduler import Scheduler
from scheduler.request import Request
import threading 

class Server:
    def __init__(self):
        self.app = FastAPI()
        self.app.get("/v1/models")(self.models)
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.scheduler = Scheduler(self.tokenizer)
        thread = threading.Thread(
            target=self.scheduler.process,
            daemon=True,
        )
        thread.start()

    def models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt2",
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    def chat_completions(self, request: dict):
        # todo add validation that model provided is supported 
        messages = request['messages']
        if len(messages) > 1:
            raise HTTPException(
                status_code=400,
                detail="Only one message at a time is supported right now"
            )
        prompt = messages[0]['content']

        input_token_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'][0]
        request = Request(input_token_ids)
        self.scheduler.add(request)
        result_token_ids = request.future.result()
        response = self.tokenizer.batch_decode(result_token_ids)
        # todo separate tokenization into its own abstraction

        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt2",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

# server = Server()
# server.run()