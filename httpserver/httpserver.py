from fastapi import FastAPI, HTTPException
import uvicorn
from transformers import AutoTokenizer
from scheduler.scheduler import Scheduler
from scheduler.request import Request, DONE_TOKEN
import threading 
import queue
import json
import random
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from typing import AsyncIterator
app = FastAPI()

class Server:
    def __init__(self, batch_size=15, kv_pool_size=1000):
        self.app = FastAPI()
        self.app.get("/v1/models")(self.models)
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.scheduler = Scheduler(self.tokenizer, batch_size, kv_pool_size)
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

    def consume_tokens(self, queue):
        while True:
            token_id = queue.get()
            if isinstance(token, DONE_TOKEN):
               return  
            token = self.tokenizer.decode([token_id])
            yield token

    @app.post("/v1/chat/completions")
    def chat_completions(self, request: dict):        
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

        return StreamingResponse(
            self.stream_response(request.resp_tokens),
            media_type="text/event-stream",
        )

        # yield self.generate_chunk(completion_id, {"role": "assistant", "content": ""})

        # while True:
        #     token_id = queue.get()
        #     if isinstance(token, DONE_TOKEN):
        #         yield self.generate_chunk(completion_id, "data: [DONE]\n\n", finish_reason="stop")
        #         break  
        #     token = self.tokenizer.decode([token_id])
        #     yield self.generate_chunk(completion_id, {"content": token})

        
        # result_token_ids = request.future.result()
        # response = self.tokenizer.batch_decode(result_token_ids)
        # # todo separate tokenization into its own abstraction

        # return {
        #     "id": "chatcmpl-123",
        #     "object": "chat.completion",
        #     "model": "gpt2",
        #     "choices": [
        #         {
        #             "index": 0,
        #             "message": {
        #                 "role": "assistant",
        #                 "content": response,
        #             },
        #             "finish_reason": "stop",
        #         }
        #     ],
        # }

    def generate_chunk(self, completion_id, delta, finish_reason=None):
        # print("completion id", completion_id)
        # print("type", type(completion_id))
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "model": "gpt2",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(payload)}"

    async def stream_response(self, queue) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{random.randrange(0, 100000, 1)}"

        yield self.generate_chunk(completion_id, {"role": "assistant", "content": ""})
        
        while True:
            token_id = queue.get()
            if token_id == DONE_TOKEN:
                yield self.generate_chunk(completion_id, "data: [DONE]\n\n", finish_reason="stop")
                break  
            token = self.tokenizer.decode([token_id])
            print("yielding this token: ", token)
            yield self.generate_chunk(completion_id, {"content": token})

                
    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

