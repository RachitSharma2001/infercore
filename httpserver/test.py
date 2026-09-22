import threading
import time
import requests
from httpserver.httpserver import Server

from concurrent.futures import ThreadPoolExecutor


def post(prompt):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "gpt2",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
    )
    return response.json()


server = Server()
thread = threading.Thread(
    target=server.run,
    daemon=True,
)
thread.start()
time.sleep(0.5)

prompts = ["Where is Paris?", "What is capital of France?", "How many feet are in a mile?"]

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(post, prompts[i])
        for i in range(len(prompts))
    ]
    results = [future.result() for future in futures]

print(results)


# server = Server()
# thread = threading.Thread(
#     target=server.run,
#     daemon=True,
# )
# thread.start()
# time.sleep(0.5)
# response = requests.post(
#     "http://localhost:8000/v1/chat/completions",
#     json={
#         "model": "gpt2",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Where is Paris?",
#             }
#         ],
#     },
# )

# print(response.status_code)
# print(response.json())