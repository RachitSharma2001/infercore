import threading
import time
import httpserver 
import requests

server = httpserver.Server()
thread = threading.Thread(
    target=server.run,
    daemon=True,
)
thread.start()
time.sleep(0.5)
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt2",
        "messages": [
            {
                "role": "user",
                "content": "Where is Paris?",
            }
        ],
    },
)

print(response.status_code)
print(response.json())