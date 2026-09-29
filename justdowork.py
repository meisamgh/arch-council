import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("JUSTWOKER_API_KEY")
if not API_KEY:
    raise RuntimeError("JUSTWOKER_API_KEY is missing from .env")


URL = "https://api.justwoker.icu/v1/messages"

headers = {
    "x-api-key": API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

payload = {
    "model": "claude-opus-4-8",
    "max_tokens": 100,
    "messages": [
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ]
}

try:
    response = requests.post(
        URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    print("Status:", response.status_code)
    print("Response:")
    print(response.text)

    if response.ok:
        data = response.json()

        print("\nModel answer:")
        print(data["content"][0]["text"])

except requests.exceptions.RequestException as e:
    print("Request error:", e)
