
import os
import logging
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------------
# AIVORA-NEURAL SERVER
# ----------------------------------

APP_NAME = "Aivora-Neural"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-7B-Instruct"
)

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aivora-neural")

app = FastAPI(
    title="Aivora-Neural API",
    description="Independent AI backend for Aivora AI Dual Brain",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------
# REQUEST MODELS
# ----------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    history: Optional[List[ChatMessage]] = []


# ----------------------------------
# HEALTH CHECK
# ----------------------------------

@app.get("/")
async def home():
    return {
        "status": "online",
        "name": APP_NAME,
        "message": "Welcome to Aivora-Neural"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ai": APP_NAME,
        "model": HF_MODEL,
        "configured": bool(HF_TOKEN)
    }


# ----------------------------------
# AI CHAT
# ----------------------------------

@app.post("/chat")
async def chat(request: ChatRequest):

    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Hugging Face token is not configured on the server."
        )

    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Aivora-Neural, the AI assistant of Aivora AI. "
                "Be helpful, accurate, friendly and clear. "
                "Reply in the language used by the user whenever possible. "
                "Do not claim to have abilities or information you do not have."
            )
        }
    ]

    # Add recent conversation history
    for item in (request.history or [])[-10:]:
        if item.role in ["user", "assistant"]:
            messages.append({
                "role": item.role,
                "content": item.content[:5000]
            })

    messages.append({
        "role": "user",
        "content": user_message
    })

    headers = {
        "Authorization": "Bearer " + HF_TOKEN,
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 700
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:

            response = await client.post(
                HF_API_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            logger.error(
                "Hugging Face error: %s",
                response.text[:1000]
            )

            raise HTTPException(
                status_code=502,
                detail="AI model service error. Please try again later."
            )

        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        if not answer:
            raise ValueError("Empty AI response")

        return {
            "status": "success",
            "ai": APP_NAME,
            "model": HF_MODEL,
            "reply": answer
        }

    except HTTPException:
        raise

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI response timed out. Please try again."
        )

    except Exception as error:
        logger.exception("Chat error: %s", error)

        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the AI response."
        )


# ----------------------------------
# SERVER START
# ----------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
                           )
