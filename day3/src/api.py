import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import build_agent


app = FastAPI(title="Day 3 Agent API")
agent = build_agent()


class ResponseRequest(BaseModel):
    input: str
    model: str | None = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/responses")
async def responses(request: ResponseRequest):
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.input,
                }
            ]
        }
    )

    last_message = response["messages"][-1]
    text = last_message["content"]
    

    return {
        "id": f"resp_{int(time.time())}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": request.model or "day3-agent",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {"todo": True}
