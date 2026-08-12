import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

load_dotenv()

WORK_DIR = Path(__file__).resolve().parents[1] / "work"


def make_backend():
    backend = LocalShellBackend(
        root_dir=str(WORK_DIR),
        virtual_mode=True,
        env={"PATH": os.environ["PATH"]},
    )
    return backend, lambda: None


async def main() -> None:
    backend, cleanup = make_backend()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    system_prompt = """
You are a Deep Agent for Day 4.

You have access to a shell through the execute tool.

Complete the user's task carefully. Use the shell to create files,
run tests, inspect failures, and fix problems until the tests pass.

Do not invent test results. Report the actual final pytest output.
"""

    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        backend=backend,
    )

    task = """
1. Create calculator.py with add, sub, mul, and div functions.
   div must raise an exception when dividing by zero.

2. Write pytest tests covering the functions, including the zero case.

3. Run the tests using execute with:
   python -m pytest

   If pytest is missing, install it first.

4. Fix any failures until all tests are green.

5. Report the final pytest output.
"""

    try:
        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task,
                    }
                ]
            }
        )
        print(response["messages"][-1].content)
    finally:
        cleanup()


if __name__ == "__main__":
    asyncio.run(main())
    
