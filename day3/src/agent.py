import ast
import datetime as dt
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

USE_FAKE = os.getenv("USE_FAKE") == "1"


def calculate(expression: str) -> float | int:
    """Safely calculate basic arithmetic without using eval()."""

    allowed = (
        ast.Expression,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(
                f"Unsupported expression: {type(node).__name__}"
            )

        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("Only numbers are allowed")

    return _eval_ast(tree.body)


def _eval_ast(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand)

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return +value

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        if isinstance(node.op, ast.Mult):
            return left * right

        if isinstance(node.op, ast.Div):
            return left / right

        if isinstance(node.op, ast.FloorDiv):
            return left // right

        if isinstance(node.op, ast.Mod):
            return left % right

        if isinstance(node.op, ast.Pow):
            return left**right

    raise ValueError("Unsupported expression")


def current_time() -> str:
    """Return the current local date and time."""

    return dt.datetime.now().astimezone().isoformat()


class FakeAgent:
    """Small fake agent with the same ainvoke interface."""

    async def ainvoke(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Fake agent reply: tools and agent boundary "
                        "are working."
                    ),
                }
            ]
        }


def build_agent() -> object:
    """Build the Day 3 agent behind one stable interface."""

    if USE_FAKE:
        return FakeAgent()

    project_root = Path(__file__).resolve().parents[1]

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    system_prompt = """
You are a Deep Agent for Day 3.

You have two important tools:

- calculate(expression): use it for arithmetic instead of calculating mentally.
- current_time(): use it whenever the user asks for the current time.

If the user asks a question that requires one of these tools, you MUST call
the appropriate tool before answering.

Do not invent tool results.
"""

    return create_deep_agent(
        model=llm,
        tools=[calculate, current_time],
        system_prompt=system_prompt,
        backend=FilesystemBackend(
            root_dir=str(project_root),
            virtual_mode=True,
        ),
        skills=["/skills/"],
    )


async def main() -> None:
    agent = build_agent()

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 * 23 and what time is it?",
                }
            ]
        }
    )

    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    
