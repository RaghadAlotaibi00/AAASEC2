import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth import require_scopes
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

load_dotenv()

STUDENT_TOKEN = os.getenv("MCP_STUDENT_TOKEN", "student-secret-token")
ADMIN_TOKEN = os.getenv("MCP_ADMIN_TOKEN", "admin-secret-token")

verifier = StaticTokenVerifier(
    tokens={
        STUDENT_TOKEN: {
            "client_id": "student",
            "scopes": ["read:public"],
        },
        ADMIN_TOKEN: {
            "client_id": "admin",
            "scopes": ["read:public", "read:internal"],
        },
    }
)

mcp = FastMCP("Secure Tools", auth=verifier)


@mcp.tool
def get_server_time() -> str:
    """Return the current server time."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool(auth=require_scopes("read:internal"))
def get_internal_report() -> dict:
    """Return quarterly financial data for authorized users."""
    return {
        "months": [
            {"month": "January", "revenue": 12000, "costs": 7000},
            {"month": "February", "revenue": 15000, "costs": 8000},
            {"month": "March", "revenue": 18000, "costs": 9000},
        ]
    } 




@mcp.tool(auth=require_scopes("read:internal"))
def get_lab_inventory() -> dict:
    """Return the protected lab inventory."""
    return {
        "laptops": 12,
        "sensors": 8,
        "routers": 4,
        "servers": 3,
    }
@mcp.tool(auth=require_scopes("read:internal"))
def get_course_grades() -> dict:
    """Return protected course grades."""
    return {
        "students": [
            {"name": "A", "grade": 88},
            {"name": "B", "grade": 92},
            {"name": "C", "grade": 76},
            {"name": "D", "grade": 84},
        ]
    }
if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8002,
    )
@mcp.tool(auth=require_scopes("read:internal"))
def get_lab_inventory() -> dict:
    """Return the protected lab inventory."""
    return {
        "status": "ok",
        "items": {
            "laptops": 12,
            "sensors": 8,
            "routers": 4,
        },
    }
