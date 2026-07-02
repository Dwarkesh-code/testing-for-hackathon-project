"""
Backend API Server — lightweight purely asynchronous web server using built-in / aiohttp.
Serves the modern frontend portfolio UI and connects to the LangGraph pipeline.

Run: python server.py
Open: http://localhost:8080
"""

import asyncio
import json
import os
from aiohttp import web
from main import run_pipeline


async def handle_generate(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        username = data.get("username") or "Dwarkesh-code"
        leetcode = data.get("leetcode") or ""
        linkedin = data.get("linkedin") or ""
        credly = data.get("credly") or ""

        print(f"\n[API] Portfolio requested for: {username}")
        portfolio = await run_pipeline(username, leetcode, linkedin, credly)
        
        return web.json_response({"status": "success", "portfolio": portfolio})
    except Exception as e:
        print(f"[API Error] {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)


async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse("static/index.html")


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def init_app() -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    
    # API endpoints
    app.router.add_post("/api/generate-portfolio", handle_generate)
    app.router.add_options("/api/generate-portfolio", handle_generate)
    
    # Static files & index
    app.router.add_get("/", handle_index)
    if os.path.exists("static"):
        app.router.add_static("/", "static")
        
    return app


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app = init_app()
    print("\n🚀 Starting Proof-of-Work AI Portfolio Platform on http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080)
