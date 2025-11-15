"""
API Server para bike sharing

Uso:
    uvicorn bike_sharing.api_server:app --reload --port 8000
"""

from fastapi import FastAPI

app = FastAPI(title="Bike Sharing API")


@app.get("/")
async def root():
    """Endpoint raíz."""
    return {"message": "Hello world"}
