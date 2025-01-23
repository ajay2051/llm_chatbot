import os

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from langserve import add_routes
from pydantic import BaseModel
from redis import Redis
from requests import Request
from starlette.responses import JSONResponse

from chatbot import get_response
from db_as_dataset import get_db_connection

load_dotenv()

API_KEY_HEADER = "X-API-KEY"
API_ACCESS_KEY = os.environ.get("API_ACCESS_KEY")
API_ACCESS_USER = os.environ.get("API_ACCESS_USER")

app = FastAPI()

redis_instance = Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=os.environ.get("REDIS_PORT", 6379),
    password=os.environ.get("REDIS_PASSWORD", "<PASSWORD>"),
    ssl=True
)

chain = get_response(query="Which language do you want to use?")


# Initialize FastAPI router
router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/process-query")
async def process_query(request: QueryRequest):
    try:
        result = get_response(query=request.query)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Add routes
# add_routes(app, chain, path="/langchain")
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    rate_limit = 100
    rate_window = 10 # seconds
    # api_key = request.headers.get(API_KEY_HEADER)
    # print("API Key: ", api_key)
    # if api_key is None:
    #     return JSONResponse(status_code=400, content={"message": "Missing API KEY"})
    visits_key = "visits"
    current_value = redis_instance.get(visits_key)

    if current_value is None:
        await redis_instance.set(visits_key, rate_window)
    redis_instance.incr(visits_key)
    final_visits = redis_instance.get(visits_key)
    do_rate_limit = False
    try:
        do_rate_limit = int(final_visits) > rate_limit
    except ValueError:
        do_rate_limit = True
    if do_rate_limit:
        return JSONResponse(status_code=429, content={"error": "Rate Limit Exceeded"})
    response = await call_next(request)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
