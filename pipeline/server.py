import uvicorn
from fastapi import FastAPI, APIRouter

from src.routers import predict_router

app = FastAPI()
app.include_router(predict_router)


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
