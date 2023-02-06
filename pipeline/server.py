import uvicorn
from fastapi import FastAPI

from src.routes import predict_router

app = FastAPI()
app.include_router(predict_router)


if __name__ == '__main__':
    uvicorn.run(app, port=1928, host='0.0.0.0')
