from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Initial FastAPI application"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)