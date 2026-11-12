# -*- coding: utf-8 -*-
import os
from typing import Literal
from model_utils import load_model, make_inference
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field


class Instance(BaseModel):
    """Raw data about an insurance client."""
    age: int = Field(..., ge=18, le=100, description="Age, years")
    sex: Literal["female", "male"]
    bmi: float = Field(..., gt=10, lt=70, description="Body mass index")
    children: int = Field(..., ge=0, le=10,
                          description="Number of covered children")
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]

    class Config:
        schema_extra = {
            "example": {
                "age": 19, "sex": "female", "bmi": 27.9, "children": 0,
                "smoker": "yes", "region": "southwest",
            }
        }


app = FastAPI(
    title="Medical Cost Inference Service",
    description="Predicts yearly medical insurance charges (USD)",
    version="1.0.0",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")
model_path: str = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def is_token_correct(token: str) -> bool:
    dummy_correct_token = "00000"
    return token == dummy_correct_token


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    if not await is_token_correct(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions")
async def predictions(instance: Instance,
                      token: str = Depends(check_token)) -> dict[str, float]:
    return make_inference(load_model(model_path), instance.dict())
