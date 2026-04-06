from fastapi import FastAPI

from app import create_app


def test_create_app_returns_fastapi() -> None:
    app: FastAPI = create_app()
    assert isinstance(app, FastAPI)
