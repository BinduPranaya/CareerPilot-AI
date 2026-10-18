"""Entrypoint: `python main.py` or `uvicorn main:app --reload`."""
from dotenv import load_dotenv

load_dotenv()

from app.api import app  # noqa: E402  (load_dotenv must run first)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
