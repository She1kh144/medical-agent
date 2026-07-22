import json
import queue
import threading
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import run_agent

app = FastAPI(title="Medical Agent API")
DONE = object()

class RunRequest(BaseModel):
    question: str

def sse_message(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse("static/index.html")

@app.post("/run")
def run(request: RunRequest) -> dict[str, object]:
    """Blocking run: returns the complete trace when the agent finishes."""
    try:
        return run_agent(request.question)
    except RuntimeError:
        return {"outcome": "timeout", "steps": None, "messages": []}

@app.post("/run/stream")
def run_stream(request: RunRequest) -> StreamingResponse:
    """Streaming run: emits agent events as SSE while the agent works."""
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            trace = run_agent(
                request.question,
                on_event=lambda kind, payload: events.put((kind, payload)),
            )
            events.put(("done", {"outcome": trace["outcome"], "steps": trace["steps"]}))
        except RuntimeError:
            events.put(("done", {"outcome": "timeout", "steps": None}))
        except Exception as error:
            events.put(("error", {"detail": str(error)}))
        finally:
            events.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = events.get()

            if item is DONE:
                break

            kind, payload = item
            yield sse_message(kind, payload)

    return StreamingResponse(generate(), media_type="text/event-stream")