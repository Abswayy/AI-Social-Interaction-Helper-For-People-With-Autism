"""HTTP + WebSocket server.

    python -m server.main            -> open http://localhost:8000

Any client (laptop browser, phone companion app, glasses firmware) talks the
same WebSocket protocol, documented in PROTOCOL.md.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .face import FaceEngine, decode_jpeg, encode_jpeg
from .session import DeviceSession
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("main")

WEB = config.ROOT / "web"
app = FastAPI(title="Social Cue Glasses")
app.mount("/static", StaticFiles(directory=WEB), name="static")

_NO_STORE = {"Cache-Control": "no-store"}

store = Store()
faces: FaceEngine | None = None
stt = None

MSG_AUDIO, MSG_FRAME = 1, 2      # first byte of binary WebSocket messages


@app.on_event("startup")
async def _load_models():
    global faces, stt
    from .stt import make_transcriber
    from .llm import resolve_models
    await resolve_models()
    loop = asyncio.get_running_loop()
    faces = await loop.run_in_executor(None, FaceEngine)
    stt = await loop.run_in_executor(None, make_transcriber)
    log.info("Models ready. Open http://localhost:%s", config.PORT)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers=_NO_STORE)


@app.get("/register")
def register_page():
    return FileResponse(WEB / "register.html")


@app.get("/people")
def people_page():
    return FileResponse(WEB / "people.html")


@app.get("/health")
def health():
    return {"faces": faces is not None, "stt": stt is not None, "database": store.backend,
            "people": len(store.list_people())}


# ------------------------------------------------------------------ people API
class RegisterBody(BaseModel):
    name: str
    relationship: str = ""
    photos: list[str]          # base64 JPEG/PNG (data URLs allowed)


@app.post("/api/register")
def register(body: RegisterBody):
    """Self-registration: a person enrols their own face (with consent)."""
    if faces is None:
        raise HTTPException(503, "Face models are still loading. Try again in a few seconds.")
    found_faces = []
    for b64 in body.photos[:5]:
        try:
            img = decode_jpeg(base64.b64decode(b64.split(",")[-1]))
        except Exception:
            img = None
        if img is None:
            continue
        found = faces.analyze(img, max_faces=2)
        if len(found) == 1:
            found_faces.append(found[0])
    if not found_faces:
        raise HTTPException(400, "No single clear face found. Use a well-lit, front-facing photo with only you in it.")
    # If the glasses already auto-registered this face, claim that record instead of duplicating it.
    existing, _ = store.match(found_faces[0].embedding)
    pid = existing or store.create_person(body.name.strip(), source="self")
    store.rename(pid, body.name)
    if body.relationship.strip():
        store.update_person(pid, profile={"relationship": body.relationship})
    for f in found_faces:
        store.add_embedding(pid, f.embedding, force=existing is None)
    best = max(found_faces, key=lambda f: f.score)
    if best.crop is not None:
        store.set_photo(pid, encode_jpeg(best.crop), only_if_missing=False)
    return {"id": pid, "name": body.name, "linked_existing": existing is not None,
            "photos_used": len(found_faces)}


@app.get("/api/people")
def people():
    return [p.public() for p in store.list_people()]


@app.get("/api/people/{pid}")
def person(pid: int):
    p = store.get(pid)
    if not p:
        raise HTTPException(404, "Person not found")
    return {**p.public(), "conversations": store.conversations(pid)}


@app.get("/api/people/{pid}/photo")
def person_photo(pid: int):
    data = store.photo(pid)
    if not data:
        raise HTTPException(404, "No photo")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


class PersonPatch(BaseModel):
    name: str | None = None
    notes: str | None = None
    profile: dict | None = None     # relationship, designation, recall_hint, key_facts, interests, communication_tips, follow_ups


@app.patch("/api/people/{pid}")
def edit_person(pid: int, body: PersonPatch):
    p = store.update_person(pid, name=body.name, notes=body.notes, profile=body.profile)
    if not p:
        raise HTTPException(404, "Person not found")
    return p.public()


@app.delete("/api/people/{pid}")
def delete(pid: int):
    store.delete_person(pid)
    return {"ok": True}


@app.post("/api/people/{keep}/merge/{drop}")
def merge(keep: int, drop: int):
    if keep == drop or not store.get(keep) or not store.get(drop):
        raise HTTPException(400, "Pick two different people")
    store.merge(keep, drop)
    return store.get(keep).public()


# ------------------------------------------------------------------ conversations API
@app.get("/api/conversations/unassigned")
def unassigned():
    return store.conversations(None)


@app.get("/api/conversations/{cid}")
def conversation(cid: int):
    c = store.conversation(cid)
    if not c:
        raise HTTPException(404, "Conversation not found")
    return c


class AssignBody(BaseModel):
    person_id: int | None


@app.patch("/api/conversations/{cid}")
def assign(cid: int, body: AssignBody):
    if not store.conversation(cid):
        raise HTTPException(404, "Conversation not found")
    if body.person_id is not None and not store.get(body.person_id):
        raise HTTPException(404, "Person not found")
    store.assign_conversation(cid, body.person_id)
    return store.conversation(cid)


@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: int):
    store.delete_conversation(cid)
    return {"ok": True}


@app.get("/api/export")
def export():
    return JSONResponse(store.export(), headers={"Content-Disposition": "attachment; filename=people-export.json"})


# ------------------------------------------------------------------ live stream
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    if stt is None or faces is None:
        await websocket.send_json({"type": "status", "ready": False, "message": "Models loading, retry in a moment"})
        await websocket.close()
        return

    async def send(msg: dict):
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            pass

    session = DeviceSession(send, store, faces, stt)
    await send({"type": "status", "ready": True})
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                data = msg["bytes"]
                if data[0] == MSG_AUDIO:
                    await session.on_audio(data[1:])
                elif data[0] == MSG_FRAME:
                    asyncio.create_task(session.on_frame(data[1:]))
            elif msg.get("text"):
                await session.control(json.loads(msg["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


def run():
    ssl = {}
    if os.getenv("SSL_CERTFILE"):
        # Needed when a phone/glasses opens the page over Wi-Fi (camera/mic require HTTPS).
        ssl = {"ssl_certfile": os.getenv("SSL_CERTFILE"), "ssl_keyfile": os.getenv("SSL_KEYFILE")}
    uvicorn.run(app, host=config.HOST, port=config.PORT, **ssl)


if __name__ == "__main__":
    run()
