import os
import httpx
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, AsyncGenerator
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
import docx

load_dotenv()

app = FastAPI(title="Health AI Chat Service")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client()
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

class ChatRequest(BaseModel):
    conversation_id: Optional[int] = None
    message: str

# ---------------------------------------------------------------------------
# Laravel helper functions — all with explicit timeouts so a slow/dead
# Laravel process never hangs a uvicorn worker indefinitely.
# ---------------------------------------------------------------------------

async def fetch_health_data(user_token: str):
    """Helper tool to fetch health data from Laravel"""
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/json"
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{LARAVEL_API_URL}/user/health", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch health data: {str(e)}"}

async def fetch_profile_data(user_token: str):
    """Helper tool to fetch user profile data from Laravel"""
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/json"
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{LARAVEL_API_URL}/user/profile", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch profile data: {str(e)}"}

# ---------------------------------------------------------------------------
# Non-blocking Gemini streaming helper
#
# The Gemini Python SDK exposes a *synchronous* iterator for streaming. Calling
# `for chunk in response:` inside an async function blocks the entire event
# loop — no other request can run while we wait for the next chunk.
#
# Fix: run the blocking iterator in a thread-pool worker via
# loop.run_in_executor(). A queue bridges the thread and the async generator
# so chunks are yielded asynchronously without blocking the event loop.
# ---------------------------------------------------------------------------

async def gemini_stream_chunks(
    contents_snapshot: list,
    config: types.GenerateContentConfig,
) -> AsyncGenerator:
    """
    Async generator that wraps the blocking Gemini SDK stream.
    The event loop stays free to serve other requests while Gemini chunks arrive.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _iterate():
        """Runs in a thread — calls the blocking Gemini SDK iterator."""
        try:
            response = gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents_snapshot,
                config=config,
            )
            for chunk in response:
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    # Start the blocking work in a thread — returns immediately
    loop.run_in_executor(None, _iterate)

    # Consume chunks from the queue asynchronously
    while True:
        kind, data = await queue.get()
        if kind == "done":
            return
        if kind == "error":
            raise data
        yield data  # kind == "chunk"


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(request: Request, chat_req: ChatRequest):
    # --- Auth ---
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    user_token = auth_header.replace("Bearer ", "").strip()

    # --- 1. Fetch chat context from Laravel ---
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {}
        if chat_req.conversation_id:
            payload["conversation_id"] = chat_req.conversation_id

        context_res = await client.get(
            f"{LARAVEL_API_URL}/internal/chat/context",
            headers=headers,
            params=payload,
        )

        if context_res.status_code != 200:
            error_message = "Unauthorized by Laravel backend"
            try:
                error_data = context_res.json()
                if isinstance(error_data, dict) and "message" in error_data:
                    error_message = error_data["message"]
                elif context_res.status_code >= 500:
                    error_message = "Internal server error from backend service."
            except Exception:
                if context_res.status_code >= 500:
                    error_message = "Internal server error from backend service."

            raise HTTPException(status_code=context_res.status_code, detail=error_message)

        context_data = context_res.json()
        conversation_id = context_data.get("conversation_id")
        db_messages = context_data.get("messages", [])

    # --- 2. Build conversation history for Gemini ---
    contents = []
    system_instruction = (
        "You are an expert health and wellness assistant. "
        "You have tools to fetch the user's health records and profile. "
        "Be concise, helpful, and format responses in Markdown."
    )

    for msg in db_messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=chat_req.message)])
    )

    # --- 3. Define AI tools ---
    ai_tools = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_health_data",
                description=(
                    "Fetches the user's current health records, charts, and metrics from the database. "
                    "Also includes the user's profile (age, gender, country, address). "
                    "Use this when the user asks about their health, medical records, or when profile "
                    "context would help answer a health question."
                ),
            ),
            types.FunctionDeclaration(
                name="get_user_profile",
                description=(
                    "Fetches the user's personal profile information: name, age, gender, country, "
                    "address, email, and phone. Use this ONLY when the user asks about their personal "
                    "details and does NOT need health records."
                ),
            ),
        ]
    )

    gemini_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[ai_tools],
        temperature=0.7,
    )

    # --- 4. Streaming response generator ---
    async def stream_generator():
        full_response = ""

        try:
            # ---- First Gemini call (non-blocking via thread pool) ----
            async for chunk in gemini_stream_chunks(list(contents), gemini_config):

                # If the browser tab was closed or user navigated away, stop immediately
                if await request.is_disconnected():
                    return

                if chunk.function_calls:
                    # Model wants to call a tool — handle it
                    contents.append(chunk.candidates[0].content)

                    for function_call in chunk.function_calls:

                        if function_call.name == "get_health_data":
                            yield "*(Fetching health records...)*\n\n"
                            health_data = await fetch_health_data(user_token)

                            # Process any attached documents
                            uploaded_files = []
                            extracted_texts = []

                            async def handle_file(path, name_for_log=None):
                                if not path or not os.path.exists(path):
                                    return
                                file_name = name_for_log or os.path.basename(path)
                                if path.endswith(".docx"):
                                    try:
                                        doc = await asyncio.to_thread(docx.Document, path)
                                        doc_content = "\n".join(p.text for p in doc.paragraphs)
                                        part = types.Part.from_text(
                                            text=f"\n\n--- Contents of {file_name} ---\n{doc_content}\n---------------------------\n"
                                        )
                                        extracted_texts.append(part)
                                    except Exception as e:
                                        print(f"Failed to read docx {path}: {e}")
                                else:
                                    try:
                                        uploaded = await asyncio.to_thread(
                                            gemini_client.files.upload, file=path
                                        )
                                        # Wait for Gemini to finish processing the file.
                                        # Uploading is async on Gemini's side — the file moves from
                                        # PROCESSING → ACTIVE before it can be used in a prompt.
                                        max_wait = 30  # seconds
                                        waited = 0
                                        while uploaded.state.name == "PROCESSING" and waited < max_wait:
                                            await asyncio.sleep(2)
                                            waited += 2
                                            uploaded = await asyncio.to_thread(
                                                gemini_client.files.get, name=uploaded.name
                                            )
                                        if uploaded.state.name == "ACTIVE":
                                            uploaded_files.append(uploaded)
                                        else:
                                            print(f"File {uploaded.name} did not become ACTIVE (state: {uploaded.state.name}), skipping.")
                                    except Exception as e:
                                        print(f"Failed to upload {path}: {e}")

                            for record in health_data.get("health_records", []):
                                if "detail_file_absolute_path" in record:
                                    path = record["detail_file_absolute_path"]
                                    if path and os.path.exists(path):
                                        yield f"*(Processing document: {os.path.basename(path)}...)*\n\n"
                                        await handle_file(path)

                                if "additional_files" in record:
                                    for f in record["additional_files"]:
                                        path = f.get("absolute_path")
                                        if path and os.path.exists(path):
                                            yield f"*(Processing additional file: {f.get('name', 'document')}...)*\n\n"
                                            await handle_file(path, f.get("name"))

                            contents.append(
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_function_response(
                                        name="get_health_data",
                                        response=health_data,
                                    )],
                                )
                            )
                            for uf in uploaded_files:
                                contents.append(uf)
                            if extracted_texts:
                                contents.append(types.Content(role="user", parts=extracted_texts))

                        elif function_call.name == "get_user_profile":
                            yield "*(Fetching profile...)*\n\n"
                            profile_data = await fetch_profile_data(user_token)
                            contents.append(
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_function_response(
                                        name="get_user_profile",
                                        response=profile_data,
                                    )],
                                )
                            )

                    # ---- Second Gemini call after tool results (also non-blocking) ----
                    async for second_chunk in gemini_stream_chunks(list(contents), gemini_config):
                        if await request.is_disconnected():
                            return
                        if second_chunk.text:
                            full_response += second_chunk.text
                            yield second_chunk.text

                    # Tool flow complete — exit the first stream loop
                    break

                elif chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # ---- 5. Save interaction to Laravel ----
            if full_response and not await request.is_disconnected():
                async with httpx.AsyncClient(timeout=30.0) as save_client:
                    await save_client.post(
                        f"{LARAVEL_API_URL}/internal/chat/save",
                        headers=headers,
                        json={
                            "conversation_id": conversation_id,
                            "user_message": chat_req.message,
                            "assistant_message": full_response,
                        },
                    )

        except asyncio.CancelledError:
            # Client disconnected mid-stream — exit cleanly, free the worker
            return
        except Exception as e:
            yield f"\n\nError: {str(e)}"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={
            "X-Conversation-ID": str(conversation_id),
            "Access-Control-Expose-Headers": "X-Conversation-ID",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
