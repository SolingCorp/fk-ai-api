import os
import httpx
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
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
    allow_origins=["*"], # Change this to your React app's URL in production
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

async def fetch_health_data(user_token: str):
    """Helper tool to fetch health data from Laravel"""
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{LARAVEL_API_URL}/user/health", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to fetch health data: {str(e)}"}

@app.post("/chat/stream")
async def chat_stream(request: Request, chat_req: ChatRequest):
    # Extract Bearer token from the incoming request
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    user_token = auth_header.replace('Bearer ', '').strip()
    
    # 1. Fetch Chat Context from Laravel
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        payload = {}
        if chat_req.conversation_id:
            payload['conversation_id'] = chat_req.conversation_id
            
        context_res = await client.get(
            f"{LARAVEL_API_URL}/internal/chat/context", 
            headers=headers,
            params=payload
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
        conversation_id = context_data.get('conversation_id')
        db_messages = context_data.get('messages', [])

    # 2. Format history for Gemini
    contents = []
    
    # Add a system instruction as the first message
    system_instruction = "You are an expert health and wellness assistant. You have a tool to fetch the user's health records if they ask about their data. Be concise, helpful, and format responses in Markdown."
    
    for msg in db_messages:
        role = 'user' if msg['role'] == 'user' else 'model'
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg['content'])])
        )
        
    # Add the new message
    contents.append(
        types.Content(role='user', parts=[types.Part.from_text(text=chat_req.message)])
    )

    # 3. Define the tool
    health_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_health_data",
                description="Fetches the user's current health records, charts, and metrics from the database.",
            )
        ]
    )

    async def stream_generator():
        full_response = ""
        
        # Start the chat session
        # Use generate_content_stream for streaming
        try:
            response = gemini_client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[health_tool],
                    temperature=0.7,
                ),
            )
            
            tool_called = False
            for chunk in response:
                # Check if Gemini decided to call a tool
                if chunk.function_calls:
                    tool_called = True
                    # Append the exact content chunk from the model, which includes the required thought_signatures
                    contents.append(chunk.candidates[0].content)
                    
                    for function_call in chunk.function_calls:
                        if function_call.name == "get_health_data":
                            # Yield a status message to the frontend so the user knows it's thinking
                            yield "*(Fetching health records...)*\n\n"
                            
                            # Execute the tool
                            health_data = await fetch_health_data(user_token)
                            
                            # Parse health_data for file paths and upload them
                            uploaded_files = []
                            extracted_texts = []
                            
                            async def handle_file(path, name_for_log=None):
                                if not path or not os.path.exists(path):
                                    return
                                    
                                file_name = name_for_log or os.path.basename(path)
                                
                                if path.endswith('.docx'):
                                    # Extract text directly for docx since Gemini doesn't support it well via File API
                                    try:
                                        doc = await asyncio.to_thread(docx.Document, path)
                                        full_text = []
                                        for para in doc.paragraphs:
                                            full_text.append(para.text)
                                        doc_content = '\n'.join(full_text)
                                        part = types.Part.from_text(text=f"\n\n--- Contents of {file_name} ---\n{doc_content}\n---------------------------\n")
                                        extracted_texts.append(part)
                                    except Exception as e:
                                        print(f"Failed to read docx {path}: {e}")
                                else:
                                    # Upload directly for images, pdfs, etc
                                    try:
                                        uploaded = await asyncio.to_thread(gemini_client.files.upload, file=path)
                                        uploaded_files.append(uploaded)
                                    except Exception as e:
                                        print(f"Failed to upload {path}: {e}")

                            for record in health_data.get('health_records', []):
                                if 'detail_file_absolute_path' in record:
                                    path = record['detail_file_absolute_path']
                                    if path and os.path.exists(path):
                                        yield f"*(Processing document: {os.path.basename(path)}...)*\n\n"
                                        await handle_file(path)
                                            
                                if 'additional_files' in record:
                                    for f in record['additional_files']:
                                        path = f.get('absolute_path')
                                        if path and os.path.exists(path):
                                            yield f"*(Processing additional file: {f.get('name', 'document')}...)*\n\n"
                                            await handle_file(path, f.get('name'))
                                                
                            # Add tool response to contents
                            function_parts = [types.Part.from_function_response(
                                name="get_health_data",
                                response=health_data
                            )]
                            contents.append(types.Content(role='user', parts=function_parts))
                            
                            # Append any uploaded files as separate user content blocks
                            for uf in uploaded_files:
                                contents.append(uf)
                                
                            # Append any extracted text parts into a new user content block
                            if extracted_texts:
                                contents.append(types.Content(role='user', parts=extracted_texts))
                            
                            
                    # Make a second call to get the actual answer based on the tool data
                    second_response = gemini_client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=[health_tool],
                            temperature=0.7,
                        ),
                    )
                    for second_chunk in second_response:
                        if second_chunk.text:
                            full_response += second_chunk.text
                            yield second_chunk.text
                    
                    # We break here because the second_response completes the interaction
                    break
                elif chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # 4. Save the interaction to Laravel after stream finishes
            async with httpx.AsyncClient() as save_client:
                await save_client.post(
                    f"{LARAVEL_API_URL}/internal/chat/save",
                    headers=headers,
                    json={
                        "conversation_id": conversation_id,
                        "user_message": chat_req.message,
                        "assistant_message": full_response
                    }
                )

        except Exception as e:
            yield f"\n\nError: {str(e)}"
            
    return StreamingResponse(
        stream_generator(), 
        media_type="text/plain",
        headers={
            "X-Conversation-ID": str(conversation_id),
            "Access-Control-Expose-Headers": "X-Conversation-ID"
        }
    )

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8001 to avoid conflicting with Laravel on 8000
    uvicorn.run(app, host="0.0.0.1", port=8001)
