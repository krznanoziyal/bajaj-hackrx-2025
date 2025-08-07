from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types, errors
import requests
import asyncio
import aiohttp
import io
import time
import json
import random
import os
# from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
security = HTTPBearer(auto_error=True)

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"
EXPECTED_TOKEN = "bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d"

# Multiple API Keys for rotation
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3")
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None]

# Global aiohttp session for connection pooling
http_session = None

@app.on_event("startup")
async def startup_event():
    global http_session
    # Create persistent HTTP session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=100,  # Total connection limit
        limit_per_host=30,  # Per host limit
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

@app.on_event("shutdown")
async def shutdown_event():
    global http_session
    if http_session:
        await http_session.close()

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class GeminiAnswers(BaseModel):
    answers: List[str] = Field(description="A list of answers to the questions, in the same order.")

# --- Authentication ---
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Speed-Optimized PDF Download ---
async def download_pdf_async(url: str) -> bytes:
    """Download PDF using async HTTP with connection pooling"""
    global http_session
    
    try:
        async with http_session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download PDF: HTTP {response.status}")
            
            content = await response.read()
            
            # Quick PDF validation
            if not content.startswith(b'%PDF'):
                raise HTTPException(status_code=400, detail="Invalid PDF document")
                
            return content
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Download error: {str(e)}")

# --- Optimized Prompt (Reduced Token Count) ---
def build_speed_optimized_prompt(questions: List[str]) -> str:
    """Build a concise, speed-focused prompt"""
    
    prompt = f"""You are an expert insurance analyst. Analyze the document with precision and speed.

ANALYSIS METHOD:
- Deep contextual reading beyond keywords
- Cross-reference sections for complete understanding  
- Apply insurance industry knowledge
- Reference specific clauses/sections

FOR EACH QUESTION:
1. Locate relevant policy sections
2. Analyze conditions & exclusions
3. Apply logic & context
4. Provide complete answer with details

QUESTIONS ({len(questions)}):
"""
    
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"

    prompt += f"""
ANSWER FORMAT: [Direct Answer] → [Key Conditions] → [Specific Details] → [Policy References]

REQUIREMENTS:
- Professional claims-level analysis
- Include amounts, percentages, timeframes
- Reference specific policy sections
- Address all question components
- Show reasoning chain from policy to conclusion

Return JSON with "answers" array containing exactly {len(questions)} comprehensive responses."""

    return prompt

# --- Fast Gemini Processing ---
def get_gemini_client():
    """Get Gemini client with random API key"""
    api_key = random.choice(GEMINI_API_KEYS)
    return genai.Client(api_key=api_key)

async def process_with_gemini(pdf_data: bytes, prompt: str, questions_count: int) -> List[str]:
    """Process PDF with Gemini using optimized settings"""
    
    client = get_gemini_client()
    uploaded_file = None
    
    try:
        # Upload PDF
        pdf_io = io.BytesIO(pdf_data)
        uploaded_file = client.files.upload(
            file=pdf_io,
            config=types.UploadFileConfig(
                mime_type='application/pdf',
                display_name=f"doc_{int(time.time())}.pdf"
            )
        )
        
        # Fast polling with shorter intervals
        max_wait = 60  # Maximum wait time
        start_wait = time.time()
        
        while True:
            file_status = client.files.get(name=uploaded_file.name)
            
            if file_status.state.name == 'ACTIVE':
                break
            elif file_status.state.name == 'FAILED':
                error_msg = file_status.error.message if file_status.error else 'Unknown error'
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {error_msg}")
            elif time.time() - start_wait > max_wait:
                raise HTTPException(status_code=504, detail="PDF processing timeout")
                
            await asyncio.sleep(2)  # Reduced from 5 to 2 seconds

        # Generate content with speed optimizations
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GeminiAnswers,
                thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disable thinking for speed
                maxOutputTokens=30000,  # Reduced from 60000
                temperature=0.1,  # Lower temperature for faster, more deterministic responses
                candidateCount=1,  # Only generate one candidate
                top_p=0.8,  # Reduced for speed
                top_k=40   # Reduced for speed
            )
        )

        parsed_json = json.loads(response.text)
        answers = parsed_json.get('answers', [])
        
        if len(answers) != questions_count:
            # Fallback: pad with empty strings if needed
            while len(answers) < questions_count:
                answers.append("Unable to process this question due to processing constraints.")
        
        return answers[:questions_count]  # Ensure exact count
        
    finally:
        # Cleanup
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except:
                pass  # Ignore cleanup errors

# --- Optimized Main Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest, 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    start_time = time.time()
    
    try:
        # Parallel operations where possible
        pdf_download_task = download_pdf_async(request.documents)
        prompt = build_speed_optimized_prompt(request.questions)  # Build prompt while downloading
        
        # Wait for PDF download
        pdf_data = await pdf_download_task
        
        # Process with Gemini
        answers = await process_with_gemini(pdf_data, prompt, len(request.questions))
        
        duration = time.time() - start_time
        print(f"Request processed in {duration:.2f} seconds")
        
        return QueryResponse(answers=answers)

    except HTTPException:
        raise
    except errors.APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e.message}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "gemini_keys": len(GEMINI_API_KEYS),
        "optimization": "speed_focused"
    }