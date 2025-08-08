from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai import errors
import aiohttp

from dotenv import load_dotenv
import io
import asyncio
import json
import os
import logging
from contextlib import asynccontextmanager
from itertools import cycle
import threading
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(message)s',
)

load_dotenv()

# --- Custom Exceptions ---
class DocumentDownloadError(Exception):
    """Raised when PDF download fails"""
    pass

class DocumentProcessingTimeoutError(Exception):
    """Raised when document processing times out"""
    pass

class FileUploadError(Exception):
    """Raised when file upload to Gemini fails"""
    pass

class ContentGenerationError(Exception):
    """Raised when content generation fails"""
    pass

# --- Global Variables ---
http_session: Optional[aiohttp.ClientSession] = None
api_keys: List[str] = []
key_cycle = None
key_lock = threading.Lock()

# --- Configuration ---
EXPECTED_BEARER_TOKEN = "bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d"
MODEL_NAME = "gemini-2.5-pro"
PROCESSING_TIMEOUT_SECONDS = 300
PROCESSING_POLL_INTERVAL = 2

# --- Helper Functions ---
def load_api_keys():
    """Load all available Gemini API keys from environment variables"""
    keys = []
    # Check for primary key
    primary_key = os.getenv("GEMINI_API_KEY")
    if primary_key:
        keys.append(primary_key)
    
    # Check for additional keys
    for i in range(1, 4):  # GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    
    if not keys:
        raise RuntimeError("No Gemini API keys found in environment variables")
    
    return keys

def get_next_api_key() -> str:
    """Get the next API key in rotation (thread-safe)"""
    with key_lock:
        return next(key_cycle)

def create_gemini_client(api_key: str) -> genai.Client:
    """Create a Gemini client with the specified API key"""
    return genai.Client(api_key=api_key)

# --- Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global http_session, api_keys, key_cycle
    
    # Load API keys
    api_keys = load_api_keys()
    key_cycle = cycle(api_keys)
    logging.info(f"Loaded {len(api_keys)} Gemini API keys")
    
    # Create HTTP session
    timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 min total, 30s connect
    http_session = aiohttp.ClientSession(timeout=timeout)
    logging.info("HTTP session created")
    
    yield
    
    # Shutdown
    if http_session:
        await http_session.close()
        logging.info("HTTP session closed")

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class GeminiAnswers(BaseModel):
    answers: List[str] = Field(description="A list of precise answers to the questions, in the same order.")

class HealthResponse(BaseModel):
    status: str
    features: List[str]
    supported_formats: List[str]
    active_api_keys: int

class KeyTestResult(BaseModel):
    key_index: int
    status: str
    error: Optional[str] = None

class KeyTestResponse(BaseModel):
    results: List[KeyTestResult]
    total_keys: int
    working_keys: int

# --- Security ---
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    if credentials.credentials != EXPECTED_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

# --- Health Check ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        features=[
            "Async HTTP with connection pooling",
            "API key rotation",
            "Granular error handling",
            "PDF document analysis",
            "Insurance/Policy document specialization"
        ],
        supported_formats=["PDF"],
        active_api_keys=len(api_keys)
    )

# --- API Key Testing ---
@app.get("/test-keys", response_model=KeyTestResponse)
async def test_api_keys(_=Depends(verify_token)):
    """Test all available Gemini API keys"""
    results = []
    working_keys = 0
    
    for i, api_key in enumerate(api_keys):
        try:
            client = create_gemini_client(api_key)
            # If we get here without exception, the key works
            results.append(KeyTestResult(key_index=i, status="working"))
            working_keys += 1
        except Exception as e:
            results.append(KeyTestResult(
                key_index=i, 
                status="failed", 
                error=str(e)
            ))
    
    return KeyTestResponse(
        results=results,
        total_keys=len(api_keys),
        working_keys=working_keys
    )

# --- Document Processing Functions ---
async def download_pdf(url: str) -> bytes:
    """Download PDF from URL using aiohttp"""
    try:
        async with http_session.get(url) as response:
            if response.status != 200:
                raise DocumentDownloadError(f"Failed to download PDF: HTTP {response.status}")
            
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type:
                raise DocumentDownloadError("The provided URL does not point to a PDF document.")
            
            return await response.read()
    except aiohttp.ClientError as e:
        raise DocumentDownloadError(f"Network error while downloading PDF: {str(e)}")
    except asyncio.TimeoutError:
        raise DocumentDownloadError("Timeout while downloading PDF")

async def upload_and_process_pdf(client: genai.Client, pdf_data: bytes, filename: str) -> object:
    """Upload PDF to Gemini and wait for processing"""
    try:
        # Upload file
        uploaded_file = client.files.upload(
            file=io.BytesIO(pdf_data),
            config=types.UploadFileConfig(
                mime_type='application/pdf',
                display_name=filename
            )
        )
        
        # Wait for processing with timeout
        start_time = asyncio.get_event_loop().time()
        while True:
            get_file = client.files.get(name=uploaded_file.name)
            
            if get_file.state.name == 'ACTIVE':
                logging.info("File processed successfully")
                return uploaded_file
            elif get_file.state.name == 'FAILED':
                error_message = get_file.error.message if get_file.error else 'Unknown error'
                raise FileUploadError(f"PDF processing failed: {error_message}")
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > PROCESSING_TIMEOUT_SECONDS:
                raise DocumentProcessingTimeoutError(
                    f"Document processing timed out after {PROCESSING_TIMEOUT_SECONDS} seconds"
                )
            
            # Wait before next check
            await asyncio.sleep(PROCESSING_POLL_INTERVAL)
            
    except errors.APIError as e:
        raise FileUploadError(f"Failed to upload file to Gemini: {e.message}")

async def generate_answers(client: genai.Client, uploaded_file: object, questions: List[str]) -> List[str]:
    """Generate answers using Gemini"""
    # Improved prompt for insurance/policy documents
    prompt = f"""You are an expert analyst specializing in insurance policies, legal documents, and regulatory compliance.

    DOCUMENT CONTEXT: This document contains insurance policy information, terms, conditions, coverage details, and/or regulatory compliance requirements.

    ANALYSIS INSTRUCTIONS:
    - Extract information ONLY from the provided document
    - Consider the document's and question's context and content
    - Provide precise, factual answers
    - Include specific details: amounts, percentages, time periods, eligibility criteria
    - Use exact terminology from the document
    - For coverage questions, specify what IS and IS NOT covered
    - For procedural questions, provide step-by-step requirements

    QUESTIONS ({len(questions)} total):
    """
    
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    
    prompt += f"""
    RESPONSE FORMAT:
    - Provide exactly {len(questions)} answers in the specified JSON format
    - Each answer should be complete, accurate, and directly address the question
    - Maintain document terminology and be concise yet comprehensive
    - If information is not in the document, state "Information not found in document"
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                uploaded_file,
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GeminiAnswers,
                # thinking_config=types.ThinkingConfig(thinking_budget=0),
                maxOutputTokens=60000
            )
        )
        
        parsed_json = json.loads(response.text)
        answers = parsed_json.get('answers', [])
        
        if len(answers) != len(questions):
            raise ContentGenerationError(
                f"Expected {len(questions)} answers, got {len(answers)}"
            )
        
        return answers
        
    except errors.APIError as e:
        raise ContentGenerationError(f"Gemini API Error: {e.message}")
    except json.JSONDecodeError as e:
        raise ContentGenerationError(f"Failed to parse JSON response: {str(e)}")

# --- Main Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(request: QueryRequest, _=Depends(verify_token)):
    start_time = asyncio.get_event_loop().time()
    uploaded_file = None
    api_key = get_next_api_key()
    client = None
    try:
        # 1. Download PDF
        logging.info(f"Downloading PDF from: {request.documents}")
        pdf_data = await download_pdf(request.documents)
        logging.info(f"Downloaded PDF: {len(pdf_data)} bytes")

        # 2. Create Gemini client and upload PDF
        client = create_gemini_client(api_key)
        filename = request.documents.split('/')[-1]
        upload_start = asyncio.get_event_loop().time()
        logging.info(f"Starting upload to Gemini at {upload_start:.2f}s")
        uploaded_file = await upload_and_process_pdf(client, pdf_data, filename)
        upload_end = asyncio.get_event_loop().time()
        logging.info(f"Upload finished at {upload_end:.2f}s (duration: {upload_end-upload_start:.2f}s)")

        # 3. Generate answers with retry logic
        try:
            answers = await generate_answers(client, uploaded_file, request.questions)
        except ContentGenerationError as e:
            logging.warning(f"Content generation failed, retrying with fresh client: {e}")
            if len(api_keys) > 1:
                retry_key = get_next_api_key()
                retry_client = create_gemini_client(retry_key)
                if uploaded_file:
                    try:
                        client.files.delete(name=uploaded_file.name)
                    except:
                        pass
                upload_retry_start = asyncio.get_event_loop().time()
                logging.info(f"Retrying upload to Gemini at {upload_retry_start:.2f}s")
                uploaded_file = await upload_and_process_pdf(retry_client, pdf_data, filename)
                upload_retry_end = asyncio.get_event_loop().time()
                logging.info(f"Retry upload finished at {upload_retry_end:.2f}s (duration: {upload_retry_end-upload_retry_start:.2f}s)")
                answers = await generate_answers(retry_client, uploaded_file, request.questions)
                client = retry_client
            else:
                raise e

        duration = asyncio.get_event_loop().time() - start_time
        logging.info(f"/hackrx/run processed in {duration:.2f} seconds")
        return QueryResponse(answers=answers)

    except DocumentDownloadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (FileUploadError, DocumentProcessingTimeoutError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ContentGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if uploaded_file and client:
            try:
                logging.info(f"Deleting uploaded file: {uploaded_file.name}")
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to delete uploaded file: {e}")

# To run: uvicorn main:app --reload