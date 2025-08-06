from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types, errors
import requests
from dotenv import load_dotenv
import io
import time
import json
import random
import os

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

# Filter out None values in case some keys aren't set
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None]

# If no additional keys, fall back to main key
if not GEMINI_API_KEYS:
    GEMINI_API_KEYS = [os.getenv("GEMINI_API_KEY")]

print(f"Loaded {len(GEMINI_API_KEYS)} Gemini API keys for rotation")

def get_gemini_client():
    """Get a Gemini client with a randomly selected API key"""
    api_key = random.choice(GEMINI_API_KEYS)
    return genai.Client(api_key=api_key)

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

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest, 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify the token
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    start_time = time.time()
    uploaded_file = None

    try:
        # 1. Download PDF
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
        
        # Check content type, but be more flexible for cloud storage services
        content_type = pdf_response.headers.get('Content-Type', '').lower()
        content = pdf_response.content
        
        # Validate it's actually a PDF by checking the file signature
        if not content.startswith(b'%PDF'):
            raise HTTPException(status_code=400, detail="The provided URL does not point to a valid PDF document.")
        
        pdf_data = io.BytesIO(content)

        # 2. Upload PDF to Gemini
        client = get_gemini_client()
        uploaded_file = client.files.upload(
            file=pdf_data,
            config=types.UploadFileConfig(
                mime_type='application/pdf',
                display_name=request.documents.split('/')[-1]
            )
        )

        # 3. Wait for processing
        get_file = client.files.get(name=uploaded_file.name)
        while get_file.state.name == 'PROCESSING':
            time.sleep(5)
            get_file = client.files.get(name=uploaded_file.name)

        if get_file.state.name == 'FAILED':
            error_message = get_file.error.message if get_file.error else 'Unknown'
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {error_message}")

        # 4. Build Prompt
        prompt = f"""
    You are an expert document analyst specializing in processing and interpreting various types of documents including insurance policies, legal contracts, HR documents, compliance materials, technical manuals, business reports, and other professional documentation. Your role is to provide comprehensive, contextually grounded analysis based on document content.

    DOCUMENT ANALYSIS FRAMEWORK:
    1. **Deep Contextual Reading**: Understand the document's structure, purpose, and interconnected clauses
    2. **Multi-layered Analysis**: Consider explicit statements, implicit conditions, and logical relationships
    3. **Professional Reasoning**: Apply industry knowledge to interpret complex policy language
    4. **Comprehensive Coverage**: Address all aspects of each question with supporting evidence

    CRITICAL ANALYSIS REQUIREMENTS:

    CONTEXT GROUNDING:
    - Never rely on simple keyword matching
    - Understand the PURPOSE behind each clause and condition
    - Consider how different sections of the document interact
    - Identify underlying logic and rationale for policy terms
    - Reference specific section numbers, clause identifiers, or page locations when available

    ANSWER COMPLETENESS:
    - Address EVERY component of multi-part questions
    - Provide specific details: amounts, percentages, time periods, conditions
    - Include eligibility criteria, exclusions, and special circumstances  
    - Explain the reasoning chain from policy text to conclusion
    - Quote exact terminology and definitions from the document

    PROFESSIONAL STANDARDS:
    - Use precise insurance/legal terminology as found in the document
    - Structure answers logically: main point → conditions → exceptions → supporting details
    - Provide actionable information a claims processor would need
    - Include relevant cross-references between related policy sections

    QUESTIONS TO ANALYZE:
    """

        for i, q in enumerate(request.questions):
            prompt += f"{i+1}. {q}\n"

        prompt += f"""

    RESPONSE METHODOLOGY:
    For each question, follow this analytical process:
    1. **Locate Relevant Sections**: Identify all document sections that relate to the question
    2. **Extract Key Information**: Pull specific terms, conditions, amounts, and requirements
    3. **Analyze Relationships**: Understand how different clauses work together
    4. **Apply Logic**: Reason through the policy's intent and application
    5. **Provide Complete Answer**: Cover all question components with supporting evidence

    ANSWER STRUCTURE REQUIREMENTS:
    - Lead with the direct answer to the primary question
    - Follow with specific conditions, requirements, or limitations
    - Include exact amounts, percentages, time frames where relevant
    - Reference specific policy sections or clause numbers
    - Explain any exceptions or special circumstances
    - Conclude with practical implications or next steps if applicable

    QUALITY STANDARDS:
    - Demonstrate deep understanding of document content and structure
    - Show logical reasoning from policy text to conclusions
    - Provide information that would satisfy a professional claims reviewer
    - Include sufficient detail for audit trail purposes
    - Maintain consistency with document terminology and formatting

    Return your response as a JSON object with an "answers" array containing exactly {len(request.questions)} comprehensive string elements, each following the analytical framework above.

    IMPORTANT: Each answer should be substantial enough to demonstrate thorough document analysis while remaining focused on the specific question asked. Avoid generic responses - every answer should show specific engagement with the document content.
    """

        # 5. Generate Content
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=GeminiAnswers,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                maxOutputTokens=60000
            )
        )

        parsed_json = json.loads(response.text)
        answers = parsed_json.get('answers', [])
        duration = time.time() - start_time
        print(f"/hackrx/run processed in {duration:.2f} seconds.")
        return QueryResponse(answers=answers)

    except errors.APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e.message}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse JSON response from the model.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
