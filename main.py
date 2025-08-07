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
from pymongo import MongoClient

load_dotenv()

app = FastAPI()
security = HTTPBearer(auto_error=True)

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"
EXPECTED_TOKEN = "bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d"

# --- MongoDB Setup ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['hackrx']
requests_collection = db['requests']

def store_request(request_data):
    """
    Stores a request dictionary in the hackrx.requests collection.
    """
    result = requests_collection.insert_one(request_data)
    return result.inserted_id

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

    # Store incoming request in MongoDB
    try:
        store_request({
            "documents": request.documents,
            "questions": request.questions,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"MongoDB insert error: {e}")

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
    You are a senior insurance claims analyst with 15+ years of experience in policy interpretation, claims assessment, and regulatory compliance. Your expertise covers medical underwriting, coverage determination, and complex policy language interpretation.

    DOCUMENT ANALYSIS METHODOLOGY:
    🔍 DEEP CONTEXTUAL READING
    - Understand document structure, cross-references between sections, and policy hierarchy
    - Identify the PURPOSE and INTENT behind each clause, not just literal text
    - Consider how different policy sections interact and modify each other
    - Look for conditional clauses, exceptions, and qualifying language

    📊 MULTI-LAYERED ANALYSIS FRAMEWORK
    1. EXPLICIT COVERAGE: Direct statements about what is covered/excluded
    2. IMPLICIT CONDITIONS: Underlying requirements and qualifications
    3. LOGICAL RELATIONSHIPS: How clauses connect and affect each other
    4. INDUSTRY CONTEXT: Standard insurance practices and interpretations

    ⚡ CRITICAL ANALYSIS REQUIREMENTS:

    CONTEXT GROUNDING (PRIORITY):
    - NEVER rely on simple keyword matching or surface-level reading
    - Analyze the LOGICAL CHAIN: Policy Section → Conditions → Exceptions → Final Determination
    - Cross-reference multiple sections for comprehensive understanding
    - Consider waiting periods, exclusions, and benefit limitations holistically
    - Reference specific clause numbers, section identifiers, or page locations

    PROFESSIONAL REASONING:
    - Apply insurance industry knowledge to interpret complex policy language
    - Consider standard practice interpretations for ambiguous terms
    - Evaluate claims from both coverage and exclusion perspectives
    - Assess medical necessity and policy compliance requirements

    COMPREHENSIVE COVERAGE ANALYSIS:
    - Address EVERY component of multi-part questions systematically
    - Provide specific details: amounts, percentages, time periods, eligibility criteria
    - Include ALL relevant conditions, exclusions, and special circumstances
    - Explain the reasoning chain from policy text to final conclusion
    - Quote exact policy language and definitions where applicable

    TOKEN EFFICIENCY OPTIMIZATION:
    - Prioritize high-value, decision-critical information
    - Use precise insurance terminology from the document
    - Eliminate redundant explanations while maintaining completeness
    - Structure answers logically: Main Decision → Key Conditions → Supporting Details

    QUESTIONS TO ANALYZE:
    """

        for i, q in enumerate(request.questions):
            prompt += f"{i+1}. {q}\n"

        prompt += f"""

    ANALYTICAL PROCESS FOR EACH QUESTION:
    🎯 STEP 1: LOCATE & EXTRACT
    - Identify ALL relevant policy sections, definitions, and cross-references
    - Extract specific terms, amounts, conditions, and requirements
    - Note any applicable waiting periods, exclusions, or limitations

    🔗 STEP 2: ANALYZE RELATIONSHIPS  
    - Understand how different clauses work together
    - Identify any conflicts or special conditions that apply
    - Consider the policy's hierarchy and precedence rules

    🧠 STEP 3: APPLY LOGIC & CONTEXT
    - Reason through the policy's intent and practical application
    - Consider industry standards and regulatory requirements
    - Evaluate from both coverage and claims processing perspectives

    ✅ STEP 4: FORMULATE COMPLETE ANSWER
    - Lead with direct answer to primary question
    - Follow with specific conditions and requirements
    - Include exact amounts, percentages, time frames
    - Reference policy sections and clause numbers
    - Address exceptions and special circumstances
    - Conclude with practical implications

    ANSWER STRUCTURE REQUIREMENTS:
    📝 FORMAT: [Direct Answer] → [Key Conditions] → [Specific Details] → [Policy References] → [Exceptions/Special Cases]

    QUALITY STANDARDS:
    - Demonstrate deep policy comprehension beyond surface reading
    - Show logical reasoning from policy text to conclusions
    - Provide information sufficient for professional claims review
    - Include audit trail with specific policy references
    - Maintain consistency with document terminology
    - Ensure answers would satisfy regulatory compliance requirements

    EXPLAINABILITY & TRACEABILITY:
    - Reference specific document sections for each claim made
    - Show the logical path from policy language to conclusion
    - Include clause numbers or section identifiers where available
    - Explain any interpretive reasoning or industry standard applications
    - Provide sufficient detail to support claims processing decisions

    Return your response as a JSON object with an "answers" array containing exactly {len(request.questions)} comprehensive string elements. Each answer must demonstrate thorough document analysis, professional reasoning, and complete coverage of the question components.

    IMPORTANT: Treat each question as a professional claims assessment requiring detailed analysis, not simple fact retrieval. Your responses should reflect the depth of understanding expected from a senior claims analyst reviewing complex policy coverage scenarios.
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