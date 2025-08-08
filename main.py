from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types, errors
import asyncio
import aiohttp
import io
import time
import json
import random
import os
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import zipfile
import tempfile
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
security = HTTPBearer(auto_error=True)

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"
EXPECTED_TOKEN = "bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d"

GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),     # Base key first
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3")
]
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None]

if not GEMINI_API_KEYS:
    raise ValueError("No Gemini API keys found. Please set at least GEMINI_API_KEY environment variable.")

print(f"Loaded {len(GEMINI_API_KEYS)} Gemini API keys for rotation")

# Global HTTP session
http_session = None

@app.on_event("startup")
async def startup_event():
    global http_session
    connector = aiohttp.TCPConnector(
        limit=100, limit_per_host=50, keepalive_timeout=60,
        enable_cleanup_closed=True, use_dns_cache=True, ttl_dns_cache=300
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
    http_session = aiohttp.ClientSession(
        connector=connector, timeout=timeout,
        headers={'User-Agent': 'MultiFormat-Processor/1.0'}
    )

@app.on_event("shutdown")
async def shutdown_event():
    global http_session
    if http_session:
        await http_session.close()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class GeminiAnswers(BaseModel):
    answers: List[str] = Field(description="Professional analysis responses")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# --- Document Type Detection and Processing ---
class DocumentProcessor:
    def __init__(self):
        self.clients = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]
        self.client_index = 0
    
    def get_next_client(self):
        client = self.clients[self.client_index]
        api_key_prefix = GEMINI_API_KEYS[self.client_index][:8] + "..."
        print(f"Using API key: {api_key_prefix}")
        self.client_index = (self.client_index + 1) % len(self.clients)
        return client
    
    async def download_document(self, url: str) -> tuple[bytes, str]:
        """Download document and detect its type"""
        global http_session
        
        try:
            async with http_session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Download failed: {response.status}")
                
                content = await response.read()
                content_type = response.headers.get('Content-Type', '').lower()
                
                # Detect document type by content
                doc_type = self.detect_document_type(content, content_type, url)
                
                return content, doc_type
                
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Download timeout")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download error: {str(e)}")
    
    def detect_document_type(self, content: bytes, content_type: str, url: str) -> str:
        """Detect document type from content and headers"""
        
        # Check file signatures
        if content.startswith(b'%PDF'):
            return 'pdf'
        elif content.startswith(b'PK\x03\x04'):  # ZIP-based formats
            if b'word/' in content[:1024] or content_type.find('officedocument') != -1:
                return 'docx'
            elif content_type.find('document') != -1:
                return 'docx'
        elif content.startswith(b'\xd0\xcf\x11\xe0'):  # Old Office format
            return 'doc'
        elif b'From:' in content[:1000] or b'Subject:' in content[:1000]:
            return 'email'
        elif content_type.find('message/rfc822') != -1:
            return 'email'
        
        # Check URL extension
        url_lower = url.lower()
        if url_lower.endswith('.pdf'):
            return 'pdf'
        elif url_lower.endswith('.docx'):
            return 'docx'
        elif url_lower.endswith('.doc'):
            return 'doc'
        elif url_lower.endswith('.eml') or url_lower.endswith('.msg'):
            return 'email'
        
        # Default to PDF if unclear
        return 'pdf'
    
    async def process_document(self, content: bytes, doc_type: str) -> tuple[str, any]:
        """Process document based on type and upload to Gemini"""
        
        client = self.get_next_client()
        uploaded_file = None
        
        try:
            if doc_type == 'pdf':
                file_name = await self._process_pdf(client, content)
                return file_name, client
            elif doc_type in ['docx', 'doc']:
                file_name = await self._process_docx(client, content, doc_type)
                return file_name, client
            elif doc_type == 'email':
                file_name = await self._process_email(client, content)
                return file_name, client
            else:
                # Fallback: try as PDF
                file_name = await self._process_pdf(client, content)
                return file_name, client
                
        except Exception as e:
            print(f"Document processing error ({doc_type}): {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process {doc_type}: {str(e)}")
    
    async def _process_pdf(self, client, content: bytes) -> str:
        """Process PDF document"""
        pdf_io = io.BytesIO(content)
        
        uploaded_file = client.files.upload(
            file=pdf_io,
            config=types.UploadFileConfig(
                mime_type='application/pdf',
                display_name=f"document_{int(time.time())}.pdf"
            )
        )
        
        print(f"PDF uploaded successfully: {uploaded_file.name}")
        return await self._wait_for_processing(client, uploaded_file)
    
    async def _process_docx(self, client, content: bytes, doc_type: str) -> str:
        """Process DOCX/DOC document"""
        docx_io = io.BytesIO(content)
        
        # Determine MIME type
        mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' if doc_type == 'docx' else 'application/msword'
        
        uploaded_file = client.files.upload(
            file=docx_io,
            config=types.UploadFileConfig(
                mime_type=mime_type,
                display_name=f"document_{int(time.time())}.{doc_type}"
            )
        )
        
        print(f"{doc_type.upper()} uploaded successfully: {uploaded_file.name}")
        return await self._wait_for_processing(client, uploaded_file)
    
    async def _process_email(self, client, content: bytes) -> str:
        """Process email document"""
        try:
            # Try to parse as email
            email_content = content.decode('utf-8', errors='ignore')
            
            # Convert to text format for Gemini
            email_io = io.BytesIO(email_content.encode('utf-8'))
            
            uploaded_file = client.files.upload(
                file=email_io,
                config=types.UploadFileConfig(
                    mime_type='text/plain',
                    display_name=f"email_{int(time.time())}.txt"
                )
            )
            
            print(f"Email uploaded successfully: {uploaded_file.name}")
            return await self._wait_for_processing(client, uploaded_file)
            
        except Exception as e:
            print(f"Email processing error: {str(e)}")
            # Fallback: treat as text
            text_io = io.BytesIO(content)
            uploaded_file = client.files.upload(
                file=text_io,
                config=types.UploadFileConfig(
                    mime_type='text/plain',
                    display_name=f"document_{int(time.time())}.txt"
                )
            )
            return await self._wait_for_processing(client, uploaded_file)
    
    async def _wait_for_processing(self, client, uploaded_file) -> str:
        """Wait for document processing and return file reference"""
        max_wait = 45
        start_wait = time.time()
        
        while True:
            # Use the SAME client for status check
            file_status = client.files.get(name=uploaded_file.name)
            
            if file_status.state.name == 'ACTIVE':
                print(f"Document processed successfully: {uploaded_file.name}")
                return uploaded_file.name
            elif file_status.state.name == 'FAILED':
                error_msg = file_status.error.message if file_status.error else 'Processing failed'
                raise HTTPException(status_code=500, detail=f"Document processing failed: {error_msg}")
            elif time.time() - start_wait > max_wait:
                raise HTTPException(status_code=504, detail="Document processing timeout")
                
            await asyncio.sleep(1.5)

# Global processor instance
processor = DocumentProcessor()

def build_multi_format_prompt(questions: List[str], doc_type: str) -> str:
    """Build prompt optimized for different document types"""
    
    doc_specific_instructions = {
        'pdf': "This is a PDF document. Focus on policy clauses, benefit schedules, and formal contract language.",
        'docx': "This is a Word document. May contain informal policy details, amendments, or supplementary information.",
        'email': "This is an email document. Look for policy communications, clarifications, amendments, or coverage decisions.",
        'doc': "This is a legacy Word document. May contain historical policy information or formal documentation."
    }
    
    prompt = f"""You are a senior insurance analyst processing a {doc_type.upper()} document. {doc_specific_instructions.get(doc_type, '')}

MULTI-FORMAT DOCUMENT ANALYSIS FRAMEWORK:

🔍 DOCUMENT-SPECIFIC PROCESSING:
- Adapt analysis approach based on document format and structure
- Extract information from tables, schedules, email threads, or policy sections as appropriate
- Handle format-specific elements (headers, footers, email signatures, attachments references)

📊 COMPREHENSIVE CLAUSE RETRIEVAL & MATCHING:
- Locate ALL relevant policy clauses across document sections
- Match specific terms and conditions to query requirements
- Cross-reference between different document sections/emails
- Extract exact benefit amounts, percentages, and coverage limits

⚖️ EXPLAINABLE DECISION RATIONALE:
- Provide clear reasoning chain from document text to conclusion
- Reference specific clauses, sections, email content, or table entries
- Show how different policy components interact to determine coverage
- Explain any conflicts and how they are resolved using policy hierarchy

🎯 STRUCTURED PROFESSIONAL ANALYSIS:
For each question, provide:

📋 COVERAGE DETERMINATION: [Clear coverage decision with specifics]
💰 BENEFIT DETAILS: [Exact amounts, percentages, limits from document]
📝 ELIGIBILITY CONDITIONS: [All requirements that must be met]
🚫 EXCLUSIONS/LIMITATIONS: [Specific restrictions or exceptions]
🔍 DOCUMENT REFERENCES: [Exact locations - page numbers, section names, email dates]
⚖️ DECISION RATIONALE: [Step-by-step reasoning showing how conclusion was reached]

QUESTIONS FOR ANALYSIS:
"""
    
    for i, q in enumerate(questions, 1):
        prompt += f"""
{i}. {q}

ANALYSIS REQUIREMENTS:
- Perform comprehensive clause retrieval for all relevant sections
- Provide explainable rationale linking document evidence to conclusions
- Include specific document references (pages/sections/email content)
- Address all components of multi-part questions
- Show professional reasoning chain

"""

    prompt += f"""
CRITICAL SUCCESS FACTORS:
✓ Complete clause retrieval from document (don't miss hidden relevant sections)
✓ Precise benefit matching with exact figures from document
✓ Clear explainable rationale for each decision
✓ Professional insurance expertise demonstrated
✓ Structured response format for maximum scoring recognition
✓ Handle document format appropriately (PDF formal vs email informal)

Return JSON with "answers" array containing {len(questions)} comprehensive, professionally-structured responses with complete explainable decision rationale and precise clause matching."""

    return prompt

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_multi_format_queries(
    request: QueryRequest, 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    start_time = time.time()
    file_reference = None
    client = None
    
    try:
        # Download and detect document type
        print("Downloading and detecting document type...")
        content, doc_type = await processor.download_document(request.documents)
        print(f"Detected document type: {doc_type}")
        
        # Process document based on type - get both file reference and client
        print(f"Processing {doc_type} document...")
        file_reference, client = await processor.process_document(content, doc_type)
        print(f"Document processed successfully: {file_reference}")
        
        # Build format-specific prompt
        prompt = build_multi_format_prompt(request.questions, doc_type)
        
        # Generate response using the SAME client that uploaded the file
        print(f"Generating content with file: {file_reference}")
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[client.files.get(name=file_reference), prompt],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=GeminiAnswers,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    maxOutputTokens=50000,
                    temperature=0.1,
                    candidateCount=1,
                    top_p=0.9,
                    top_k=50
                )
            )
        except Exception as content_error:
            print(f"Content generation error with same client: {content_error}")
            # If there's still an issue, try with a fresh client
            fresh_client = processor.get_next_client()
            response = fresh_client.models.generate_content(
                model=MODEL_NAME,
                contents=[fresh_client.files.get(name=file_reference), prompt],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=GeminiAnswers,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    maxOutputTokens=50000,
                    temperature=0.1,
                    candidateCount=1,
                    top_p=0.9,
                    top_k=50
                )
            )
        
        parsed_json = json.loads(response.text)
        answers = parsed_json.get('answers', [])
        
        # Ensure correct count
        while len(answers) < len(request.questions):
            answers.append("Unable to process this question completely.")
        
        duration = time.time() - start_time
        print(f"Multi-format request processed in {duration:.2f}s - Type: {doc_type}")
        
        return QueryResponse(answers=answers[:len(request.questions)])
        
    except Exception as e:
        print(f"Error processing multi-format document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        
    finally:
        # Cleanup using the same client that uploaded the file
        if file_reference and client:
            try:
                client.files.delete(name=file_reference)
                print(f"Cleaned up file: {file_reference}")
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")
                pass

@app.get("/health")
async def health_check():
    return {
        "status": "multi_format_ready",
        "supported_formats": ["PDF", "DOCX", "DOC", "EMAIL"],
        "clients": len(processor.clients),
        "features": ["clause_retrieval", "explainable_decisions", "multi_format"]
    }

@app.get("/test-keys")
async def test_api_keys():
    """Test each Gemini API key individually"""
    results = []
    
    for i, key in enumerate(GEMINI_API_KEYS):
        try:
            client = genai.Client(api_key=key)
            # Test with a simple model list call
            models = client.models.list()
            results.append({
                "key_index": i,
                "key_prefix": key[:8] + "..." if key else "None",
                "status": "working",
                "models_count": len(list(models))
            })
        except Exception as e:
            results.append({
                "key_index": i,
                "key_prefix": key[:8] + "..." if key else "None",
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total_keys": len(GEMINI_API_KEYS),
        "key_tests": results
    }