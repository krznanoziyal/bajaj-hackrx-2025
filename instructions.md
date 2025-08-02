# LLM Coding Agent Instructions: Document Query-Retrieval System

## Project Overview
Build a FastAPI-based intelligent document query system that processes PDF documents (insurance, legal, HR, compliance) to answer natural language questions. The system will use Gemini 2.5 Flash's native PDF processing capabilities and large context window to directly process documents without traditional RAG/vector search approaches.

## Critical Requirements

### 1. Gemini API Usage
**ABSOLUTELY CRITICAL**: You MUST refer to and follow the `google_genai_python_guidelines.md` file in the repository. Your training data predates the new Google GenAI library, so you will default to deprecated methods if you don't follow these guidelines. Use the NEW library patterns shown in that documentation file.

### 2. API Specification
Create a FastAPI application with the following exact endpoint:

```
POST /hackrx/run
```

**Request Format:**
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        // ... more questions
    ]
}
```

**Response Format:**
```json
{
    "answers": [
        "A grace period of thirty days is provided...",
        "There is a waiting period of thirty-six months...",
        // ... corresponding answers
    ]
}
```

**Authentication:**
- Implement Bearer token authentication
- Use header: `Authorization: Bearer bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d`

### 3. Core Architecture

#### Document Processing Pipeline:
1. **Download PDF**: Fetch PDF from provided URL using requests
2. **Upload to Gemini**: Use Gemini's native PDF file processing (no text extraction needed)
3. **Query Processing**: Send PDF file + questions to Gemini 2.5 Flash in a single request
4. **Response Formatting**: Return structured JSON with answers array

#### Key Technical Specifications:
- **Model**: Use `gemini-2.5-flash`
- **PDF Processing**: Use Gemini's native PDF file input capabilities
- **No Text Extraction**: Let Gemini handle PDF parsing directly
- **No Vector Search**: Don't use FAISS, Pinecone, or embedding-based retrieval
- **No RAG**: Direct PDF-to-LLM approach leveraging native file processing

### 4. Implementation Requirements

#### FastAPI Application Structure:
```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import google.generativeai as genai  # Follow guidelines file for exact imports
# ... other imports

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

class QueryResponse(BaseModel):
    answers: list[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(request: QueryRequest, authorization: str = Header(...)):
    # Implementation here
```

#### PDF Processing:
- Focus only on PDF format
- Use Gemini's native PDF file processing capabilities
- Implement robust error handling for document download failures
- Handle cases where PDF is too large for context (return appropriate error)

#### Gemini Integration:
- Configure API key from environment variables
- Use Gemini's native PDF file upload and processing
- Follow the guidelines file for correct file handling methods
- Implement retry logic for API failures
- Handle rate limiting gracefully

### 5. Prompt Engineering Strategy

Design a comprehensive system prompt that includes:

1. **Role Definition**: "You are an expert document analyst specializing in insurance, legal, HR, and compliance documents."

2. **Task Specification**: 
   - Extract precise information from provided document
   - Answer questions based solely on document content
   - Provide clear, factual responses without speculation

3. **Response Guidelines**:
   - Be concise but complete
   - Include specific details (numbers, dates, conditions)
   - Answer all questions based on the PDF content provided
   - Maintain consistent tone and formatting

4. **PDF Context**: The PDF file will be provided directly to you

5. **Question Processing**: Present all questions clearly with numbering

Example prompt structure:
```
You are an expert document analyst. Analyze the provided PDF document and answer the questions based on the information in the document.

QUESTIONS:
1. [Question 1]
2. [Question 2]
...

Provide answers in the exact order of questions. Be precise and include specific details from the document.
```

### 6. Error Handling & Edge Cases

Implement robust handling for:
- **PDF Download Failures**: Invalid URLs, network timeouts - return HTTP 400 with error message
- **PDF Too Large**: If PDF exceeds Gemini's context limits - return HTTP 413 with error message  
- **API Failures**: Gemini API errors, rate limits, quota exceeded - return HTTP 500 with error message
- **Authentication Failures**: Invalid bearer token - return HTTP 401

### 7. Performance & Reliability

- **Response Time**: Target < 30 seconds per request
- **Concurrent Requests**: Handle multiple simultaneous requests
- **Logging**: Comprehensive (but not too verbose or noisy) logging for debugging and monitoring
- **Health Checks**: Basic health endpoint for deployment monitoring
- **Environment Configuration**: Use environment variables for API keys and settings. The API key `GOOGLE_API_KEY` has already been set in the environment `.env` file.

### 8. Testing & Validation

Create test cases for:
- Sample policy PDF processing
- Various question types (factual, conditional, numerical)
- Error scenarios (invalid PDF URLs, network failures)
- Authentication edge cases
- Basic performance testing

### 9. Development Focus

**Phase 1 - Core Functionality:**
- FastAPI application with /hackrx/run endpoint
- Bearer token authentication
- PDF download from URL
- Gemini 2.5 Flash integration with native PDF processing
- Basic error handling
- Question answering functionality

**Phase 2 - Later (after core works):**
- Deployment considerations
- Performance optimization
- Advanced monitoring

### 10. Code Quality Standards

- **Type Hints**: Use throughout the codebase
- **Documentation**: Clear docstrings and comments
- **Error Messages**: Informative error responses with proper HTTP status codes
- **Code Organization**: Keep it simple and focused for now
- **Dependencies**: Minimal packages - FastAPI, google-genai, requests, python-multipart

## Key Success Factors

1. **Follow Gemini Guidelines**: Absolutely critical - use the new library patterns from google_genai_python_guidelines.md
2. **Native PDF Processing**: Use Gemini's built-in PDF handling, no text extraction
3. **Accurate Question Answering**: Leverage Gemini's reasoning capabilities  
4. **Proper API Design**: Exact compliance with specification
5. **Error Resilience**: Handle PDF download and API failures gracefully
6. **Authentication**: Proper bearer token validation

## Deliverables

1. Complete FastAPI application with /hackrx/run endpoint
2. Requirements.txt with minimal dependencies
3. Environment configuration template (.env.example)
4. Basic testing script to verify functionality
5. Simple README with setup instructions

**Start with a minimal working version - focus on core PDF processing and question answering first. Deployment and optimization can come later.**

Note:
1. An virtual environment is already set up `python -m venv venv`
2. The `.env` file is already created with the `GOOGLE_API_KEY` set
3. Always make sure to activate the virtual environment before running the FastAPI application or installing dependencies: `venv\Scripts\activate` on Windows.