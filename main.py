from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai import errors
import requests
from dotenv import load_dotenv
import io
import time
import json

load_dotenv()

app = FastAPI()

# --- Configuration ---
EXPECTED_BEARER_TOKEN = "bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d"
MODEL_NAME = "gemini-2.5-flash"

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class GeminiAnswers(BaseModel):
    answers: List[str] = Field(description="A list of answers to the questions, in the same order.")

# --- Security (Optional for Hackathon) ---
async def verify_token(authorization: Optional[str] = Header(default=None)):
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        token = authorization.split(" ")[1]
        if token != EXPECTED_BEARER_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(request: QueryRequest, _=Depends(verify_token)):
    start_time = time.time()
    try:
        # 1. Download PDF
        pdf_response = requests.get(request.documents)
        pdf_response.raise_for_status()
        if 'application/pdf' not in pdf_response.headers.get('Content-Type', ''):
            raise HTTPException(status_code=400, detail="The provided URL does not point to a PDF document.")
        pdf_data = io.BytesIO(pdf_response.content)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    uploaded_file = None
    try:
        client = genai.Client()

        # 2. Upload PDF to Gemini
        print("Uploading file to Gemini...")
        uploaded_file = client.files.upload(
            file=pdf_data,
            config=types.UploadFileConfig(
                mime_type='application/pdf',
                display_name=request.documents.split('/')[-1]
            )
        )

        print(f"File uploaded: {uploaded_file.name}. Waiting for processing...")
        get_file = client.files.get(name=uploaded_file.name)
        while get_file.state.name == 'PROCESSING':
            time.sleep(5)
            get_file = client.files.get(name=uploaded_file.name)
            print(f"File status: {get_file.state.name}")

        if get_file.state.name == 'FAILED':
            error_message = get_file.error.message if get_file.error else 'Unknown'
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {error_message}")

        print("File processed successfully.")

        # 3. Build Prompt
        prompt = """
        You are an expert document analyst specializing in insurance, legal, HR, and compliance documents.

        TASK: Analyze the provided PDF document and answer each question based EXCLUSIVELY on the information contained in the document.

        INSTRUCTIONS:
        - Read through the entire document carefully before answering
        - Answer ALL questions in the exact order they are presented
        - Provide complete, detailed answers that directly address what each question is asking
        - Include specific details such as numbers, percentages, time periods, conditions, and eligibility criteria
        - Quote exact terms and definitions when relevant
        - If a question asks about multiple aspects (e.g., "Does X cover Y, and what are the conditions?"), address ALL parts
        - Be precise with terminology used in the document
        - Do not skip any questions - provide an answer for each one

        ANSWER FORMAT:
        - Provide exactly one answer per question
        - Answers should be comprehensive but concise
        - Include relevant context and conditions
        - Use the same terminology as found in the document

        QUESTIONS TO ANSWER:
        """

        for i, q in enumerate(request.questions):
            prompt += f"{i+1}. {q}\n"

        prompt += f"""
        CRITICAL: You must provide exactly {len(request.questions)} answers in your JSON response, one for each question above, in the same order. Each answer should directly and completely address the specific question asked.
        Return your response as a JSON object with an "answers" array containing exactly {len(request.questions)} string elements.
        """

        # 4. Generate Content
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

        # 5. Parse JSON response and return
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
        # 6. Clean up the uploaded file
        if uploaded_file:
            print(f"Deleting uploaded file: {uploaded_file.name}")
            client.files.delete(name=uploaded_file.name)
