# Google GenAI Python SDK Guidelines

## Library Names
This library is sometimes called:
- Google Gemini API
- Google GenAI API
- Google GenAI SDK
- Gemini API
- google-genai (Python package)

The Google GenAI Python SDK can be used to call Gemini models.

## Deprecated API - Do Not Use
Do not use or import below types from `google-generativeai`, these are old deprecated API and no longer work:
- Incorrect: `google.generativeai`
- Incorrect: `genai.GenerativeModel`
- Incorrect: `genai.configure`
- Incorrect: `genai.list_models`
- Incorrect: `model.generate_content`
- Incorrect: `model.start_chat`
- Incorrect: `genai.embed_content`

The old `google-generativeai` package is deprecated and will stop receiving updates on September 30, 2025.

When using generate content for text answers, do not define the model first and call generate content later, you must use `client.models.generate_content` to query GenAI with both model name and prompt.

## Installation
Always use: `pip install google-genai`
Incorrect: `pip install google-generativeai` (deprecated package)

## Client Initialization
Always create a client using one of these methods:

For Gemini Developer API:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
```

For Vertex AI API:
```python
from google import genai

client = genai.Client(
    vertexai=True, 
    project='your-project-id', 
    location='us-central1'
)
```

## Environment Variables
The API key/credentials must be obtained from environment variables:

For Gemini Developer API:
```bash
export GOOGLE_API_KEY='your-api-key'
```

For Vertex AI:
```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

Then create client:
```python
client = genai.Client()  # Automatically uses environment variables
```

Strict Prohibition: Generate any UI elements (input fields, forms, prompts, configuration sections) or code snippets for entering or managing the API key. Do not define environment variables or request from the user to update the API keys in the code. The key's availability is handled externally and is a hard requirement. The application must not ask the user for it under any circumstances.

## Models
Only use below models when using google-genai:
- General Text Tasks: `'gemini-2.5-flash'` or `'gemini-2.5-pro'`
- Image Generation Tasks: `'imagen-3.0-generate-002'`
- Video Generation Tasks: `'veo-2.0-generate-001'`
- Embedding Tasks: `'text-embedding-004'`

Do not use below deprecated models:
- Prohibited: `gemini-1.5-flash`
- Prohibited: `gemini-1.5-pro`
- Prohibited: `gemini-pro`

## Import
Always use:
```python
from google import genai
from google.genai import types
```

Prohibited:
- `import google.generativeai as genai`
- `from google.generativeai import GenerativeModel`
- `import genai`

## Generate Content
Generate response from the model:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Why is the sky blue?'
)
print(response.text)
```

Generate content with multiple parts (text and image):
```python
from google import genai
from google.genai import types

client = genai.Client(api_key='your-api-key')

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://your-bucket/image.jpg',
            mime_type='image/jpeg'
        )
    ]
)
print(response.text)
```

## Extracting Text Output from GenerateContentResponse
When you use `client.models.generate_content`, it returns a `GenerateContentResponse` object.
The simplest and most direct way to get the generated text content is by accessing the `.text` property on this object.

### Correct Method
The `GenerateContentResponse` object has a property called `text` that directly provides the string output:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Why is the sky blue?'
)
text = response.text
print(text)
```

### Incorrect Methods to avoid
- Incorrect: `text = response?.response?.text?`
- Incorrect: `text = response?.candidates[0]?.content?.parts[0]?.text`
- Incorrect: `text = response?.result?.text`
- Incorrect: `json = response.candidates?.[0]?.content?.parts?.[0]?.json`

## System Instruction and Other Model Configs
Generate response with system instruction and other model configs:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Tell me a story in 100 words.',
    config=types.GenerateContentConfig(
        system_instruction='You are a storyteller for kids under 5 years old',
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        max_output_tokens=150,
        seed=42
    )
)
print(response.text)
```

## JSON Response
Ask the model to return a response in JSON format:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Tell me about Python programming.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json'
    )
)

# Parse the JSON response
import json
try:
    parsed_data = json.loads(response.text)
    print(parsed_data)
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON response: {e}")
```

## Generate Content (Streaming)
Generate response from the model in streaming mode:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
for chunk in client.models.generate_content_stream(
    model='gemini-2.5-flash',
    contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

## Generate Image
Generate images from the model:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key='your-api-key')
response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='Robot holding a red skateboard',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        output_mime_type='image/jpeg'
    )
)

generated_image = response.generated_images[0].image
generated_image.show()  # Display the image
```

## Chat
Start a chat and send messages to the model:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
chat = client.chats.create(model='gemini-2.5-flash')

response = chat.send_message('Tell me a story in 100 words')
print(response.text)

response = chat.send_message('What happened after that?')
print(response.text)
```

## Chat (Streaming)
Start a chat and send messages with streaming response:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
chat = client.chats.create(model='gemini-2.5-flash')

for chunk in chat.send_message_stream('Tell me a story in 100 words'):
    print(chunk.text, end='')

for chunk in chat.send_message_stream('What happened after that?'):
    print(chunk.text, end='')
```

## Function Calling
You can pass a Python function directly and it will be automatically called:
```python
from google import genai
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather]
    )
)
print(response.text)
```

Manual function declaration:
```python
from google import genai
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA'
            )
        },
        required=['location']
    )
)

tool = types.Tool(function_declarations=[function])
client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool]
    )
)
print(response.function_calls[0])
```

## Async Support
All client methods have async equivalents using `client.aio`:
```python
from google import genai

client = genai.Client(api_key='your-api-key')

# Async generate content
response = await client.aio.models.generate_content(
    model='gemini-2.5-flash',
    contents='Tell me a story in 300 words.'
)
print(response.text)

# Async streaming
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.5-flash',
    contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

## Count Tokens
Count tokens in content:
```python
from google import genai

client = genai.Client(api_key='your-api-key')
response = client.models.count_tokens(
    model='gemini-2.5-flash',
    contents='Why is the sky blue?'
)
print(response)
```

## Embed Content
Generate embeddings:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key='your-api-key')
response = client.models.embed_content(
    model='text-embedding-004',
    contents='Why is the sky blue?'
)
print(response)

# Multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['Why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10)
)
print(response)
```

## Safety Settings
Configure safety settings:
```python
from google import genai
from google.genai import types

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH'
            )
        ]
    )
)
print(response.text)
```

## Error Handling
Implement robust handling for API errors:
```python
from google import genai
from google.genai import errors

client = genai.Client(api_key='your-api-key')

try:
    response = client.models.generate_content(
        model="invalid-model-name",
        contents="What is your name?"
    )
except errors.APIError as e:
    print(f"API Error {e.code}: {e.message}")
    # Implement retry logic here
except Exception as e:
    print(f"Unexpected error: {e}")
```

Use graceful retry logic (like exponential backoff) to avoid overwhelming the backend.

## JSON Response Schema
Define structured response schemas using Pydantic models:
```python
from google import genai
from google.genai import types
from pydantic import BaseModel

class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int

client = genai.Client(api_key='your-api-key')
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo
    )
)
print(response.text)
```

## Document Understanding
Gemini models can process documents in PDF format, using native
vision to understand entire document contexts. This goes beyond
simple text extraction, allowing Gemini to:

- Analyze and interpret content, including text, images, diagrams, charts, and tables, even in long documents up to 1000 pages.
- Extract information into [structured output](/gemini-api/docs/structured-output) formats.
- Summarize and answer questions based on both the visual and textual elements in a document.
- Transcribe document content (e.g. to HTML), preserving layouts and formatting, for use in downstream applications.

Passing inline PDF data

You can pass inline PDF data in the request to `generateContent`.
For PDF payloads under 20MB, you can choose between uploading base64
encoded documents or directly uploading locally stored files.

The following example shows you how to fetch a PDF from a URL and convert it to
bytes for processing:  

Python  

    from google import genai
    from google.genai import types
    import httpx

    client = genai.Client()

    doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"

    # Retrieve and encode the PDF byte
    doc_data = httpx.get(doc_url).content

    prompt = "Summarize this document"
    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=[
          types.Part.from_bytes(
            data=doc_data,
            mime_type='application/pdf',
          ),
          prompt])
    print(response.text)

JavaScript  

    import { GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({ apiKey: "GEMINI_API_KEY" });

    async function main() {
        const pdfResp = await fetch('https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf')
            .then((response) => response.arrayBuffer());

        const contents = [
            { text: "Summarize this document" },
            {
                inlineData: {
                    mimeType: 'application/pdf',
                    data: Buffer.from(pdfResp).toString("base64")
                }
            }
        ];

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: contents
        });
        console.log(response.text);
    }

    main();

Go  

    package main

    import (
        "context"
        "fmt"
        "io"
        "net/http"
        "os"
        "google.golang.org/genai"
    )

    func main() {

        ctx := context.Background()
        client, _ := genai.NewClient(ctx, &genai.ClientConfig{
            APIKey:  os.Getenv("GEMINI_API_KEY"),
            Backend: genai.BackendGeminiAPI,
        })

        pdfResp, _ := http.Get("https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf")
        var pdfBytes []byte
        if pdfResp != nil && pdfResp.Body != nil {
            pdfBytes, _ = io.ReadAll(pdfResp.Body)
            pdfResp.Body.Close()
        }

        parts := []*genai.Part{
            &genai.Part{
                InlineData: &genai.Blob{
                    MIMEType: "application/pdf",
                    Data:     pdfBytes,
                },
            },
            genai.NewPartFromText("Summarize this document"),
        }

        contents := []*genai.Content{
            genai.NewContentFromParts(parts, genai.RoleUser),
        }

        result, _ := client.Models.GenerateContent(
            ctx,
            "gemini-2.5-flash",
            contents,
            nil,
        )

        fmt.Println(result.Text())
    }

REST  

    DOC_URL="https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"
    PROMPT="Summarize this document"
    DISPLAY_NAME="base64_pdf"

    # Download the PDF
    wget -O "${DISPLAY_NAME}.pdf" "${DOC_URL}"

    # Check for FreeBSD base64 and set flags accordingly
    if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
      B64FLAGS="--input"
    else
      B64FLAGS="-w0"
    fi

    # Base64 encode the PDF
    ENCODED_PDF=$(base64 $B64FLAGS "${DISPLAY_NAME}.pdf")

    # Generate content using the base64 encoded PDF
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"inline_data": {"mime_type": "application/pdf", "data": "'"$ENCODED_PDF"'"}},
              {"text": "'$PROMPT'"}
            ]
          }]
        }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

    # Clean up the downloaded PDF
    rm "${DISPLAY_NAME}.pdf"

You can also read a PDF from a local file for processing:  

Python  

    from google import genai
    from google.genai import types
    import pathlib

    client = genai.Client()

    # Retrieve and encode the PDF byte
    filepath = pathlib.Path('file.pdf')

    prompt = "Summarize this document"
    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=[
          types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
          ),
          prompt])
    print(response.text)

JavaScript  

    import { GoogleGenAI } from "@google/genai";
    import * as fs from 'fs';

    const ai = new GoogleGenAI({ apiKey: "GEMINI_API_KEY" });

    async function main() {
        const contents = [
            { text: "Summarize this document" },
            {
                inlineData: {
                    mimeType: 'application/pdf',
                    data: Buffer.from(fs.readFileSync("content/343019_3_art_0_py4t4l_convrt.pdf")).toString("base64")
                }
            }
        ];

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: contents
        });
        console.log(response.text);
    }

    main();

Go  

    package main

    import (
        "context"
        "fmt"
        "os"
        "google.golang.org/genai"
    )

    func main() {

        ctx := context.Background()
        client, _ := genai.NewClient(ctx, &genai.ClientConfig{
            APIKey:  os.Getenv("GEMINI_API_KEY"),
            Backend: genai.BackendGeminiAPI,
        })

        pdfBytes, _ := os.ReadFile("path/to/your/file.pdf")

        parts := []*genai.Part{
            &genai.Part{
                InlineData: &genai.Blob{
                    MIMEType: "application/pdf",
                    Data:     pdfBytes,
                },
            },
            genai.NewPartFromText("Summarize this document"),
        }
        contents := []*genai.Content{
            genai.NewContentFromParts(parts, genai.RoleUser),
        }

        result, _ := client.Models.GenerateContent(
            ctx,
            "gemini-2.5-flash",
            contents,
            nil,
        )

        fmt.Println(result.Text())
    }

Uploading PDFs using the File API

You can use the [File API](/gemini-api/docs/files) to upload larger documents. Always use the File
API when the total request size (including the files, text prompt, system
instructions, etc.) is larger than 20MB.
| **Note:** The [File API](/gemini-api/docs/files) lets you store up to 50MB of PDF files. Files are stored for 48 hours. You can access them in that period with your API key, but you can't download them from the API. The File API is available at no cost in all regions where the Gemini API is available.

Call [`media.upload`](/api/rest/v1beta/media/upload) to upload a file using the
File API. The following code uploads a document file and then uses the file in a
call to
[`models.generateContent`](/api/generate-content#method:-models.generatecontent).

Large PDFs from URLs

Use the File API to simplify uploading and processing large PDF files from URLs:  

Python  

    from google import genai
    from google.genai import types
    import io
    import httpx

    client = genai.Client()

    long_context_pdf_path = "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a17/A17_FlightPlan.pdf"

    # Retrieve and upload the PDF using the File API
    doc_io = io.BytesIO(httpx.get(long_context_pdf_path).content)

    sample_doc = client.files.upload(
      # You can pass a path or a file-like object here
      file=doc_io,
      config=dict(
        mime_type='application/pdf')
    )

    prompt = "Summarize this document"

    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=[sample_doc, prompt])
    print(response.text)

JavaScript  

    import { createPartFromUri, GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({ apiKey: "GEMINI_API_KEY" });

    async function main() {

        const pdfBuffer = await fetch("https://www.nasa.gov/wp-content/uploads/static/history/alsj/a17/A17_FlightPlan.pdf")
            .then((response) => response.arrayBuffer());

        const fileBlob = new Blob([pdfBuffer], { type: 'application/pdf' });

        const file = await ai.files.upload({
            file: fileBlob,
            config: {
                displayName: 'A17_FlightPlan.pdf',
            },
        });

        // Wait for the file to be processed.
        let getFile = await ai.files.get({ name: file.name });
        while (getFile.state === 'PROCESSING') {
            getFile = await ai.files.get({ name: file.name });
            console.log(`current file status: ${getFile.state}`);
            console.log('File is still processing, retrying in 5 seconds');

            await new Promise((resolve) => {
                setTimeout(resolve, 5000);
            });
        }
        if (file.state === 'FAILED') {
            throw new Error('File processing failed.');
        }

        // Add the file to the contents.
        const content = [
            'Summarize this document',
        ];

        if (file.uri && file.mimeType) {
            const fileContent = createPartFromUri(file.uri, file.mimeType);
            content.push(fileContent);
        }

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: content,
        });

        console.log(response.text);

    }

    main();

Go  

    package main

    import (
      "context"
      "fmt"
      "io"
      "net/http"
      "os"
      "google.golang.org/genai"
    )

    func main() {

      ctx := context.Background()
      client, _ := genai.NewClient(ctx, &genai.ClientConfig{
        APIKey:  os.Getenv("GEMINI_API_KEY"),
        Backend: genai.BackendGeminiAPI,
      })

      pdfURL := "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a17/A17_FlightPlan.pdf"
      localPdfPath := "A17_FlightPlan_downloaded.pdf"

      respHttp, _ := http.Get(pdfURL)
      defer respHttp.Body.Close()

      outFile, _ := os.Create(localPdfPath)
      defer outFile.Close()

      _, _ = io.Copy(outFile, respHttp.Body)

      uploadConfig := &genai.UploadFileConfig{MIMEType: "application/pdf"}
      uploadedFile, _ := client.Files.UploadFromPath(ctx, localPdfPath, uploadConfig)

      promptParts := []*genai.Part{
        genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
        genai.NewPartFromText("Summarize this document"),
      }
      contents := []*genai.Content{
        genai.NewContentFromParts(promptParts, genai.RoleUser), // Specify role
      }

        result, _ := client.Models.GenerateContent(
            ctx,
            "gemini-2.5-flash",
            contents,
            nil,
        )

      fmt.Println(result.Text())
    }

REST  

    PDF_PATH="https://www.nasa.gov/wp-content/uploads/static/history/alsj/a17/A17_FlightPlan.pdf"
    DISPLAY_NAME="A17_FlightPlan"
    PROMPT="Summarize this document"

    # Download the PDF from the provided URL
    wget -O "${DISPLAY_NAME}.pdf" "${PDF_PATH}"

    MIME_TYPE=$(file -b --mime-type "${DISPLAY_NAME}.pdf")
    NUM_BYTES=$(wc -c < "${DISPLAY_NAME}.pdf")

    echo "MIME_TYPE: ${MIME_TYPE}"
    echo "NUM_BYTES: ${NUM_BYTES}"

    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
      -D upload-header.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
      -H "Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${DISPLAY_NAME}.pdf" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)
    echo "file_uri: ${file_uri}"

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "'$PROMPT'"},
              {"file_data":{"mime_type": "application/pdf", "file_uri": '$file_uri'}}]
            }]
          }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

    # Clean up the downloaded PDF
    rm "${DISPLAY_NAME}.pdf"

Large PDFs stored locally  

Python  

    from google import genai
    from google.genai import types
    import pathlib
    import httpx

    client = genai.Client()

    # Retrieve and encode the PDF byte
    file_path = pathlib.Path('large_file.pdf')

    # Upload the PDF using the File API
    sample_file = client.files.upload(
      file=file_path,
    )

    prompt="Summarize this document"

    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=[sample_file, "Summarize this document"])
    print(response.text)

JavaScript  

    import { createPartFromUri, GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({ apiKey: "GEMINI_API_KEY" });

    async function main() {
        const file = await ai.files.upload({
            file: 'path-to-localfile.pdf'
            config: {
                displayName: 'A17_FlightPlan.pdf',
            },
        });

        // Wait for the file to be processed.
        let getFile = await ai.files.get({ name: file.name });
        while (getFile.state === 'PROCESSING') {
            getFile = await ai.files.get({ name: file.name });
            console.log(`current file status: ${getFile.state}`);
            console.log('File is still processing, retrying in 5 seconds');

            await new Promise((resolve) => {
                setTimeout(resolve, 5000);
            });
        }
        if (file.state === 'FAILED') {
            throw new Error('File processing failed.');
        }

        // Add the file to the contents.
        const content = [
            'Summarize this document',
        ];

        if (file.uri && file.mimeType) {
            const fileContent = createPartFromUri(file.uri, file.mimeType);
            content.push(fileContent);
        }

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: content,
        });

        console.log(response.text);

    }

    main();

Go  

    package main

    import (
        "context"
        "fmt"
        "os"
        "google.golang.org/genai"
    )

    func main() {

        ctx := context.Background()
        client, _ := genai.NewClient(ctx, &genai.ClientConfig{
            APIKey:  os.Getenv("GEMINI_API_KEY"),
            Backend: genai.BackendGeminiAPI,
        })
        localPdfPath := "/path/to/file.pdf"

        uploadConfig := &genai.UploadFileConfig{MIMEType: "application/pdf"}
        uploadedFile, _ := client.Files.UploadFromPath(ctx, localPdfPath, uploadConfig)

        promptParts := []*genai.Part{
            genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
            genai.NewPartFromText("Give me a summary of this pdf file."),
        }
        contents := []*genai.Content{
            genai.NewContentFromParts(promptParts, genai.RoleUser),
        }

        result, _ := client.Models.GenerateContent(
            ctx,
            "gemini-2.5-flash",
            contents,
            nil,
        )

        fmt.Println(result.Text())
    }

REST  

    NUM_BYTES=$(wc -c < "${PDF_PATH}")
    DISPLAY_NAME=TEXT
    tmp_header_file=upload-header.tmp

    # Initial resumable request defining metadata.
    # The upload url is in the response headers dump them to a file.
    curl "${BASE_URL}/upload/v1beta/files?key=${GEMINI_API_KEY}" \
      -D upload-header.tmp \
      -H "X-Goog-Upload-Protocol: resumable" \
      -H "X-Goog-Upload-Command: start" \
      -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Header-Content-Type: application/pdf" \
      -H "Content-Type: application/json" \
      -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

    upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
    rm "${tmp_header_file}"

    # Upload the actual bytes.
    curl "${upload_url}" \
      -H "Content-Length: ${NUM_BYTES}" \
      -H "X-Goog-Upload-Offset: 0" \
      -H "X-Goog-Upload-Command: upload, finalize" \
      --data-binary "@${PDF_PATH}" 2> /dev/null > file_info.json

    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_uri

    # Now generate content using that file
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"text": "Can you add a few more lines to this poem?"},
              {"file_data":{"mime_type": "application/pdf", "file_uri": '$file_uri'}}]
            }]
          }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

You can verify the API successfully stored the uploaded file and get its
metadata by calling [`files.get`](/api/rest/v1beta/files/get). Only the `name`
(and by extension, the `uri`) are unique.  

Python  

    from google import genai
    import pathlib

    client = genai.Client()

    fpath = pathlib.Path('example.txt')
    fpath.write_text('hello')

    file = client.files.upload(file='example.txt')

    file_info = client.files.get(name=file.name)
    print(file_info.model_dump_json(indent=4))

REST  

    name=$(jq ".file.name" file_info.json)
    # Get the file of interest to check state
    curl https://generativelanguage.googleapis.com/v1beta/files/$name > file_info.json
    # Print some information about the file you got
    name=$(jq ".file.name" file_info.json)
    echo name=$name
    file_uri=$(jq ".file.uri" file_info.json)
    echo file_uri=$file_uri

Passing multiple PDFs

The Gemini API is capable of processing multiple PDF documents (up to 1000 pages)
in a single request, as long as the combined size of the documents and the text
prompt stays within the model's context window.  

Python  

    from google import genai
    import io
    import httpx

    client = genai.Client()

    doc_url_1 = "https://arxiv.org/pdf/2312.11805"
    doc_url_2 = "https://arxiv.org/pdf/2403.05530"

    # Retrieve and upload both PDFs using the File API
    doc_data_1 = io.BytesIO(httpx.get(doc_url_1).content)
    doc_data_2 = io.BytesIO(httpx.get(doc_url_2).content)

    sample_pdf_1 = client.files.upload(
      file=doc_data_1,
      config=dict(mime_type='application/pdf')
    )
    sample_pdf_2 = client.files.upload(
      file=doc_data_2,
      config=dict(mime_type='application/pdf')
    )

    prompt = "What is the difference between each of the main benchmarks between these two papers? Output these in a table."

    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=[sample_pdf_1, sample_pdf_2, prompt])
    print(response.text)

JavaScript  

    import { createPartFromUri, GoogleGenAI } from "@google/genai";

    const ai = new GoogleGenAI({ apiKey: "GEMINI_API_KEY" });

    async function uploadRemotePDF(url, displayName) {
        const pdfBuffer = await fetch(url)
            .then((response) => response.arrayBuffer());

        const fileBlob = new Blob([pdfBuffer], { type: 'application/pdf' });

        const file = await ai.files.upload({
            file: fileBlob,
            config: {
                displayName: displayName,
            },
        });

        // Wait for the file to be processed.
        let getFile = await ai.files.get({ name: file.name });
        while (getFile.state === 'PROCESSING') {
            getFile = await ai.files.get({ name: file.name });
            console.log(`current file status: ${getFile.state}`);
            console.log('File is still processing, retrying in 5 seconds');

            await new Promise((resolve) => {
                setTimeout(resolve, 5000);
            });
        }
        if (file.state === 'FAILED') {
            throw new Error('File processing failed.');
        }

        return file;
    }

    async function main() {
        const content = [
            'What is the difference between each of the main benchmarks between these two papers? Output these in a table.',
        ];

        let file1 = await uploadRemotePDF("https://arxiv.org/pdf/2312.11805", "PDF 1")
        if (file1.uri && file1.mimeType) {
            const fileContent = createPartFromUri(file1.uri, file1.mimeType);
            content.push(fileContent);
        }
        let file2 = await uploadRemotePDF("https://arxiv.org/pdf/2403.05530", "PDF 2")
        if (file2.uri && file2.mimeType) {
            const fileContent = createPartFromUri(file2.uri, file2.mimeType);
            content.push(fileContent);
        }

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: content,
        });

        console.log(response.text);
    }

    main();

Go  

    package main

    import (
        "context"
        "fmt"
        "io"
        "net/http"
        "os"
        "google.golang.org/genai"
    )

    func main() {

        ctx := context.Background()
        client, _ := genai.NewClient(ctx, &genai.ClientConfig{
            APIKey:  os.Getenv("GEMINI_API_KEY"),
            Backend: genai.BackendGeminiAPI,
        })

        docUrl1 := "https://arxiv.org/pdf/2312.11805"
        docUrl2 := "https://arxiv.org/pdf/2403.05530"
        localPath1 := "doc1_downloaded.pdf"
        localPath2 := "doc2_downloaded.pdf"

        respHttp1, _ := http.Get(docUrl1)
        defer respHttp1.Body.Close()

        outFile1, _ := os.Create(localPath1)
        _, _ = io.Copy(outFile1, respHttp1.Body)
        outFile1.Close()

        respHttp2, _ := http.Get(docUrl2)
        defer respHttp2.Body.Close()

        outFile2, _ := os.Create(localPath2)
        _, _ = io.Copy(outFile2, respHttp2.Body)
        outFile2.Close()

        uploadConfig1 := &genai.UploadFileConfig{MIMEType: "application/pdf"}
        uploadedFile1, _ := client.Files.UploadFromPath(ctx, localPath1, uploadConfig1)

        uploadConfig2 := &genai.UploadFileConfig{MIMEType: "application/pdf"}
        uploadedFile2, _ := client.Files.UploadFromPath(ctx, localPath2, uploadConfig2)

        promptParts := []*genai.Part{
            genai.NewPartFromURI(uploadedFile1.URI, uploadedFile1.MIMEType),
            genai.NewPartFromURI(uploadedFile2.URI, uploadedFile2.MIMEType),
            genai.NewPartFromText("What is the difference between each of the " +
                                  "main benchmarks between these two papers? " +
                                  "Output these in a table."),
        }
        contents := []*genai.Content{
            genai.NewContentFromParts(promptParts, genai.RoleUser),
        }

        modelName := "gemini-2.5-flash"
        result, _ := client.Models.GenerateContent(
            ctx,
            modelName,
            contents,
            nil,
        )

        fmt.Println(result.Text())
    }

REST  

    DOC_URL_1="https://arxiv.org/pdf/2312.11805"
    DOC_URL_2="https://arxiv.org/pdf/2403.05530"
    DISPLAY_NAME_1="Gemini_paper"
    DISPLAY_NAME_2="Gemini_1.5_paper"
    PROMPT="What is the difference between each of the main benchmarks between these two papers? Output these in a table."

    # Function to download and upload a PDF
    upload_pdf() {
      local doc_url="$1"
      local display_name="$2"

      # Download the PDF
      wget -O "${display_name}.pdf" "${doc_url}"

      local MIME_TYPE=$(file -b --mime-type "${display_name}.pdf")
      local NUM_BYTES=$(wc -c < "${display_name}.pdf")

      echo "MIME_TYPE: ${MIME_TYPE}"
      echo "NUM_BYTES: ${NUM_BYTES}"

      local tmp_header_file=upload-header.tmp

      # Initial resumable request
      curl "${BASE_URL}/upload/v1beta/files?key=${GOOGLE_API_KEY}" \
        -D "${tmp_header_file}" \
        -H "X-Goog-Upload-Protocol: resumable" \
        -H "X-Goog-Upload-Command: start" \
        -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
        -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
        -H "Content-Type: application/json" \
        -d "{'file': {'display_name': '${display_name}'}}" 2> /dev/null

      local upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
      rm "${tmp_header_file}"

      # Upload the PDF
      curl "${upload_url}" \
        -H "Content-Length: ${NUM_BYTES}" \
        -H "X-Goog-Upload-Offset: 0" \
        -H "X-Goog-Upload-Command: upload, finalize" \
        --data-binary "@${display_name}.pdf" 2> /dev/null > "file_info_${display_name}.json"

      local file_uri=$(jq ".file.uri" "file_info_${display_name}.json")
      echo "file_uri for ${display_name}: ${file_uri}"

      # Clean up the downloaded PDF
      rm "${display_name}.pdf"

      echo "${file_uri}"
    }

    # Upload the first PDF
    file_uri_1=$(upload_pdf "${DOC_URL_1}" "${DISPLAY_NAME_1}")

    # Upload the second PDF
    file_uri_2=$(upload_pdf "${DOC_URL_2}" "${DISPLAY_NAME_2}")

    # Now generate content using both files
    curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
        -H 'Content-Type: application/json' \
        -X POST \
        -d '{
          "contents": [{
            "parts":[
              {"file_data": {"mime_type": "application/pdf", "file_uri": '$file_uri_1'}},
              {"file_data": {"mime_type": "application/pdf", "file_uri": '$file_uri_2'}},
              {"text": "'$PROMPT'"}
            ]
          }]
        }' 2> /dev/null > response.json

    cat response.json
    echo

    jq ".candidates[].content.parts[].text" response.json

Technical details

Gemini supports a maximum of 1,000 document pages.
Each document page is equivalent to 258 tokens.

While there are no specific limits to the number of pixels in a document besides
the model's [context window](/gemini-api/docs/long-context), larger pages are
scaled down to a maximum resolution of 3072x3072 while preserving their original
aspect ratio, while smaller pages are scaled up to 768x768 pixels. There is no
cost reduction for pages at lower sizes, other than bandwidth, or performance
improvement for pages at higher resolution.

Document types

Technically, you can pass other MIME types for document understanding, like
TXT, Markdown, HTML, XML, etc. However, document vision ***only meaningfully
understands PDFs***. Other types will be extracted as pure text, and the model
won't be able to interpret what we see in the rendering of those files. Any
file-type specifics like charts, diagrams, HTML tags, Markdown formatting, etc.,
will be lost.

Best practices

For best results:

- Rotate pages to the correct orientation before uploading.
- Avoid blurry pages.
- If using a single page, place the text prompt after the page.
