import requests
import json

# --- Configuration ---
# URL of the running FastAPI application
URL = "http://127.0.0.1:8000/hackrx/run"

# Bearer token for authentication
BEARER_TOKEN = "bcc3195dc18ebeba6405e0a6940cc5678c76c5d26cd202f3a79524fb5e83916d"

# Sample request payload
PAYLOAD = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Headers for the request
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

# --- Main execution ---
def main():
    """Sends the test request to the API and prints the response."""
    print("Sending request to the server...")
    print(f"URL: {URL}")
    print(f"Payload: {json.dumps(PAYLOAD, indent=2)}")

    try:
        # Send the POST request
        response = requests.post(URL, headers=headers, json=PAYLOAD, timeout=120) # Increased timeout for model processing

        # Print the results
        print(f"\n--- Response ---")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error Response:")
            try:
                # Try to print JSON error detail if available
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                # Otherwise, print raw text
                print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while sending the request: {e}")

if __name__ == "__main__":
    main()