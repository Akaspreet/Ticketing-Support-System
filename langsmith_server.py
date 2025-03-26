from langsmith.client import Client
import os
import sys

def start_local_server():
    """Start a local LangSmith tracing environment"""
    # Set environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "local-dev-key"  # Any value works for local
    os.environ["LANGCHAIN_PROJECT"] = "digital-chatbot"

    print("Starting LangSmith local tracing...")
    print("Ensure LangSmith is running at http://localhost:3000")

    try:
        client = Client()  # Just instantiate the client, no need for start_trace_server()
        print("LangSmith tracing initialized successfully.")
    except Exception as e:
        print(f"Error initializing LangSmith: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_local_server()
