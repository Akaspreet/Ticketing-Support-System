# Support API

This is the documentation for the Support API, a FastAPI application designed to handle query processing and ticket management. The API provides various endpoints to interact with the system, including query processing, ticket creation, and session management.

## Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
  - [Query Processing](#query-processing)
  - [Ticket Management](#ticket-management)
  - [Session Management](#session-management)
- [Swagger UI](#swagger-ui)
- [ReDoc](#redoc)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Logging](#logging)
- [Dependencies](#dependencies)

## Installation

To set up the Support API, follow these steps:

1. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
    ```
2. **Install Dependencies:**
    ```sh
   pip install -r requirements.txt
    ```
3. **Set Up Environment Variables:**
Create a .env file in the root directory and add the necessary environment variables. Example:
    ```sh
   API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
ENVIRONMENT=production
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORG_KEY=your_openai_org_key
OPENAI_MODEL=your_openai_model
EMBEDDING_MODEL=your_embedding_model
EMBEDDING_DIMENSIONS=your_embedding_dimensions
TIKTOKEN_MODEL_BASE_FOR_TOKENS=your_tiktoken_model_base
ES_HOST=http://localhost:9200
ES_USERNAME=your_es_username
ES_PASSWORD=your_es_password
ES_INDEX_NAME=your_es_index_name
ES_MIN_SCORE=0.2
MONGO_URI=mongodb://localhost:27017
MONGO_DB_TICKETS=your_mongo_db_tickets
MONGO_DB_CHAT_HISTORY=your_mongo_db_chat_history
MONGO_DB_SESSIONS=your_mongo_db_sessions
SESSION_ACTIVE_HOURS=48
LANGCHAIN_TRACING=False
LANGCHAIN_PROJECT=your_langchain_project
LANGSMITH_API_KEY=your_langsmith_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=your_langfuse_host
    ```
4. **Running the Application:**
To run the application, use the following command:
    ```sh
   python -m app.main
    ```

# API Documentation

## API Endpoints

### Query Processing

#### Process a User Query

- **Endpoint:** `POST /query/process`
- **Description:** Process a user query using LangChain and return search results.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "query": "string",
    "session_id": "string"
  }
  ```
- **Response:**
  ```json
  {
    "query": "string",
    "results": [
      {
        "question": "string",
        "answer": "string",
        "category": "string",
        "similarity_score": 0
      }
    ],
    "found": true
  }
  ```

#### Search for Information

- **Endpoint:** `GET /query/search`
- **Description:** Search for information based on a user query using Elasticsearch.
- **Query Parameters:**
  - `query` (string): The search query.
  - `user_id` (string): The user ID for tracking.
  - `top_k` (int, optional): The number of results to return. Default is 5.
- **Response:**
  ```json
  {
    "query": "string",
    "results": [
      {
        "question": "string",
        "answer": "string",
        "category": "string",
        "similarity_score": 0
      }
    ],
    "found": true
  }
  ```

#### Generate a Chat Response

- **Endpoint:** `POST /query/chat`
- **Description:** Generate a natural language response to the user's query.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "query": "string",
    "session_id": "string"
  }
  ```
- **Response:**
  ```json
  {
    "query": "string",
    "results": [
      {
        "question": "string",
        "answer": "string",
        "category": "string",
        "similarity_score": 0
      }
    ],
    "found": true,
    "chat_response": "string",
    "error": "string"
  }
  ```

### Ticket Management

#### Create a New Support Ticket

- **Endpoint:** `POST /ticket/create`
- **Description:** Create a new ticket for a query that requires human assistance.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "query": "string",
    "email": "string",
    "phone_number": "string",
    "additional_info": "string"
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "ticket": {
      "ticket_id": "string",
      "user_id": "string",
      "query": "string",
      "email": "string",
      "phone_number": "string",
      "additional_info": "string",
      "status": "string",
      "created_at": "string",
      "updated_at": "string"
    },
    "message": "string"
  }
  ```

### Session Management

#### Send a Message in a Chat Session

- **Endpoint:** `POST /chat/sessions/sendmessage`
- **Description:** Send a message in a chat session with streaming response.
- **Request Body:**
  ```json
  {
    "user_id": "string",
    "message": "string",
    "session_id": "string"
  }
  ```
- **Response:** Streaming response with the chat message.

#### Retrieve All Messages from a Specific Chat Session

- **Endpoint:** `GET /chat/sessions/{session_id}/messages`
- **Description:** Retrieve all messages from a specific chat session.
- **Response:**
  ```json
  {
    "session_id": "string",
    "messages": [
      {
        "role": "string",
        "message": "string",
        "stream_id": "string"
      }
    ],
    "last_active": "string"
  }
  ```

## Documentation & Dashboard

### Swagger UI

The Swagger UI for the API can be accessed at:
```
http://localhost:8000/api/docs
```

### ReDoc

The ReDoc documentation for the API can be accessed at:
```
http://localhost:8000/api/redoc
```


## Configuration

The application uses environment variables for configuration. Ensure that the `.env` file is correctly set up with all the required variables.

## Logging

The application uses structured logging with the `loguru` library. Logs are stored in the `logs` directory and include timestamps, log levels, and relevant metadata.

# Langfuse Integration with LangChain Service

Welcome to the Langfuse integration guide for LangChain service! This comprehensive guide will walk you through the process of setting up and running Langfuse locally to track traces, spans, and generations. Follow these steps to get started:


## Setup Instructions

### 1. Clone the Langfuse Repository

Begin by cloning the Langfuse repository from GitHub:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse

```

### 2. Build and Run Docker Containers

Use Docker Compose to build and run the necessary containers in detached mode:

```bash
docker-compose up -d
```

### 3. Run the Application
Start your application by executing the following command in your terminal:

```bash
python -m app.main
```

### 4. Access the Dashboad
Open your web browser and navigate to http://localhost:3000. This will give you access to the Langfuse dashboard, where you can monitor traces, spans, and generations as you interact with your APIs.

## Dependencies

The application depends on several external services and libraries, including:

- **FastAPI**: For building the API.
- **Uvicorn**: ASGI server for serving the application.
- **Pydantic**: For data validation and settings management.
- **Elasticsearch**: For search functionality.
- **MongoDB**: For storing tickets, chat history, and sessions.
- **LangChain**: For integrating with language models.
- **OpenAI**: For generating text and embeddings.
- **Langfuse and LangSmith**: For monitoring and tracing.
