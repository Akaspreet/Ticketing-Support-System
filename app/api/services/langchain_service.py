# """
# Service for integrating with LangChain and processing queries.
# """
# #pylint: disable=too-many-lines
# from typing import Dict, List, Any, AsyncIterator
# from datetime import datetime
# import json
# import uuid
# import asyncio
# import threading
# import tiktoken

# # Import existing modules
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.agents import initialize_agent, AgentType, Tool
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain           #pylint: disable = no-name-in-module
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.schema import LLMResult
# from langfuse.callback import CallbackHandler   #pylint: disable = import-error

# from app.api.core.config import settings
# from app.api.core.logging import app_logger
# from app.api.services.elastic_search_service import es_service
# from app.api.services.ticket_service import ticket_service
# from app.api.db.mongodb import mongodb_client
# from app.api.services.langsmith_service import langsmith_service
# from app.api.services.verification_service import verification_service
# from app.api.services.langfuse_service import langfuse_service

# # Create a global dictionary to track active streaming sessions
# active_streams = {}
# stream_lock = threading.Lock()

# # Encoder for token counting
# class TokenCounter:     #pylint:disable=too-few-public-methods
#     """Class to count tokens for OpenAI models"""

#     def __init__(self, model_name=settings.OPENAI_MODEL):
#         try:
#             self.encoding = tiktoken.encoding_for_model(model_name)
#         except KeyError:
#             self.encoding = tiktoken.get_encoding(settings.TIKTOKEN_MODEL_BASE_FOR_TOKENS)
#         app_logger.info(f"Initialized TokenCounter with encoding for {model_name}")

#     def count_tokens(self, text):
#         """Count the number of tokens in a text string"""
#         if not text:
#             return 0
#         token_ids = self.encoding.encode(text)
#         token_count = len(token_ids)
#         return token_count

# #pylint: disable = abstract-method
# class TokenTrackingCallbackHandler(BaseCallbackHandler):        #pylint: disable=too-many-instance-attributes
#     """Custom callback handler for streaming responses with token tracking."""

#     def __init__(self, queue, stream_id, token_counter):
#         super().__init__()
#         self._queue = queue
#         self._stop_signal = "DONE"
#         self._stream_id = stream_id
#         self._is_interrupted = False
#         self._token_counter = token_counter
#         self._input_tokens = 0
#         self._output_tokens = 0
#         self._output_text = ""
#         app_logger.info(f"Initialized TokenTracking Callback Handler for stream {stream_id}")

#     def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
#         """Run when LLM starts running. Count input tokens."""
#         self._input_tokens = sum(self._token_counter.count_tokens(p) for p in prompts)
#         app_logger.info(f"""[Token Tracker] Stream
#                         {self._stream_id} -
#                         Input tokens: {self._input_tokens}""")

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         """Called when the LLM produces a new token."""
#         # Check if stream has been interrupted
#         with stream_lock:
#             if active_streams.get(self._stream_id, {}).get("interrupted", False):
#                 self._is_interrupted = True
#                 self._queue.put("[INTERRUPTED]")
#                 return

#         # Continue with normal token streaming if not interrupted
#         self._output_text += token
#         self._output_tokens += 1

#         # Log every 10 tokens
#         if self._output_tokens > 0:
#             app_logger.debug(f"""[Token Tracker]
#                              Stream {self._stream_id} -
#                              Output tokens so far:
#                              {self._output_tokens}""")

#         self._queue.put(token)

#     def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
#         """Run when LLM ends running."""
#         # Log final token counts
#         app_logger.info(f"[Token Tracker] Stream {self._stream_id} - Final counts:")
#         app_logger.info(f"[Token Tracker] Input tokens: {self._input_tokens}")
#         app_logger.info(f"[Token Tracker] Output tokens: {self._output_tokens}")
#         app_logger.info(f"[Token Tracker] Total tokens: {self._input_tokens + self._output_tokens}")

#         # Check output token count using tiktoken for verification
#         actual_output_tokens = self._token_counter.count_tokens(self._output_text)
#         app_logger.info(f"""[Token Tracker]
#                         Verified output tokens (tiktoken):
#                         {actual_output_tokens}""")

#         # If we were interrupted, add a special token to indicate interruption
#         if self._is_interrupted:
#             self._queue.put("[INTERRUPTED]")

#         # Always send the stop signal
#         self._queue.put(self._stop_signal)

#         # Clean up the stream record
#         with stream_lock:
#             if self._stream_id in active_streams:
#                 active_streams[self._stream_id]["token_usage"] = {
#                     "input_tokens": self._input_tokens,
#                     "output_tokens": self._output_tokens,
#                     "total_tokens": self._input_tokens + self._output_tokens
#                 }
#                 del active_streams[self._stream_id]

# #pylint: disable = abstract-method
# class AsyncTokenTrackingCallbackHandler(BaseCallbackHandler):
#     """Async callback handler for streaming responses with token tracking."""

#     def __init__(self, token_counter):
#         super().__init__()
#         self._token_counter = token_counter
#         self._input_tokens = 0
#         self._output_tokens = 0
#         self._output_text = ""
#         self._is_interrupted = False
#         self._stream_id = None
#         self.tokens = []  # Store tokens here for async access
#         app_logger.info("Initialized AsyncTokenTracking Callback Handler")

#     def set_stream_id(self, stream_id):
#         """Set the stream ID for this handler."""
#         self._stream_id = stream_id

#     def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
#         """Run when LLM starts running. Count input tokens."""
#         self._input_tokens = sum(self._token_counter.count_tokens(p) for p in prompts)
#         app_logger.info(f"""[Token Tracker] Stream
#                         {self._stream_id} - Input tokens: {self._input_tokens}""")

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         """Called when the LLM produces a new token."""
#         # Check if stream has been interrupted
#         with stream_lock:
#             if active_streams.get(self._stream_id, {}).get("interrupted", False):
#                 self._is_interrupted = True
#                 self.tokens.append("[INTERRUPTED]")
#                 return

#         # Continue with normal token streaming if not interrupted
#         self._output_text += token
#         self._output_tokens += 1
#         self.tokens.append(token)

#         # Log every 10 tokens
#         if self._output_tokens % 10 == 0:
#             app_logger.debug(f"""[Token Tracker] Stream
#                              {self._stream_id} - Output tokens so far: {self._output_tokens}""")

#     def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
#         """Run when LLM ends running."""
#         # Log final token counts
#         app_logger.info(f"[Token Tracker] Stream {self._stream_id} - Final counts:")
#         app_logger.info(f"[Token Tracker] Input tokens: {self._input_tokens}")
#         app_logger.info(f"[Token Tracker] Output tokens: {self._output_tokens}")
#         app_logger.info(f"[Token Tracker] Total tokens: {self._input_tokens + self._output_tokens}")

#         # Check output token count using tiktoken for verification
#         actual_output_tokens = self._token_counter.count_tokens(self._output_text)
#         app_logger.info(f"""[Token Tracker] Verified
#                         output tokens (tiktoken): {actual_output_tokens}""")

#         # If we were interrupted, add a special token to indicate interruption
#         if self._is_interrupted:
#             self.tokens.append("[INTERRUPTED]")

#         # Add a special token to indicate completion
#         self.tokens.append("[DONE]")

#         # Update the stream record
#         with stream_lock:
#             if self._stream_id in active_streams:
#                 active_streams[self._stream_id]["token_usage"] = {
#                     "input_tokens": self._input_tokens,
#                     "output_tokens": self._output_tokens,
#                     "total_tokens": self._input_tokens + self._output_tokens
#                 }
# completed_responses = {}

# class LangChainService:             #pylint: disable=too-many-instance-attributes
#     """Service for LangChain integration and query processing with token tracking."""

#     def __init__(self):
#         """Initialize LangChain components."""
#         # Initialize token counter
#         self.token_counter = TokenCounter(settings.OPENAI_MODEL)


#         # Track total token usage
#         self.total_token_usage = {
#             "input_tokens": 0,
#             "output_tokens": 0,
#             "total_tokens": 0,
#             "request_count": 0
#         }

#         self.langfuse_callbacks = [CallbackHandler(
#             public_key=settings.LANGFUSE_PUBLIC_KEY,
#             secret_key=settings.LANGFUSE_SECRET_KEY,
#             host=settings.LANGFUSE_HOST
#         )] if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY else []

#         # General initialization
#         # self.stream_queues = {}  # Changed to dict to track multiple streams
#         self.top_k = settings.top_k

#         # Initialize LLMs
#         self.llm = ChatOpenAI(
#             temperature=0.0,
#             model_name=settings.OPENAI_MODEL,
#             callbacks=self.langfuse_callbacks
#         )

#         # Initialize streaming LLM
#         self.streaming_llm = ChatOpenAI(
#             temperature=0.4,
#             model_name=settings.OPENAI_MODEL,
#             streaming=True,
#             callbacks = self.langfuse_callbacks
#         )

#         self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

#         # Initialize chat history collection
#         self.chat_history_collection = mongodb_client.chat_history_db.chat_history

#         # New collection for token usage statistics
#         self.token_usage_collection = mongodb_client.chat_history_db.token_usage

#         # Set up the tools
#         self.tools = self._create_tools()

#         # Set up the agent
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


#         agent_prefix = """You are a helpful assistant.
#         You can answer questions using the available tools.
#         For creating tickets, you'll need to collect the user's
#         email and phone number, and verify both.

#             Follow these steps for ticket creation:
#             1. Start with user_id and query
#             2. Ask for email
#             3. Send verification code to email
#             4. Collect and verify the email code
#             5. Ask for phone number
#             6. Send verification code to phone
#             7. Collect and verify the phone code
#             8. Create the ticket
#             Lead the user through this process patiently and clearly.
#         """

#         self.agent = initialize_agent(
#             self.tools,
#             self.llm,
#             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#             memory=self.memory,
#             verbose=settings.DEBUG,
#             agent_kwargs={"prefix": agent_prefix}

#         )

#         self.chat_prompt = PromptTemplate(
#             input_variables=["query", "search_results"],
#             template="""
#             You are a helpful assistant. Based on the following search results, provide a concise,
#             natural-sounding response to the user's query. If the search results don't contain
#             relevant information, suggest what the user might do next.

#             User Query: {query}

#             Search Results:
#             {search_results}

#             Response:
#             """
#         )

#         # Regular chat chain
#         self.chat_chain = LLMChain(
#             llm=ChatOpenAI(temperature=0.4,
#                            model_name=settings.OPENAI_MODEL,
#                            callbacks=self.langfuse_callbacks),
#             prompt=self.chat_prompt
#         )

#         # Streaming chat chain
#         self.streaming_chat_chain = LLMChain(
#             llm=self.streaming_llm,
#             prompt=self.chat_prompt
#         )


#         app_logger.info("Initialized LangChain Service with token tracking")

#     #pylint: disable=too-many-arguments
#     #pylint: disable=too-many-positional-arguments
#     def _track_token_usage(self,
#                            input_text,
#                            output_text,
#                            source,
#                            user_id=None,
#                            query=None):
#         """
#         Track token usage for a request.

#         Args:
#             input_text: The input text sent to the model
#             output_text: The output text generated by the model
#             source: Source of the request (e.g., "chat", "agent", "search")
#             user_id: Optional user ID
#             query: Optional query text
#         """
#         input_tokens = self.token_counter.count_tokens(input_text)
#         output_tokens = self.token_counter.count_tokens(output_text)
#         total_tokens = input_tokens + output_tokens

#         # Log the token usage
#         app_logger.info(f"""[Token Tracker]
#                         {source.upper()} -
#                         Input tokens at line 1246:
#                         {input_tokens}""")
#         app_logger.info(f"""[Token Tracker]
#                         {source.upper()} -
#                         Output tokens at line 1247:
#                         {output_tokens}""")
#         app_logger.info(f"""[Token Tracker]
#                         {source.upper()} -
#                         Total tokens at line 1248:
#                         {total_tokens}""")

#         # Update global counters
#         self.total_token_usage["input_tokens"] += input_tokens
#         self.total_token_usage["output_tokens"] += output_tokens
#         self.total_token_usage["total_tokens"] += total_tokens
#         self.total_token_usage["request_count"] += 1

#         # Save to database
#         usage_record = {
#             "timestamp": datetime.utcnow(),
#             "source": source,
#             "input_tokens": input_tokens,
#             "output_tokens": output_tokens,
#             "total_tokens": total_tokens,
#             "user_id": user_id,
#             "query": query
#         }

#         try:
#             self.token_usage_collection.insert_one(usage_record)
#         except Exception as e:          #pylint: disable=broad-exception-caught
#             app_logger.error(f"Error saving token usage: {str(e)}")

#         return {
#             "input_tokens": input_tokens,
#             "output_tokens": output_tokens,
#             "total_tokens": total_tokens
#         }

#     def _create_tools(self) -> List[Tool]:
#         """Create LangChain tools for different components."""
#         # Tool for retrieving information from Elasticsearch
#         get_info_tool = Tool(
#             name="GetInformation",
#             func=self._get_information,
#             description="""Useful for retrieving information based on the user's query.
#                             Input should be a user question."""
#         )

#         # Tool for creating tickets
#         create_ticket_tool = Tool(
#             name="RaiseTicket",
#             func=self._raise_ticket,
#             description="""Useful for raising a ticket when information cannot be found.
#                         Input should be a JSON string with these possible fields:
#                         'user_id' (required): The ID of the user
#                         'query' (required): The user's original question or request
#                         'additional_info' (optional): Any additional context
#                         'email' (optional): User's email address if in the ticket creation flow
#                         'phone_number' (optional): User's phone number if in the ticket creation flow
#                         'otp' (optional): OTP code provided by user during verification"""
#             )

#         # Tool for retrieving tickets
#         get_ticket_tool = Tool(
#             name="GetTicket",
#             func=self._get_ticket,
#             description="""Useful for retrieving information about a specific ticket.
#                             Input should be a ticket ID."""
#         )

#         return [get_info_tool, create_ticket_tool, get_ticket_tool]

#     def _get_information(self, query: str) -> str:
#         """Get information from Elasticsearch with token tracking."""
#         app_logger.info(f"Getting information for query: {query}")
#         input_text = query

#         # Perform search using the Elasticsearch service
#         results = es_service.hybrid_search(query, top_k=self.top_k)

#         output_text = ""
#         if not results:
#             output_text = "Information not found. Consider raising a ticket for this query."
#         else:
#             # Format results as a single string
#             response_parts = ["Here's what I found:"]
#             for i, result in enumerate(results, 1):
#                 response_parts.append(f"\n\n{i}. {result['question']}")
#                 response_parts.append(f"   Answer: {result['answer']}")
#                 response_parts.append(f"   Confidence: {result['similarity_score']:.2f}")
#             output_text = "\n".join(response_parts)

#         token_usage = self._track_token_usage(input_text, output_text, "elasticsearch_search")
#         return output_text, token_usage

#     #pylint: disable = too-many-return-statements
#     def _raise_ticket(self, input_str: str) -> str:
#         """Raise a ticket in the system with token tracking."""
#         try:
#             # Parse input as JSON
#             data = json.loads(input_str)

#             user_id = data.get("user_id")
#             query = data.get("query")
#             email = data.get("email")
#             phone_number = data.get("phone_number")
#             additional_info = data.get("additional_info", "")

#             if not user_id or not query:
#                 output_text = "Error: Missing required fields (user_id, query)"
#                 token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#                 return output_text, token_usage

#             # Interactive collection and verification of email if not provided
#             if not email:
#                 output_text = "Please provide your email address for ticket creation."
#                 token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#                 return output_text, token_usage

#             # Attempt to verify the email with OTP
#             try:
#                 verification_service.send_otp(email, "email")
#                 output_text = f"""An OTP has been sent
#                 to your email {email}.
#                 Please provide the OTP to verify."""
#                 token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#                 return output_text, token_usage

#             except Exception as e:              #pylint: disable=broad-exception-caught
#                 output_text = f"""Error sending email verification:
#                 {str(e)}.
#                 Please provide a valid email address."""
#                 token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#                 # return output_text
#                 # return output_text, token_usage

#             # Interactive collection and verification of phone number if not provided
#             if not phone_number:
#                 output_text = "Please provide your phone number for ticket creation."
#                 token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#                 return output_text, token_usage

#             # Attempt to verify the phone number with OTP
#             try:
#                 verification_service.send_otp(phone_number, "phone")
#                 output_text = f"""An OTP has been sent to your phone {phone_number}.
#                 Please provide the OTP to verify."""
#                 token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#                 return output_text, token_usage
#             except Exception as e:          #pylint: disable=broad-exception-caught
#                 output_text = f"""Error sending phone verification: {str(e)}.
#                 Please provide a valid phone number."""
#                 self._track_token_usage(input_str, output_text, "raise_ticket")
#                 # return output_text

#             # Create ticket using the ticket service
#             ticket = ticket_service.create_ticket(user_id,
#                                                   query,
#                                                   email,
#                                                   phone_number,
#                                                   additional_info)

#             if ticket:
#                 output_text = f"Ticket created successfully with ID: {ticket['ticket_id']}"
#                 token_usage = self._track_token_usage(input_str,
#                                                       output_text,
#                                                       "raise_ticket",
#                                                       user_id, query)
#                 return output_text, token_usage

#             output_text = "Error: Failed to create ticket"
#             token_usage = self._track_token_usage(input_str,
#                                                   output_text,
#                                                   "raise_ticket",
#                                                   user_id,
#                                                   query)
#             return output_text, token_usage

#         except Exception as e:          #pylint: disable=broad-exception-caught
#             app_logger.error(f"Error in raise_ticket: {str(e)}")
#             output_text = f"Error: {str(e)}"
#             token_usage = self._track_token_usage(input_str, output_text, "raise_ticket")
#             return output_text, token_usage




#     def _get_ticket(self, ticket_id: str) -> str:
#         """Get ticket information with token tracking."""
#         input_text = f"Get ticket: {ticket_id}"

#         try:
#             ticket = ticket_service.get_ticket(ticket_id)

#             if not ticket:
#                 output_text = f"Ticket with ID {ticket_id} not found"
#                 token_usage = self._track_token_usage(input_text, output_text, "get_ticket")
#                 return output_text, token_usage

#             # Format ticket information
#             created_at = ticket["created_at"].strftime("%Y-%m-%d %H:%M:%S")
#             updated_at = ticket.get("updated_at")
#             updated_at = updated_at.strftime("%Y-%m-%d %H:%M:%S") if updated_at else "N/A"

#             output_text = (
#                 f"Ticket ID: {ticket['ticket_id']}\n"
#                 f"Status: {ticket['status']}\n"
#                 f"User ID: {ticket['user_id']}\n"
#                 f"Query: {ticket['query']}\n"
#                 f"Additional Info: {ticket['additional_info'] or 'N/A'}\n"
#                 f"Created: {created_at}\n"
#                 f"Last Updated: {updated_at}"
#             )

#             token_usage = self._track_token_usage(input_text,
#                                     output_text,
#                                     "get_ticket",
#                                     ticket['user_id'],
#                                     ticket['query'])
#             return output_text, token_usage

#         except Exception as e:          #pylint: disable=broad-exception-caught
#             app_logger.error(f"Error in get_ticket: {str(e)}")
#             output_text = f"Error retrieving ticket: {str(e)}"
#             token_usage = self._track_token_usage(input_text, output_text, "get_ticket")
#             return output_text, token_usage


#     def process_query(self, user_id: str, query: str) -> Dict[str, Any]:    #pylint: disable=too-many-locals
#         """Process a user query through the LangChain agent with token tracking."""
#         try:
#             app_logger.info(f"Processing query for user {user_id}: {query}")

#             # Create a Langfuse trace for this query
#             trace = langfuse_service.create_trace(
#                 name="process_query",
#                 user_id=user_id,
#                 metadata={"query": query}
#             )
#             trace_id = trace.id if trace else None

#             # Get information directly first
#             search_span = None
#             if trace_id:
#                 search_span = langfuse_service.create_span(
#                     trace_id=trace_id,
#                     name="elasticsearch_search",
#                     metadata={"query": query}
#                 )

#             search_results = es_service.hybrid_search(query, top_k=self.top_k)
#             found = len(search_results) > 0

#             if search_span:
#                 search_span.end()

#             # Prepare response
#             response = {
#                 "query": query,
#                 "results": search_results,
#                 "found": found,
#                 "agent_response": None,
#                 "token_usage": token_usage
#             }

#             # If not found, let the agent decide next steps
#             if not found:
#                 app_logger.info("No results found, processing with agent")
#                 agent_input = f"""User {user_id} asked: {query}.
#                             No information was found. Should I raise a ticket?"""

#                 agent_span = None
#                 if trace_id:
#                     agent_span = langfuse_service.create_span(
#                         trace_id=trace_id,
#                         name="agent_run",
#                         metadata={"input": agent_input}
#                     )

#                 agent_response, agent_token_usage = self.agent.run(input=agent_input)
#                 response["agent_response"] = agent_response
#                 response["token_usage"]["input_tokens"] += agent_token_usage["input_tokens"]
#                 response["token_usage"]["output_tokens"] += agent_token_usage["output_tokens"]
#                 response["token_usage"]["total_tokens"] += agent_token_usage["total_tokens"]

#                 if agent_span:
#                     agent_span.end()

#                 # Track token usage for agent
#                 token_usage = self._track_token_usage(
#                     agent_input,
#                     agent_response,
#                     "agent",
#                     user_id,
#                     query
#                 )

#                 # Record the generation in Langfuse
#                 if trace_id:
#                     langfuse_service.create_generation(
#                         trace_id=trace_id,
#                         name="agent_generation",
#                         model=settings.OPENAI_MODEL,
#                         prompt=agent_input,
#                         completion=agent_response,
#                         token_usage=token_usage
#                     )

#             # Store in chat history
#             # self._save_to_chat_history(user_id, query, response)

#             # End the trace
#             if trace_id:
#                 langfuse_service.end_trace(trace_id, output=response)

#             # Create LangSmith run
#             run_id = langsmith_service.create_run(name="process_query",
#                                                 inputs={"user_id": user_id, "query": query})
#             langsmith_service.update_run(run_id, outputs=response)

#             return response

#         except Exception as e:      #pylint: disable=broad-exception-caught
#             app_logger.error(f"Error processing query: {str(e)}")

#             # End the trace with error
#             if trace_id:
#                 langfuse_service.end_trace(trace_id, error=str(e))

#             # Update LangSmith run with error
#             langsmith_service.update_run(run_id, outputs={}, error=str(e))

#             return {
#                 "query": query,
#                 "results": [],
#                 "found": False,
#                 "error": str(e)
#             }

#     #pylint: disable=too-many-locals
#     def generate_chat_response(self, user_id: str, query: str, session_id: str) -> Dict[str, Any]:
#         """Generate a conversational response with token tracking."""
#         trace_id = None
#         run_id = None
#         try:
#             app_logger.info(f"Generating chat response for user {user_id}: {query}")

#             # Create a Langfuse trace for this query
#             trace = langfuse_service.create_trace(
#                 name="generate_chat_response",
#                 user_id=user_id,
#                 metadata={"query": query, "session_id": session_id}
#             )
#             trace_id = trace.id if trace else None

#             # Create LangSmith run
#             run_id = langsmith_service.create_run(name="generate_chat_response",
#                                                 inputs={"user_id": user_id,
#                                                         "query": query,
#                                                         "session_id": session_id})

#             # Get search results
#             search_span = None
#             if trace_id:
#                 search_span = langfuse_service.create_span(
#                     trace_id=trace_id,
#                     name="elasticsearch_search",
#                     metadata={"query": query}
#                 )

#             search_results = es_service.hybrid_search(query, top_k=self.top_k)
#             found = len(search_results) > 0

#             if search_span:
#                 search_span.end()

#             # Format search results for the prompt
#             formatted_results = ""
#             if found:
#                 for i, result in enumerate(search_results, 1):
#                     formatted_results += f"{i}. Question: {result['question']}\n"
#                     formatted_results += f"   Answer: {result['answer']}\n"
#                     formatted_results += f"   Category: {result.get('category', 'General')}\n"
#                     formatted_results += f"   Confidence: {result['similarity_score']:.2f}\n\n"
#             else:
#                 formatted_results = "No relevant information found."

#             # Prepare the prompt input
#             prompt_input = {
#                 "query": query,
#                 "search_results": formatted_results
#             }

#             # Convert prompt input to string for token counting
#             prompt_str = self.chat_prompt.format(**prompt_input)

#             # Generate a natural language response using the LLM chain
#             llm_span = None
#             if trace_id:
#                 llm_span = langfuse_service.create_span(
#                     trace_id=trace_id,
#                     name="llm_chain",
#                     metadata={"prompt": prompt_str}
#                 )

#             llm_response = self.chat_chain.run(**prompt_input)

#             if llm_span:
#                 llm_span.end()

#             # Track token usage
#             token_usage = self._track_token_usage(
#                 prompt_str,
#                 llm_response,
#                 "chat_response",
#                 user_id,
#                 query
#             )

#             # Record the generation in Langfuse
#             if trace_id:
#                 langfuse_service.create_generation(
#                     trace_id=trace_id,
#                     name="chat_generation",
#                     model=settings.OPENAI_MODEL,
#                     prompt=prompt_str,
#                     completion=llm_response,
#                     token_usage=token_usage
#                 )

#             # Prepare the complete response
#             response = {
#                 "query": query,
#                 "results": search_results,
#                 "found": found,
#                 "chat_response": llm_response.strip(),
#                 "token_usage": token_usage,
#                 "session_id": session_id
#             }

#             # Save to chat history
#             # self._save_to_chat_history(user_id, session_id, query, response)

#             # End the trace
#             if trace_id:
#                 langfuse_service.end_trace(trace_id, output=response)

#             # Update LangSmith run
#             langsmith_service.update_run(run_id, outputs=response)

#             return response

#         except Exception as e:      #pylint: disable=broad-exception-caught
#             app_logger.error(f"Error generating chat response: {str(e)}")

#             # End the trace with error
#             if trace_id:
#                 langfuse_service.end_trace(trace_id, error=str(e))

#             # Update LangSmith run with error
#             langsmith_service.update_run(run_id, outputs={}, error=str(e))

#             return {
#                 "query": query,
#                 "results": [],
#                 "found": False,
#                 "chat_response": "I'm sorry, I encountered an error while processing your request.",
#                 "error": str(e)
#             }


#     def interrupt_stream(self, stream_id: str) -> bool:
#         """
#         Interrupt an ongoing streaming response.

#         Args:
#             stream_id: The ID of the stream to interrupt

#         Returns:
#             bool: True if the stream was successfully interrupted, False otherwise
#         """
#         with stream_lock:
#             if stream_id in active_streams:
#                 active_streams[stream_id]["interrupted"] = True
#                 app_logger.info(f"Stream {stream_id} marked for interruption")
#                 return True

#             app_logger.warning(f"Stream {stream_id} not found for interruption")
#             return False

#     async def _cleanup_completed_response(self, stream_id: str, delay: int = 300):
#         """
#         Clean up a completed response after a delay.

#         Args:
#             stream_id (str): The stream ID to clean up.
#             delay (int): The delay in seconds before cleaning up.
#         """
#         await asyncio.sleep(delay)  # Wait for 5 minutes by default
#         if stream_id in completed_responses:
#             del completed_responses[stream_id]


#     async def get_full_response(self, stream_id: str) -> str:
#         """
#         Get the full response for a given stream ID.

#         Args:
#             stream_id (str): The stream ID to retrieve the response for.

#         Returns:
#             str: The full response text.

#         Raises:
#             ValueError: If the response for the given stream ID is not found.
#         """
#         # First, check if the response is already in the completed_responses dict
#         if stream_id in completed_responses:
#             return completed_responses[stream_id]

#         # If not, wait for a short time to see if it becomes available
#         # This is needed because the response might still be processing
#         for _ in range(10):  # Try for up to 1 second
#             await asyncio.sleep(0.1)
#             if stream_id in completed_responses:
#                 return completed_responses[stream_id]

#         # If still not found, raise an error
#         raise ValueError(f"Response for stream_id {stream_id} not found")

#     #pylint: disable = too-many-statements
#     #pylint: disable = too-many-branches
#     async def generate_streaming_chat_response(
#     self, user_id: str, query: str, session_id: str, stream_id: str = None
# ) -> AsyncIterator[str]:
#         """
#         Generate a streaming conversational response with token tracking.
#         """
#         run_id = None
#         trace_id = None
#         try:
#             # Create a Langfuse trace for this streaming query
#             trace = langfuse_service.create_trace(
#                 name="streaming_chat_response",
#                 user_id=user_id,
#                 metadata={"query": query, "session_id": session_id}
#             )
#             trace_id = trace.id if trace else None

#             # Create LangSmith run
#             run_id = langsmith_service.create_run(
#                 name="generate_streaming_chat_response",
#                 inputs={"user_id": user_id, "query": query, "session_id": session_id}
#             )

#             # Generate a stream ID if not provided
#             if not stream_id:
#                 stream_id = str(uuid.uuid4())

#             app_logger.info(f"""Generating streaming chat
#                             response for user {user_id}, stream {stream_id}: {query}""")


#             # Register this stream in the active streams dictionary
#             with stream_lock:
#                 active_streams[stream_id] = {
#                     "user_id": user_id,
#                     "query": query,
#                     "interrupted": False,
#                     "start_time": datetime.utcnow(),
#                     "token_usage": {
#                         "input_tokens": 0,
#                         "output_tokens": 0
#                     }
#                 }

#             # Create a callback handler for token tracking
#             async_handler = AsyncTokenTrackingCallbackHandler(self.token_counter)
#             async_handler.set_stream_id(stream_id)

#             # Get search results
#             search_span = None
#             if trace_id:
#                 search_span = langfuse_service.create_span(
#                     trace_id=trace_id,
#                     name="elasticsearch_search",
#                     metadata={"query": query}
#                 )

#             search_results = es_service.hybrid_search(query, top_k=self.top_k)
#             # search_results, token_usage = self._get_information(query)
#             found = len(search_results) > 0

#             if search_span:
#                 search_span.end()

#             # Format search results for the prompt
#             formatted_results = ""
#             if found:
#                 for i, result in enumerate(search_results, 1):
#                     formatted_results += f"{i}. Question: {result['question']}\n"
#                     formatted_results += f"   Answer: {result['answer']}\n"
#                     formatted_results += f"   Category: {result.get('category', 'General')}\n"
#                     formatted_results += f"   Confidence: {result['similarity_score']:.2f}\n\n"
#             else:
#                 formatted_results = "No relevant information found."

#             # Create properly configured streaming LLM instance
#             streaming_llm = ChatOpenAI(
#                 temperature=0.4,
#                 model_name=settings.OPENAI_MODEL,
#                 streaming=True,
#                 callbacks=[async_handler]
#             )

#             # Create streaming chat chain with the handler
#             streaming_chat_chain = LLMChain(
#                 llm=streaming_llm,
#                 prompt=self.chat_prompt
#             )

#             # Calculate tokens in the prompt template
#             prompt_str = self.chat_prompt.format(
#                 query=query,
#                 search_results=formatted_results
#             )
#             prompt_tokens = self.token_counter.count_tokens(prompt_str)
#             app_logger.info(f"[Token Tracker] Stream {stream_id} - Prompt tokens: {prompt_tokens}")

#             # Create a streaming span in Langfuse
#             streaming_span = None
#             if trace_id:
#                 streaming_span = langfuse_service.create_span(
#                     trace_id=trace_id,
#                     name="streaming_llm",
#                     metadata={"prompt": prompt_str}
#                 )

#             # Start a task to run the LLM chain
#             task = asyncio.create_task(
#                 streaming_chat_chain.ainvoke(
#                     {
#                         "query": query,
#                         "search_results": formatted_results
#                     }
#                 )
#             )

#             full_response = ""
#             was_interrupted = False
#             output_token_count = 0

#             # Keep checking for new tokens until we're done
#             last_idx = 0
#             while True:
#                 # Check for new tokens
#                 if len(async_handler.tokens) > last_idx:
#                     for i in range(last_idx, len(async_handler.tokens)):
#                         token = async_handler.tokens[i]

#                         # Check for special tokens
#                         #pylint: disable = no-else-break
#                         if token == "[DONE]":
#                             # We're finished
#                             break
#                         #pylint: disable = no-else-break
#                         elif token == "[INTERRUPTED]":      #pylint: disable = no-else-break
#                             was_interrupted = True
#                             yield "[INTERRUPTED]"
#                             break
#                         #pylint: disable = no-else-break
#                         else:                               #pylint: disable = no-else-break
#                             # Normal token
#                             full_response += token
#                             output_token_count += 1
#                             yield token

#                     # Update our position
#                     last_idx = len(async_handler.tokens)

#                 # Check if we're done
#                 if "[DONE]" in async_handler.tokens or "[INTERRUPTED]" in async_handler.tokens:
#                     break

#                 # Pause briefly to avoid busy-waiting
#                 await asyncio.sleep(0.01)

#             # Wait for the task to complete
#             await task

#             # End the streaming span
#             if streaming_span:
#                 streaming_span.end()

#             # Get the final token count
#             actual_input_tokens = self.token_counter.count_tokens(prompt_str)
#             actual_output_tokens = self.token_counter.count_tokens(full_response)

#             app_logger.info(f"""[Token Tracker] Stream
#                             {stream_id} - Finalized token counts:""")
#             app_logger.info(f"""[Token Tracker] Input
#                             tokens (verified): {actual_input_tokens}""")
#             app_logger.info(f"""[Token Tracker] Output tokens
#                             (verified): {actual_output_tokens}""")
#             app_logger.info(f"""[Token Tracker] Total tokens:
#                             {actual_input_tokens + actual_output_tokens}""")

#             # Prepare the complete response for saving to history
#             token_usage = {
#                 "input_tokens": actual_input_tokens,
#                 "output_tokens": actual_output_tokens,
#                 "total_tokens": actual_input_tokens + actual_output_tokens
#             }

#             # Record the generation in Langfuse
#             if trace_id:
#                 langfuse_service.create_generation(
#                     trace_id=trace_id,
#                     name="streaming_generation",
#                     model=settings.OPENAI_MODEL,
#                     prompt=prompt_str,
#                     completion=full_response,
#                     token_usage=token_usage
#                 )

#             # Update global counters
#             self.total_token_usage["input_tokens"] += actual_input_tokens
#             self.total_token_usage["output_tokens"] += actual_output_tokens
#             self.total_token_usage["total_tokens"] += actual_input_tokens + actual_output_tokens
#             self.total_token_usage["request_count"] += 1


#             response = {
#                 "query": query,
#                 "results": search_results,
#                 "found": found,
#                 "chat_response": full_response.strip(),
#                 "was_interrupted": was_interrupted,
#                 "stream_id": stream_id,
#                 "token_usage": token_usage,
#                 "session_id": session_id
#             }

#             # Store the full response for later retrieval
#             completed_responses[stream_id] = full_response.strip()

#             # Set up a task to remove the response after a while to prevent memory leaks
#             asyncio.create_task(self._cleanup_completed_response(stream_id))

#             # Save token usage to database
#             usage_record = {
#                 "timestamp": datetime.utcnow(),
#                 "source": "streaming_chat",
#                 "input_tokens": actual_input_tokens,
#                 "output_tokens": actual_output_tokens,
#                 "total_tokens": actual_input_tokens + actual_output_tokens,
#                 "user_id": user_id,
#                 "query": query,
#                 "was_interrupted": was_interrupted,
#                 "stream_id": stream_id
#             }

#             try:
#                 self.token_usage_collection.insert_one(usage_record)
#             except Exception as e:      #pylint: disable = broad-exception-caught
#                 app_logger.error(f"Error saving token usage for stream: {str(e)}")

#             # Save to chat history
#             # self._save_to_chat_history(user_id, stream_id, query, response)

#             # End the trace with successful output
#             if trace_id:
#                 langfuse_service.end_trace(trace_id, output=response)

#             # Update LangSmith run
#             langsmith_service.update_run(run_id, outputs=response)

#             # Clean up the stream record
#             with stream_lock:
#                 if stream_id in active_streams:
#                     del active_streams[stream_id]

#         except Exception as e:          #pylint: disable = broad-exception-caught
#             app_logger.error(f"Error generating streaming chat response: {str(e)}")

#             # End the trace with error
#             if trace_id:
#                 langfuse_service.end_trace(trace_id, error=str(e))

#             if run_id:
#                 langsmith_service.update_run(run_id, outputs={}, error=str(e))

#             yield f"I'm sorry, I encountered an error while processing your request: {str(e)}"


#     def get_token_usage_stats(self) -> Dict[str, Any]:
#         """
#         Get statistics about token usage across all requests.

#         Returns:
#             Dictionary with token usage statistics
#         """
#         try:
#             # Calculate averages
#             avg_input = 0
#             avg_output = 0
#             avg_total = 0

#             if self.total_token_usage["request_count"] > 0:
#                 avg_input = self.total_token_usage["input_tokens"] / self.total_token_usage["request_count"]        #pylint: disable=line-too-long
#                 avg_output = self.total_token_usage["output_tokens"] / self.total_token_usage["request_count"]      #pylint: disable=line-too-long
#                 avg_total = self.total_token_usage["total_tokens"] / self.total_token_usage["request_count"]        #pylint: disable=line-too-long

#             # Get stats from database for the last 24 hours
#             yesterday = datetime.utcnow() - datetime.timedelta(days=1)

#             daily_usage = self.token_usage_collection.aggregate([
#                 {"$match": {"timestamp": {"$gte": yesterday}}},
#                 {"$group": {
#                     "_id": None,
#                     "input_tokens": {"$sum": "$input_tokens"},
#                     "output_tokens": {"$sum": "$output_tokens"},
#                     "total_tokens": {"$sum": "$total_tokens"},
#                     "count": {"$sum": 1}
#                 }}
#             ])

#             daily_stats = next(daily_usage, {
#                 "input_tokens": 0,
#                 "output_tokens": 0,
#                 "total_tokens": 0,
#                 "count": 0
#             })

#             # Remove _id field if it exists
#             if "_id" in daily_stats:
#                 del daily_stats["_id"]

#             return {
#                 "all_time": {
#                     "input_tokens": self.total_token_usage["input_tokens"],
#                     "output_tokens": self.total_token_usage["output_tokens"],
#                     "total_tokens": self.total_token_usage["total_tokens"],
#                     "request_count": self.total_token_usage["request_count"],
#                     "avg_input_per_request": round(avg_input, 2),
#                     "avg_output_per_request": round(avg_output, 2),
#                     "avg_total_per_request": round(avg_total, 2)
#                 },
#                 "last_24_hours": daily_stats
#             }

#         except Exception as e:           #pylint: disable=broad-exception-caught
#             app_logger.error(f"Error getting token usage stats: {str(e)}")
#             return {
#                 "error": str(e)
#             }


# # Create a global instance
# langchain_service = LangChainService()


# ===================================================


"""
Service for integrating with LangChain and processing queries.
""" #pylint: disable=too-many-lines

from typing import Dict, List, Any, AsyncIterator
from datetime import datetime
import json
import uuid
import asyncio
import threading
import tiktoken

# Import existing modules
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain           #pylint: disable = no-name-in-module
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langfuse.callback import CallbackHandler   #pylint: disable = import-error

from app.api.core.config import settings
from app.api.core.logging import app_logger
from app.api.services.elastic_search_service import es_service
from app.api.services.ticket_service import ticket_service
from app.api.db.mongodb import mongodb_client
from app.api.services.langsmith_service import langsmith_service
from app.api.services.verification_service import verification_service
from app.api.services.langfuse_service import langfuse_service

# Create a global dictionary to track active streaming sessions
active_streams = {}
stream_lock = threading.Lock()

# Encoder for token counting
class TokenCounter:     #pylint:disable=too-few-public-methods
    """Class to count tokens for OpenAI models"""

    def __init__(self, model_name=settings.OPENAI_MODEL):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding(settings.TIKTOKEN_MODEL_BASE_FOR_TOKENS)
        app_logger.info(f"Initialized TokenCounter with encoding for {model_name}")

    def count_tokens(self, text):
        """Count the number of tokens in a text string"""
        if not text:
            return 0
        token_ids = self.encoding.encode(text)
        token_count = len(token_ids)
        return token_count

#pylint: disable = abstract-method
class TokenTrackingCallbackHandler(BaseCallbackHandler):        #pylint: disable=too-many-instance-attributes
    """Custom callback handler for streaming responses with token tracking."""

    def __init__(self, queue, stream_id, token_counter):
        super().__init__()
        self._queue = queue
        self._stop_signal = "DONE"
        self._stream_id = stream_id
        self._is_interrupted = False
        self._token_counter = token_counter
        self._input_tokens = 0
        self._output_tokens = 0
        self._output_text = ""
        app_logger.info(f"Initialized TokenTracking Callback Handler for stream {stream_id}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running. Count input tokens."""
        self._input_tokens = sum(self._token_counter.count_tokens(p) for p in prompts)
        app_logger.info(f"""[Token Tracker] Stream
                        {self._stream_id} -
                        Input tokens: {self._input_tokens}""")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when the LLM produces a new token."""
        # Check if stream has been interrupted
        with stream_lock:
            if active_streams.get(self._stream_id, {}).get("interrupted", False):
                self._is_interrupted = True
                self._queue.put("[INTERRUPTED]")
                return

        # Continue with normal token streaming if not interrupted
        self._output_text += token
        self._output_tokens += 1

        # Log every 10 tokens
        if self._output_tokens > 0:
            app_logger.debug(f"""[Token Tracker]
                             Stream {self._stream_id} -
                             Output tokens so far:
                             {self._output_tokens}""")

        self._queue.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        # Log final token counts
        app_logger.info(f"[Token Tracker] Stream {self._stream_id} - Final counts:")
        app_logger.info(f"[Token Tracker] Input tokens: {self._input_tokens}")
        app_logger.info(f"[Token Tracker] Output tokens: {self._output_tokens}")
        app_logger.info(f"[Token Tracker] Total tokens: {self._input_tokens + self._output_tokens}")

        # Check output token count using tiktoken for verification
        actual_output_tokens = self._token_counter.count_tokens(self._output_text)
        app_logger.info(f"""[Token Tracker]
                        Verified output tokens (tiktoken):
                        {actual_output_tokens}""")

        # If we were interrupted, add a special token to indicate interruption
        if self._is_interrupted:
            self._queue.put("[INTERRUPTED]")

        # Always send the stop signal
        self._queue.put(self._stop_signal)

        # Clean up the stream record
        with stream_lock:
            if self._stream_id in active_streams:
                active_streams[self._stream_id]["token_usage"] = {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                    "total_tokens": self._input_tokens + self._output_tokens
                }
                del active_streams[self._stream_id]

#pylint: disable = abstract-method
class AsyncTokenTrackingCallbackHandler(BaseCallbackHandler):
    """Async callback handler for streaming responses with token tracking."""

    def __init__(self, token_counter):
        super().__init__()
        self._token_counter = token_counter
        self._input_tokens = 0
        self._output_tokens = 0
        self._output_text = ""
        self._is_interrupted = False
        self._stream_id = None
        self.tokens = []  # Store tokens here for async access
        app_logger.info("Initialized AsyncTokenTracking Callback Handler")

    def set_stream_id(self, stream_id):
        """Set the stream ID for this handler."""
        self._stream_id = stream_id

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running. Count input tokens."""
        self._input_tokens = sum(self._token_counter.count_tokens(p) for p in prompts)
        app_logger.info(f"""[Token Tracker] Stream
                        {self._stream_id} - Input tokens: {self._input_tokens}""")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when the LLM produces a new token."""
        # Check if stream has been interrupted
        with stream_lock:
            if active_streams.get(self._stream_id, {}).get("interrupted", False):
                self._is_interrupted = True
                self.tokens.append("[INTERRUPTED]")
                return

        # Continue with normal token streaming if not interrupted
        self._output_text += token
        self._output_tokens += 1
        self.tokens.append(token)

        # Log every 10 tokens
        if self._output_tokens % 10 == 0:
            app_logger.debug(f"""[Token Tracker] Stream
                             {self._stream_id} - Output tokens so far: {self._output_tokens}""")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        # Log final token counts
        app_logger.info(f"[Token Tracker] Stream {self._stream_id} - Final counts:")
        app_logger.info(f"[Token Tracker] Input tokens: {self._input_tokens}")
        app_logger.info(f"[Token Tracker] Output tokens: {self._output_tokens}")
        app_logger.info(f"[Token Tracker] Total tokens: {self._input_tokens + self._output_tokens}")

        # Check output token count using tiktoken for verification
        actual_output_tokens = self._token_counter.count_tokens(self._output_text)
        app_logger.info(f"""[Token Tracker] Verified
                        output tokens (tiktoken): {actual_output_tokens}""")

        # If we were interrupted, add a special token to indicate interruption
        if self._is_interrupted:
            self.tokens.append("[INTERRUPTED]")

        # Add a special token to indicate completion
        self.tokens.append("[DONE]")

        # Update the stream record
        with stream_lock:
            if self._stream_id in active_streams:
                active_streams[self._stream_id]["token_usage"] = {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                    "total_tokens": self._input_tokens + self._output_tokens
                }
completed_responses = {}

class LangChainService:             #pylint: disable=too-many-instance-attributes
    """Service for LangChain integration and query processing with token tracking."""

    def __init__(self):
        """Initialize LangChain components."""
        # Initialize token counter
        self.token_counter = TokenCounter(settings.OPENAI_MODEL)


        # Track total token usage
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "request_count": 0
        }

        self.langfuse_callbacks = [CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )] if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY else []

        # General initialization
        # self.stream_queues = {}  # Changed to dict to track multiple streams
        self.top_k = settings.top_k

        # Initialize LLMs
        self.llm = ChatOpenAI(
            temperature=0.0,
            model_name=settings.OPENAI_MODEL,
            callbacks=self.langfuse_callbacks
        )

        # Initialize streaming LLM
        self.streaming_llm = ChatOpenAI(
            temperature=0.4,
            model_name=settings.OPENAI_MODEL,
            streaming=True,
            callbacks = self.langfuse_callbacks
        )

        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

        # Initialize chat history collection
        self.chat_history_collection = mongodb_client.chat_history_db.chat_history

        # New collection for token usage statistics
        self.token_usage_collection = mongodb_client.chat_history_db.token_usage

        # Set up the tools
        self.tools = self._create_tools()

        # Set up the agent
        self.memory = ConversationBufferMemory(memory_key="chat_history")


        agent_prefix = """You are a helpful assistant.
        You can answer questions using the available tools.
        For creating tickets, you'll need to collect the user's
        email and phone number, and verify both.

            Follow these steps for ticket creation:
            1. Start with user_id and query
            2. Ask for email
            3. Send verification code to email
            4. Collect and verify the email code
            5. Ask for phone number
            6. Send verification code to phone
            7. Collect and verify the phone code
            8. Create the ticket
            Lead the user through this process patiently and clearly.
        """

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=settings.DEBUG,
            agent_kwargs={"prefix": agent_prefix}

        )

        self.chat_prompt = PromptTemplate(
            input_variables=["query", "search_results"],
            template="""
            You are a helpful assistant. Based on the following search results, provide a concise,
            natural-sounding response to the user's query. If the search results don't contain
            relevant information, suggest what the user might do next.

            User Query: {query}

            Search Results:
            {search_results}

            Response:
            """
        )

        # Regular chat chain
        self.chat_chain = LLMChain(
            llm=ChatOpenAI(temperature=0.4,
                           model_name=settings.OPENAI_MODEL,
                           callbacks=self.langfuse_callbacks),
            prompt=self.chat_prompt
        )

        # Streaming chat chain
        self.streaming_chat_chain = LLMChain(
            llm=self.streaming_llm,
            prompt=self.chat_prompt
        )


        app_logger.info("Initialized LangChain Service with token tracking")

    #pylint: disable=too-many-arguments
    #pylint: disable=too-many-positional-arguments
    def _track_token_usage(self,
                           input_text,
                           output_text,
                           source,
                           user_id=None,
                           query=None):
        """
        Track token usage for a request.

        Args:
            input_text: The input text sent to the model
            output_text: The output text generated by the model
            source: Source of the request (e.g., "chat", "agent", "search")
            user_id: Optional user ID
            query: Optional query text
        """
        input_tokens = self.token_counter.count_tokens(input_text)
        output_tokens = self.token_counter.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens

        # Log the token usage
        app_logger.info(f"""[Token Tracker]
                        {source.upper()} -
                        Input tokens at line 1246:
                        {input_tokens}""")
        app_logger.info(f"""[Token Tracker]
                        {source.upper()} -
                        Output tokens at line 1247:
                        {output_tokens}""")
        app_logger.info(f"""[Token Tracker]
                        {source.upper()} -
                        Total tokens at line 1248:
                        {total_tokens}""")

        # Update global counters
        self.total_token_usage["input_tokens"] += input_tokens
        self.total_token_usage["output_tokens"] += output_tokens
        self.total_token_usage["total_tokens"] += total_tokens
        self.total_token_usage["request_count"] += 1

        # Save to database
        usage_record = {
            "timestamp": datetime.utcnow(),
            "source": source,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "user_id": user_id,
            "query": query
        }

        try:
            self.token_usage_collection.insert_one(usage_record)
        except Exception as e:          #pylint: disable=broad-exception-caught
            app_logger.error(f"Error saving token usage: {str(e)}")

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for different components."""
        # Tool for retrieving information from Elasticsearch
        get_info_tool = Tool(
            name="GetInformation",
            func=self._get_information,
            description="""Useful for retrieving information based on the user's query.
                            Input should be a user question."""
        )

        # Tool for creating tickets
        create_ticket_tool = Tool(
            name="RaiseTicket",
            func=self._raise_ticket,
            description="""Useful for raising a ticket when information cannot be found.
                        Input should be a JSON string with these possible fields:
                        'user_id' (required): The ID of the user
                        'query' (required): The user's original question or request
                        'additional_info' (optional): Any additional context
                        'email' (optional): User's email address if in the ticket creation flow
                        'phone_number' (optional): User's phone number if in the ticket creation flow
                        'otp' (optional): OTP code provided by user during verification"""
            )

        # Tool for retrieving tickets
        get_ticket_tool = Tool(
            name="GetTicket",
            func=self._get_ticket,
            description="""Useful for retrieving information about a specific ticket.
                            Input should be a ticket ID."""
        )

        return [get_info_tool, create_ticket_tool, get_ticket_tool]

    def _get_information(self, query: str) -> str:
        """Get information from Elasticsearch with token tracking."""
        app_logger.info(f"Getting information for query: {query}")
        input_text = query

        # Perform search using the Elasticsearch service
        results = es_service.hybrid_search(query, top_k=self.top_k)

        if not results:
            output_text = "Information not found. Consider raising a ticket for this query."
            self._track_token_usage(input_text, output_text, "elasticsearch_search")
            return output_text

        # Format results as a single string
        response_parts = ["Here's what I found:"]

        for i, result in enumerate(results, 1):
            response_parts.append(f"\n\n{i}. {result['question']}")
            response_parts.append(f"   Answer: {result['answer']}")
            response_parts.append(f"   Confidence: {result['similarity_score']:.2f}")

        output_text = "\n".join(response_parts)
        self._track_token_usage(input_text, output_text, "elasticsearch_search")
        return output_text

#pylint: disable=too-many-return-statements
    def _raise_ticket(self, input_str: str) -> str:
        """Raise a ticket in the system with token tracking."""
        try:
            # Parse input as JSON
            data = json.loads(input_str)

            user_id = data.get("user_id")
            query = data.get("query")
            email = data.get("email")
            phone_number = data.get("phone_number")
            additional_info = data.get("additional_info", "")

            if not user_id or not query:
                output_text = "Error: Missing required fields (user_id, query)"
                self._track_token_usage(input_str, output_text, "raise_ticket")
                return output_text

            # Interactive collection and verification of email if not provided
            if not email:
                output_text = "Please provide your email address for ticket creation."
                self._track_token_usage(input_str, output_text, "raise_ticket")
                return output_text

            # Attempt to verify the email with OTP
            try:
                verification_service.send_otp(email, "email")
                output_text = f"""An OTP has been sent
                to your email {email}.
                Please provide the OTP to verify."""
                self._track_token_usage(input_str, output_text, "raise_ticket")
                return output_text

            except Exception as e:              #pylint: disable=broad-exception-caught
                output_text = f"""Error sending email verification:
                {str(e)}.
                Please provide a valid email address."""
                self._track_token_usage(input_str, output_text, "raise_ticket")
                # return output_text

            # Interactive collection and verification of phone number if not provided
            if not phone_number:
                output_text = "Please provide your phone number for ticket creation."
                self._track_token_usage(input_str, output_text, "raise_ticket")
                return output_text

            # Attempt to verify the phone number with OTP
            try:
                verification_service.send_otp(phone_number, "phone")
                output_text = f"""An OTP has been sent to your phone {phone_number}.
                Please provide the OTP to verify."""
                self._track_token_usage(input_str, output_text, "raise_ticket")
                return output_text
            except Exception as e:          #pylint: disable=broad-exception-caught
                output_text = f"""Error sending phone verification: {str(e)}.
                Please provide a valid phone number."""
                self._track_token_usage(input_str, output_text, "raise_ticket")
                # return output_text

            # Create ticket using the ticket service
            ticket = ticket_service.create_ticket(user_id,
                                                  query,
                                                  email,
                                                  phone_number,
                                                  additional_info)

            if ticket:
                output_text = f"Ticket created successfully with ID: {ticket['ticket_id']}"
                self._track_token_usage(input_str, output_text, "raise_ticket", user_id, query)
                return output_text

            output_text = "Error: Failed to create ticket"
            self._track_token_usage(input_str, output_text, "raise_ticket", user_id, query)
            return output_text

        except Exception as e:          #pylint: disable=broad-exception-caught
            app_logger.error(f"Error in raise_ticket: {str(e)}")
            output_text = f"Error: {str(e)}"
            self._track_token_usage(input_str, output_text, "raise_ticket")
            return output_text

    def _get_ticket(self, ticket_id: str) -> str:
        """Get ticket information with token tracking."""
        input_text = f"Get ticket: {ticket_id}"

        try:
            ticket = ticket_service.get_ticket(ticket_id)

            if not ticket:
                output_text = f"Ticket with ID {ticket_id} not found"
                self._track_token_usage(input_text, output_text, "get_ticket")
                return output_text

            # Format ticket information
            created_at = ticket["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            updated_at = ticket.get("updated_at")
            updated_at = updated_at.strftime("%Y-%m-%d %H:%M:%S") if updated_at else "N/A"

            output_text = (
                f"Ticket ID: {ticket['ticket_id']}\n"
                f"Status: {ticket['status']}\n"
                f"User ID: {ticket['user_id']}\n"
                f"Query: {ticket['query']}\n"
                f"Additional Info: {ticket['additional_info'] or 'N/A'}\n"
                f"Created: {created_at}\n"
                f"Last Updated: {updated_at}"
            )

            self._track_token_usage(input_text,
                                    output_text,
                                    "get_ticket",
                                    ticket['user_id'],
                                    ticket['query'])
            return output_text

        except Exception as e:          #pylint: disable=broad-exception-caught
            app_logger.error(f"Error in get_ticket: {str(e)}")
            output_text = f"Error retrieving ticket: {str(e)}"
            self._track_token_usage(input_text, output_text, "get_ticket")
            return output_text

    def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a user query through the LangChain agent with token tracking."""
        try:
            app_logger.info(f"Processing query for user {user_id}: {query}")

            # Create a Langfuse trace for this query
            trace = langfuse_service.create_trace(
                name="process_query",
                user_id=user_id,
                metadata={"query": query}
            )
            trace_id = trace.id if trace else None

            # Get information directly first
            search_span = None
            if trace_id:
                search_span = langfuse_service.create_span(
                    trace_id=trace_id,
                    name="elasticsearch_search",
                    metadata={"query": query}
                )

            search_results = es_service.hybrid_search(query, top_k=self.top_k)
            found = len(search_results) > 0

            if search_span:
                search_span.end()

            # Prepare response
            response = {
                "query": query,
                "results": search_results,
                "found": found,
                "agent_response": None
            }

            # If not found, let the agent decide next steps
            if not found:
                app_logger.info("No results found, processing with agent")
                agent_input = f"""User {user_id} asked: {query}.
                            No information was found. Should I raise a ticket?"""

                agent_span = None
                if trace_id:
                    agent_span = langfuse_service.create_span(
                        trace_id=trace_id,
                        name="agent_run",
                        metadata={"input": agent_input}
                    )

                agent_response = self.agent.run(input=agent_input)
                response["agent_response"] = agent_response

                if agent_span:
                    agent_span.end()

                # Track token usage for agent
                token_usage = self._track_token_usage(
                    agent_input,
                    agent_response,
                    "agent",
                    user_id,
                    query
                )

                # Record the generation in Langfuse
                if trace_id:
                    langfuse_service.create_generation(
                        trace_id=trace_id,
                        name="agent_generation",
                        model=settings.OPENAI_MODEL,
                        prompt=agent_input,
                        completion=agent_response,
                        token_usage=token_usage
                    )

            # Store in chat history
            # self._save_to_chat_history(user_id, query, response)

            # End the trace
            if trace_id:
                langfuse_service.end_trace(trace_id, output=response)

            # Create LangSmith run
            run_id = langsmith_service.create_run(name="process_query",
                                                inputs={"user_id": user_id, "query": query})
            langsmith_service.update_run(run_id, outputs=response)

            return response

        except Exception as e:      #pylint: disable=broad-exception-caught
            app_logger.error(f"Error processing query: {str(e)}")

            # End the trace with error
            if trace_id:
                langfuse_service.end_trace(trace_id, error=str(e))

            # Update LangSmith run with error
            langsmith_service.update_run(run_id, outputs={}, error=str(e))

            return {
                "query": query,
                "results": [],
                "found": False,
                "error": str(e)
            }

    #pylint: disable=too-many-locals
    def generate_chat_response(self, user_id: str, query: str, session_id: str) -> Dict[str, Any]:
        """Generate a conversational response with token tracking."""
        trace_id = None
        run_id = None
        try:
            app_logger.info(f"Generating chat response for user {user_id}: {query}")

            # Create a Langfuse trace for this query
            trace = langfuse_service.create_trace(
                name="generate_chat_response",
                user_id=user_id,
                metadata={"query": query, "session_id": session_id}
            )
            trace_id = trace.id if trace else None

            # Create LangSmith run
            run_id = langsmith_service.create_run(name="generate_chat_response",
                                                inputs={"user_id": user_id,
                                                        "query": query,
                                                        "session_id": session_id})

            # Get search results
            search_span = None
            if trace_id:
                search_span = langfuse_service.create_span(
                    trace_id=trace_id,
                    name="elasticsearch_search",
                    metadata={"query": query}
                )

            search_results = es_service.hybrid_search(query, top_k=self.top_k)
            found = len(search_results) > 0

            if search_span:
                search_span.end()

            # Format search results for the prompt
            formatted_results = ""
            if found:
                for i, result in enumerate(search_results, 1):
                    formatted_results += f"{i}. Question: {result['question']}\n"
                    formatted_results += f"   Answer: {result['answer']}\n"
                    formatted_results += f"   Category: {result.get('category', 'General')}\n"
                    formatted_results += f"   Confidence: {result['similarity_score']:.2f}\n\n"
            else:
                formatted_results = "No relevant information found."

            # Prepare the prompt input
            prompt_input = {
                "query": query,
                "search_results": formatted_results
            }

            # Convert prompt input to string for token counting
            prompt_str = self.chat_prompt.format(**prompt_input)

            # Generate a natural language response using the LLM chain
            llm_span = None
            if trace_id:
                llm_span = langfuse_service.create_span(
                    trace_id=trace_id,
                    name="llm_chain",
                    metadata={"prompt": prompt_str}
                )

            llm_response = self.chat_chain.run(**prompt_input)

            if llm_span:
                llm_span.end()

            # Track token usage
            token_usage = self._track_token_usage(
                prompt_str,
                llm_response,
                "chat_response",
                user_id,
                query
            )

            # Record the generation in Langfuse
            if trace_id:
                langfuse_service.create_generation(
                    trace_id=trace_id,
                    name="chat_generation",
                    model=settings.OPENAI_MODEL,
                    prompt=prompt_str,
                    completion=llm_response,
                    token_usage=token_usage
                )

            # Prepare the complete response
            response = {
                "query": query,
                "results": search_results,
                "found": found,
                "chat_response": llm_response.strip(),
                "token_usage": token_usage,
                "session_id": session_id
            }

            # Save to chat history
            # self._save_to_chat_history(user_id, session_id, query, response)

            # End the trace
            if trace_id:
                langfuse_service.end_trace(trace_id, output=response)

            # Update LangSmith run
            langsmith_service.update_run(run_id, outputs=response)

            return response

        except Exception as e:      #pylint: disable=broad-exception-caught
            app_logger.error(f"Error generating chat response: {str(e)}")

            # End the trace with error
            if trace_id:
                langfuse_service.end_trace(trace_id, error=str(e))

            # Update LangSmith run with error
            langsmith_service.update_run(run_id, outputs={}, error=str(e))

            return {
                "query": query,
                "results": [],
                "found": False,
                "chat_response": "I'm sorry, I encountered an error while processing your request.",
                "error": str(e)
            }


    def interrupt_stream(self, stream_id: str) -> bool:
        """
        Interrupt an ongoing streaming response.

        Args:
            stream_id: The ID of the stream to interrupt

        Returns:
            bool: True if the stream was successfully interrupted, False otherwise
        """
        with stream_lock:
            if stream_id in active_streams:
                active_streams[stream_id]["interrupted"] = True
                app_logger.info(f"Stream {stream_id} marked for interruption")
                return True

            app_logger.warning(f"Stream {stream_id} not found for interruption")
            return False

    async def _cleanup_completed_response(self, stream_id: str, delay: int = 300):
        """
        Clean up a completed response after a delay.

        Args:
            stream_id (str): The stream ID to clean up.
            delay (int): The delay in seconds before cleaning up.
        """
        await asyncio.sleep(delay)  # Wait for 5 minutes by default
        if stream_id in completed_responses:
            del completed_responses[stream_id]


    async def get_full_response(self, stream_id: str) -> str:
        """
        Get the full response for a given stream ID.

        Args:
            stream_id (str): The stream ID to retrieve the response for.

        Returns:
            str: The full response text.

        Raises:
            ValueError: If the response for the given stream ID is not found.
        """
        # First, check if the response is already in the completed_responses dict
        if stream_id in completed_responses:
            return completed_responses[stream_id]

        # If not, wait for a short time to see if it becomes available
        # This is needed because the response might still be processing
        for _ in range(10):  # Try for up to 1 second
            await asyncio.sleep(0.1)
            if stream_id in completed_responses:
                return completed_responses[stream_id]

        # If still not found, raise an error
        raise ValueError(f"Response for stream_id {stream_id} not found")

    #pylint: disable=too-many-branches
    #pylint: disable=too-many-statements
    #pylint: disable=too-many-locals
    async def generate_streaming_chat_response(
    self, user_id: str, query: str, session_id: str, stream_id: str = None
) -> AsyncIterator[str]:
        """
        Generate a streaming conversational response with token tracking.
        """
        run_id = None
        trace_id = None
        try:
            # Create a Langfuse trace for this streaming query
            trace = langfuse_service.create_trace(
                name="streaming_chat_response",
                user_id=user_id,
                metadata={"query": query, "session_id": session_id}
            )
            trace_id = trace.id if trace else None

            # Create LangSmith run
            run_id = langsmith_service.create_run(
                name="generate_streaming_chat_response",
                inputs={"user_id": user_id, "query": query, "session_id": session_id}
            )

            # Generate a stream ID if not provided
            if not stream_id:
                stream_id = str(uuid.uuid4())

            app_logger.info(f"""Generating streaming chat
                            response for user {user_id}, stream {stream_id}: {query}""")


            # Register this stream in the active streams dictionary
            with stream_lock:
                active_streams[stream_id] = {
                    "user_id": user_id,
                    "query": query,
                    "interrupted": False,
                    "start_time": datetime.utcnow(),
                    "token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0
                    }
                }

            # Create a callback handler for token tracking
            async_handler = AsyncTokenTrackingCallbackHandler(self.token_counter)
            async_handler.set_stream_id(stream_id)

            # Get search results
            search_span = None
            if trace_id:
                search_span = langfuse_service.create_span(
                    trace_id=trace_id,
                    name="elasticsearch_search",
                    metadata={"query": query}
                )

            search_results = es_service.hybrid_search(query, top_k=self.top_k)
            # search_results, token_usage = self._get_information(query)
            found = len(search_results) > 0

            if search_span:
                search_span.end()

            # Format search results for the prompt
            formatted_results = ""
            if found:
                for i, result in enumerate(search_results, 1):
                    formatted_results += f"{i}. Question: {result['question']}\n"
                    formatted_results += f"   Answer: {result['answer']}\n"
                    formatted_results += f"   Category: {result.get('category', 'General')}\n"
                    formatted_results += f"   Confidence: {result['similarity_score']:.2f}\n\n"
            else:
                formatted_results = "No relevant information found."

            # Create properly configured streaming LLM instance
            streaming_llm = ChatOpenAI(
                temperature=0.4,
                model_name=settings.OPENAI_MODEL,
                streaming=True,
                callbacks=[async_handler]
            )

            # Create streaming chat chain with the handler
            streaming_chat_chain = LLMChain(
                llm=streaming_llm,
                prompt=self.chat_prompt
            )

            # Calculate tokens in the prompt template
            prompt_str = self.chat_prompt.format(
                query=query,
                search_results=formatted_results
            )
            prompt_tokens = self.token_counter.count_tokens(prompt_str)
            app_logger.info(f"[Token Tracker] Stream {stream_id} - Prompt tokens: {prompt_tokens}")

            # Create a streaming span in Langfuse
            streaming_span = None
            if trace_id:
                streaming_span = langfuse_service.create_span(
                    trace_id=trace_id,
                    name="streaming_llm",
                    metadata={"prompt": prompt_str}
                )

            # Start a task to run the LLM chain
            task = asyncio.create_task(
                streaming_chat_chain.ainvoke(
                    {
                        "query": query,
                        "search_results": formatted_results
                    }
                )
            )

            full_response = ""
            was_interrupted = False
            output_token_count = 0

            # Keep checking for new tokens until we're done
            last_idx = 0
            while True:
                # Check for new tokens
                if len(async_handler.tokens) > last_idx:
                    for i in range(last_idx, len(async_handler.tokens)):
                        token = async_handler.tokens[i]

                        # Check for special tokens
                        #pylint: disable = no-else-break
                        if token == "[DONE]":
                            # We're finished
                            break
                        #pylint: disable = no-else-break
                        elif token == "[INTERRUPTED]":      #pylint: disable = no-else-break
                            was_interrupted = True
                            yield "[INTERRUPTED]"
                            break
                        #pylint: disable = no-else-break
                        else:                               #pylint: disable = no-else-break
                            # Normal token
                            full_response += token
                            output_token_count += 1
                            yield token

                    # Update our position
                    last_idx = len(async_handler.tokens)

                # Check if we're done
                if "[DONE]" in async_handler.tokens or "[INTERRUPTED]" in async_handler.tokens:
                    break

                # Pause briefly to avoid busy-waiting
                await asyncio.sleep(0.01)

            # Wait for the task to complete
            await task

            # End the streaming span
            if streaming_span:
                streaming_span.end()

            # Get the final token count
            actual_input_tokens = self.token_counter.count_tokens(prompt_str)
            actual_output_tokens = self.token_counter.count_tokens(full_response)

            app_logger.info(f"""[Token Tracker] Stream
                            {stream_id} - Finalized token counts:""")
            app_logger.info(f"""[Token Tracker] Input
                            tokens (verified): {actual_input_tokens}""")
            app_logger.info(f"""[Token Tracker] Output tokens
                            (verified): {actual_output_tokens}""")
            app_logger.info(f"""[Token Tracker] Total tokens:
                            {actual_input_tokens + actual_output_tokens}""")

            # Prepare the complete response for saving to history
            token_usage = {
                "input_tokens": actual_input_tokens,
                "output_tokens": actual_output_tokens,
                "total_tokens": actual_input_tokens + actual_output_tokens
            }

            # Record the generation in Langfuse
            if trace_id:
                langfuse_service.create_generation(
                    trace_id=trace_id,
                    name="streaming_generation",
                    model=settings.OPENAI_MODEL,
                    prompt=prompt_str,
                    completion=full_response,
                    token_usage=token_usage
                )

            # Update global counters
            self.total_token_usage["input_tokens"] += actual_input_tokens
            self.total_token_usage["output_tokens"] += actual_output_tokens
            self.total_token_usage["total_tokens"] += actual_input_tokens + actual_output_tokens
            self.total_token_usage["request_count"] += 1


            response = {
                "query": query,
                "results": search_results,
                "found": found,
                "chat_response": full_response.strip(),
                "was_interrupted": was_interrupted,
                "stream_id": stream_id,
                "token_usage": token_usage,
                "session_id": session_id
            }


            # Save token usage to database
            usage_record = {
                "timestamp": datetime.utcnow(),
                "source": "streaming_chat",
                "input_tokens": actual_input_tokens,
                "output_tokens": actual_output_tokens,
                "total_tokens": actual_input_tokens + actual_output_tokens,
                "user_id": user_id,
                "query": query,
                "was_interrupted": was_interrupted,
                "stream_id": stream_id
            }

            try:
                self.token_usage_collection.insert_one(usage_record)
            except Exception as e:      #pylint: disable = broad-exception-caught
                app_logger.error(f"Error saving token usage for stream: {str(e)}")

            # Save to chat history
            # self._save_to_chat_history(user_id, stream_id, query, response)

            # End the trace with successful output
            if trace_id:
                langfuse_service.end_trace(trace_id, output=response)

            # Update LangSmith run
            langsmith_service.update_run(run_id, outputs=response)

            # Store the full response for later retrieval
            completed_responses[stream_id] = full_response.strip()

            # Set up a task to remove the response after a while to prevent memory leaks
            asyncio.create_task(self._cleanup_completed_response(stream_id))


            # Clean up the stream record
            with stream_lock:
                if stream_id in active_streams:
                    del active_streams[stream_id]



        except Exception as e:          #pylint: disable = broad-exception-caught
            app_logger.error(f"Error generating streaming chat response: {str(e)}")

            # End the trace with error
            if trace_id:
                langfuse_service.end_trace(trace_id, error=str(e))

            if run_id:
                langsmith_service.update_run(run_id, outputs={}, error=str(e))

            yield f"I'm sorry, I encountered an error while processing your request: {str(e)}"


    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about token usage across all requests.

        Returns:
            Dictionary with token usage statistics
        """
        try:
            # Calculate averages
            avg_input = 0
            avg_output = 0
            avg_total = 0

            if self.total_token_usage["request_count"] > 0:
                avg_input = self.total_token_usage["input_tokens"] / self.total_token_usage["request_count"]        #pylint: disable=line-too-long
                avg_output = self.total_token_usage["output_tokens"] / self.total_token_usage["request_count"]      #pylint: disable=line-too-long
                avg_total = self.total_token_usage["total_tokens"] / self.total_token_usage["request_count"]        #pylint: disable=line-too-long

            # Get stats from database for the last 24 hours
            yesterday = datetime.utcnow() - datetime.timedelta(days=1)

            daily_usage = self.token_usage_collection.aggregate([
                {"$match": {"timestamp": {"$gte": yesterday}}},
                {"$group": {
                    "_id": None,
                    "input_tokens": {"$sum": "$input_tokens"},
                    "output_tokens": {"$sum": "$output_tokens"},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "count": {"$sum": 1}
                }}
            ])

            daily_stats = next(daily_usage, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "count": 0
            })

            # Remove _id field if it exists
            if "_id" in daily_stats:
                del daily_stats["_id"]

            return {
                "all_time": {
                    "input_tokens": self.total_token_usage["input_tokens"],
                    "output_tokens": self.total_token_usage["output_tokens"],
                    "total_tokens": self.total_token_usage["total_tokens"],
                    "request_count": self.total_token_usage["request_count"],
                    "avg_input_per_request": round(avg_input, 2),
                    "avg_output_per_request": round(avg_output, 2),
                    "avg_total_per_request": round(avg_total, 2)
                },
                "last_24_hours": daily_stats
            }

        except Exception as e:           #pylint: disable=broad-exception-caught
            app_logger.error(f"Error getting token usage stats: {str(e)}")
            return {
                "error": str(e)
            }

# Create a global instance
langchain_service = LangChainService()
