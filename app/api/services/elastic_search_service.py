"""
Service for interacting with Elasticsearch.
"""
from typing import List, Dict, Any
# from elasticsearch import NotFoundError, ElasticsearchException
# from elasticsearch.exceptions import NotFoundError, ElasticsearchException

# from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchWarning, TransportError  # Use these in v8    # pylint: disable=import-error

from app.api.core.config import settings
from app.api.core.logging import app_logger
from app.api.db.elasticsearch import es_client
from app.api.utils.embedding_utils import embedding_generator



class ElasticsearchService:
    """Service for performing operations on Elasticsearch."""

    def __init__(self):
        """Initialize with Elasticsearch client and embedding generator."""
        self.es = es_client.client
        self.index_name = settings.ES_INDEX_NAME
        self.min_score = settings.ES_MIN_SCORE
        self.top_k = settings.top_k

    def search_by_text(self, query_text: str, top_k) -> List[Dict[str, Any]]:
        """
        Perform a basic text search in Elasticsearch.

        Args:
            query_text: The text to search for
            top_k: Number of top results to return

        Returns:
            List of matching documents
        """
        try:
            app_logger.info(f"Performing text search for: '{query_text}'")

            search_query = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["question", "answer"],
                        "fuzziness": "AUTO"
                    }
                },
                "_source": ["question", "answer", "category"]
            }

            response = self.es.search(
                index=self.index_name,
                body=search_query
            )

            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                # Normalize score to 0-1 range (approximate)
                normalized_score = min(score / 10, 1.0)

                result = {
                    "question": hit["_source"]["question"],
                    "answer": hit["_source"]["answer"],
                    "category": hit["_source"].get("category", "general"),
                    "similarity_score": normalized_score
                }
                results.append(result)

            app_logger.info(f"Text search found {len(results)} results")
            return results

        except TransportError as e:
            print(f"Elasticsearch Transport Error: {e}")
            return []
        except ElasticsearchWarning as e:
            print(f"Elasticsearch Warning: {e}")
            return []

        except Exception as e:                          # pylint: disable=broad-exception-caught
            print(f"General Elasticsearch Error: {e}")
            return []


    def search_by_embedding(self, query_text: str, top_k) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings.

        Args:
            query_text: The text to search for
            top_k: Number of top results to return

        Returns:
            List of matching documents
        """
        try:
            app_logger.info(f"Performing semantic search for: '{query_text}'")

            # Generate embedding for the query
            query_embedding = embedding_generator.generate(query_text)

            if not query_embedding:
                app_logger.warning("Failed to generate embedding for query")
                return []

            # Create search query
            search_query = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {
                                "query_vector": query_embedding
                            }
                        }
                    }
                },
                "_source": ["question", "answer", "category"]
            }

            # Execute search
            response = self.es.search(
                index=self.index_name,
                body=search_query
            )

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                # Calculate normalized score (cosine similarity)
                score = hit["_score"] - 1.0

                # Filter by minimum score
                if score >= self.min_score:
                    result = {
                        "question": hit["_source"]["question"],
                        "answer": hit["_source"]["answer"],
                        "category": hit["_source"].get("category", "general"),
                        "similarity_score": score
                    }
                    results.append(result)

            # Sort by similarity score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            app_logger.info(f"Semantic search found {len(results)} results with score >= {self.min_score}")         # pylint: disable=line-too-long
            return results

        except ElasticsearchException as e:         # pylint: disable=undefined-variable
            app_logger.error(f"Elasticsearch semantic search error: {str(e)}")
            return []

    def hybrid_search(self, query_text: str, top_k) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and text-based search.

        Args:
            query_text: The text to search for
            top_k: Number of top results to return

        Returns:
            List of matching documents
        """
        try:
            app_logger.info(f"Performing hybrid search for: '{query_text}'")

            # Get semantic search results
            semantic_results = self.search_by_embedding(query_text, top_k)

            # If we have enough semantic results, just return those
            if len(semantic_results) >= top_k:
                return semantic_results[:top_k]

            # Otherwise, supplement with text search results
            text_results = self.search_by_text(query_text, top_k)

            # Merge results, prioritizing semantic matches
            seen_questions = {result["question"] for result in semantic_results}

            # Add text results that aren't already in semantic results
            for result in text_results:
                if result["question"] not in seen_questions and len(semantic_results) < top_k:
                    semantic_results.append(result)
                    seen_questions.add(result["question"])

            # Sort by similarity score
            semantic_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            app_logger.info(f"Hybrid search found {len(semantic_results)} results")
            return semantic_results

        except Exception as e:              # pylint: disable=broad-exception-caught
            app_logger.error(f"Hybrid search error: {str(e)}")
            return []


# Create a global instance
es_service = ElasticsearchService()
