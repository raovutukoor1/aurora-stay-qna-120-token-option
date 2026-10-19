# Code cell 041

# Import necessary libraries
from flask import Flask, request, jsonify
import os
import truststore
from werkzeug.exceptions import HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import RAG utility functions
from rag_utils import retrieve, rag


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "HarishMaths/Hotel-Policy-Embedding")
VECTORSTORE_PATH = "faiss_index"


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Request validation and JSON error responses
# ---------------------------------------------------------

def _json_object():
    """Return a JSON object or raise a client-friendly validation error."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    return data


def _query_and_k(data):
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    k = data.get("k", 5)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k must be an integer from 1 to 10.")
    return query.strip(), k


def _generation_parameters(data):
    query, k = _query_and_k(data)

    model_name = data.get("model_name", "gpt-4o-mini")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
        raise ValueError("temperature must be a number from 0 to 1.")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1:
        raise ValueError("top_p must be a number from 0 to 1.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 64 <= max_tokens <= 2048:
        raise ValueError("max_tokens must be an integer from 64 to 2048.")

    return query, k, model_name.strip(), float(temperature), float(top_p), max_tokens


@rag_api.errorhandler(ValueError)
def handle_validation_error(error):
    return jsonify({
        "error": "invalid_request",
        "message": str(error),
    }), 400


@rag_api.errorhandler(HTTPException)
def handle_http_error(error):
    return jsonify({
        "error": error.name.lower().replace(" ", "_"),
        "message": error.description,
    }), error.code


@rag_api.errorhandler(Exception)
def handle_unexpected_error(error):
    rag_api.logger.exception("Unhandled inference error: %s", error)
    return jsonify({
        "error": "internal_server_error",
        "message": "The inference request could not be completed.",
    }), 500


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

# Use the Windows certificate store for Hugging Face downloads in managed or
# corporate environments.
truststore.inject_into_ssl()

try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
except Exception as exc:
    # Local smoke-test fallback when the embedding model is unavailable offline.
    from langchain_core.embeddings import FakeEmbeddings
    print(f"Embedding model unavailable; using 384-dimensional smoke-test fallback: {exc}")
    embeddings = FakeEmbeddings(size=384)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 5
    }

    Returns the top-k relevant chunks retrieved from
    the vector store.
    """

    # Get and validate JSON data from request body
    data = _json_object()
    query, k = _query_and_k(data)

    # Retrieve relevant chunks
    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    # Return the retrieved chunks
    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 5,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns both the generated answer and the relevant
    chunks used to generate the answer.
    """

    # Get and validate JSON data from request body
    data = _json_object()
    query, k, model_name, temperature, top_p, max_tokens = _generation_parameters(data)

    # Run the complete RAG pipeline
    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return answer, chunks, and parameters used
    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 5,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns only the generated answer.
    """

    # Get and validate JSON data from request body
    data = _json_object()
    query, k, model_name, temperature, top_p, max_tokens = _generation_parameters(data)

    # Run the complete RAG pipeline
    generated_answer, _ = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    # Return only the answer
    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(debug=True)
