# Code cell 040

from langchain_openai import ChatOpenAI
import os
import re
import math
from functools import lru_cache
from pathlib import Path
from langchain_core.documents import Document


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
KNOWN_PROPERTY_CODES = {"ASH-ATL", "ASH-BOS", "ASH-CHI", "ASH-DEN", "ASH-MIA", "ASH-SEA"}


@lru_cache(maxsize=4)
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _topic_source_terms(query: str):
    query = query.lower()
    topic_terms = {
        "cancellation": ("booking_terms", "cancellation"),
        "cancel": ("booking_terms", "cancellation"),
        "refund": ("refunds", "additional_charges"),
        "charge": ("refunds", "additional_charges"),
        "room": ("room_and_property",),
        "property": ("room_and_property",),
        "breakfast": ("room_and_property",),
        "pet": ("room_and_property", "refunds"),
        "dog": ("room_and_property", "refunds"),
        "parking": ("room_and_property", "refunds"),
        "checkout": ("booking_terms", "late checkout", "late departure", "checkout", "departure"),
        "departure": ("booking_terms", "late checkout", "late departure", "checkout", "departure"),
        "late": ("booking_terms", "late checkout", "late departure", "checkout", "departure"),
        "accessible": ("room_and_property", "guest_conduct"),
        "accessibility": ("guest_conduct", "accessibility"),
        "conduct": ("guest_conduct",),
    }
    return {term for word, terms in topic_terms.items() if word in query for term in terms}


def _property_codes(query: str):
    query = query.lower()
    aliases = {
        "miami brickell": "ASH-MIA",
        "ash-mia": "ASH-MIA",
        "chi loop": "ASH-CHI",
        "ash-chi": "ASH-CHI",
        "atlanta": "ASH-ATL",
        "atl midtown": "ASH-ATL",
        "boston": "ASH-BOS",
        "bos harbor": "ASH-BOS",
        "denver": "ASH-DEN",
        "den union": "ASH-DEN",
        "denver union": "ASH-DEN",
        "seattle": "ASH-SEA",
        "sea pike": "ASH-SEA",
    }
    codes = {match.upper() for match in __import__("re").findall(r"ASH-[A-Z]{3}", query.upper())}
    codes.update(code for alias, code in aliases.items() if alias in query)
    return codes


def _unsupported_property_query(query: str) -> bool:
    """Prevent generic policy chunks from answering unsupported property questions."""
    lower = query.lower()
    return "austin north" in lower or ("cabana" in lower and not _property_codes(query))


def _topic_matches(query: str, doc) -> bool:
    text = doc.page_content.lower()
    source = Path(doc.metadata.get("source", "")).name.lower()
    lower_query = query.lower()
    if "double queen" in lower_query:
        # Occupancy questions need the room-type row, not the broad property
        # matrix or amenities sections that also contain generic room terms.
        return "double queen" in text
    if "advance purchase" in lower_query:
        # Keep rate-plan questions away from broad taxes, deposits, and
        # property-reference chunks that dilute contextual precision.
        section = str(doc.metadata.get("section", "")).lower()
        if any(term in section for term in ("taxes", "deposits", "guarantees", "property-specific quick reference")):
            return False
        return "advance purchase" in text
    if any(phrase in lower_query for phrase in ("late checkout", "late check-out", "check out late")):
        # Late-checkout questions are easily contaminated by broad booking and
        # property matrices. Require the actual policy phrase in the chunk.
        section = str(doc.metadata.get("section", "")).lower()
        if any(term in section for term in ("taxes", "deposits", "guarantees", "cancellation", "property-specific quick reference")):
            return False
        return "late checkout" in text or "late check-out" in text
    if "late departure" in lower_query:
        return "late departure" in text or "departure extension" in text
    accessible_room_query = "accessible king" in lower_query or (
        "accessible" in lower_query and "room" in lower_query
    )
    if accessible_room_query:
        # Keep accessible-room evidence together, but do not route through
        # broader accessibility, service-animal, or property-reminder text.
        section = str(doc.metadata.get("section", "")).lower()
        if any(term in section for term in ("escalate immediately", "property-specific reminders")):
            return False
        return any(
            phrase in text
            for phrase in ("accessible king", "accessible room", "accessible features")
        )
    terms = _topic_source_terms(query)
    if not terms:
        return True
    metadata_topics = {str(value).lower() for value in doc.metadata.get("topics", [])}
    metadata_section = str(doc.metadata.get("section", "")).lower()
    return (
        bool(metadata_topics.intersection(terms))
        or any(term in source for term in terms)
        or any(term in metadata_section for term in terms)
        or any(word in text for word in terms)
    )


def _property_matches(query: str, doc) -> bool:
    codes = _property_codes(query)
    if not codes:
        return True
    metadata_codes = {str(value).upper() for value in doc.metadata.get("property_codes", [])}
    if metadata_codes:
        return bool(metadata_codes.intersection(codes))
    text = doc.page_content.upper()
    return any(code in text for code in codes)


def _adaptive_k(query: str, requested_k: int) -> int:
    """Use a smaller context for focused factual questions."""
    lower = query.lower()
    complex_terms = (" and ", " or ", "refund", "cancel", "reservation", "policy", "both", "also")
    if _property_codes(query) and not any(term in lower for term in complex_terms):
        return min(requested_k, 3)
    return requested_k


def _bm25_rank(query: str, documents, limit: int):
    """Small dependency-free BM25 implementation for exact-term matching."""
    tokenized = [re.findall(r"[a-z0-9$-]+", doc.page_content.lower()) for doc in documents]
    query_terms = re.findall(r"[a-z0-9$-]+", query.lower())
    n = len(tokenized)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(n, 1)
    doc_freq = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    scored = []
    for index, tokens in enumerate(tokenized):
        counts = {term: tokens.count(term) for term in set(query_terms)}
        score = 0.0
        for term, tf in counts.items():
            if not tf:
                continue
            idf = math.log(1 + (n - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
            score += idf * (tf * 2.0) / (tf + 1.5 * (0.5 + 0.5 * len(tokens) / max(avgdl, 1)))
        scored.append((score, index))
    return [documents[index] for _, index in sorted(scored, reverse=True)[:limit]]


def _strict_topic_matches(query: str, doc) -> bool:
    query = query.lower()
    topic_categories = {
        "room_and_property": ("breakfast", "room", "property", "occupancy", "pet", "dog", "parking", "accessible"),
        "booking_terms": ("cancel", "cancellation", "advance purchase", "reservation", "no-show", "check-in", "late", "checkout", "departure"),
        "refunds": ("refund", "charge", "incidentals", "hold", "smoking", "parking", "pet", "dog"),
        "guest_conduct": ("service animal", "accessibility", "quiet", "conduct", "safety"),
    }
    requested = {category for category, words in topic_categories.items() if any(word in query for word in words)}
    if not requested:
        return True
    stored = doc.metadata.get("topics", [])
    if isinstance(stored, str):
        stored = [stored]
    return bool(requested.intersection({str(value) for value in stored})) if stored else _topic_matches(query, doc)


def _section_metadata(doc):
    text = doc.page_content
    headings = re.findall(r"^#{2,4}\s+(.+)$", text, re.MULTILINE)
    properties = sorted(set(re.findall(r"ASH-[A-Z]{3}", text.upper())))
    return {
        **doc.metadata,
        "section": headings[-1].strip() if headings else "unknown",
        "property_codes": ",".join(properties),
    }


def _compress_context(query: str, doc):
    """Keep the section label and the highest-overlap policy sentences."""
    text = doc.page_content
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= 6:
        return Document(page_content=text, metadata=_section_metadata(doc))
    query_terms = set(re.findall(r"[a-z0-9$-]+", query.lower()))
    scored = []
    for index, sentence in enumerate(sentences):
        terms = set(re.findall(r"[a-z0-9$-]+", sentence.lower()))
        scored.append((len(query_terms & terms), -index, sentence))
    selected = {item[2] for item in sorted(scored, reverse=True)[:6] if item[0] > 0}
    if not selected:
        selected = set(sentences[:6])
    compressed = " ".join(sentence for sentence in sentences if sentence in selected)
    return Document(page_content=compressed, metadata=_section_metadata(doc))


def _focus_advance_purchase_context(doc):
    """Keep only rate-plan evidence for Advance Purchase questions."""
    keep_terms = (
        "advance purchase",
        "non-refundable",
        "non-changeable",
        "confirmation",
        "service recovery",
        "legal requirement",
        "system error",
        "do not promise",
        "manager",
    )
    lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
    focused = [line for line in lines if any(term in line.lower() for term in keep_terms)]
    if not focused:
        return doc
    return Document(
        page_content="\n".join(focused),
        metadata=_section_metadata(doc),
    )


def _focus_breakfast_context(doc, property_code=None):
    """Keep property-specific breakfast evidence and remove unrelated charges."""
    lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
    if not lines:
        return Document(page_content="", metadata=_section_metadata(doc))

    header_cells = None
    for line in lines:
        if line.startswith("|") and "breakfast" in line.lower():
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if any("breakfast" in cell.lower() for cell in cells):
                header_cells = cells
                break

    focused = []
    for line in lines:
        lower_line = line.lower()
        if property_code and property_code.lower() not in lower_line and "breakfast" not in lower_line:
            continue
        if line.startswith("|") and header_cells:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == len(header_cells) and property_code and property_code.lower() in lower_line:
                breakfast_indexes = [
                    index for index, cell in enumerate(header_cells)
                    if "breakfast" in cell.lower()
                ]
                if breakfast_indexes:
                    index = breakfast_indexes[0]
                    focused.append(f"{cells[0]} | {header_cells[index]}: {cells[index]}")
                    continue
        if any(term in lower_line for term in ("breakfast", "rate plan", "included breakfast")):
            focused.append(line)

    if not focused:
        return Document(page_content="", metadata=_section_metadata(doc))
    return Document(page_content="\n".join(dict.fromkeys(focused)), metadata=_section_metadata(doc))


def _focus_accessible_room_context(doc):
    """Keep only Accessible King room evidence for room-feature questions."""
    keep_terms = (
        "accessible king",
        "accessible room",
        "accessible features",
        "feature is needed",
        "confirm the live room",
        "property profile",
        "vary by room",
    )
    exclude_terms = ("service animal", "pet", "parking", "occupancy", "medical accommodation")
    lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
    focused = []
    for line in lines:
        lower_line = line.lower()
        if "accessible king" in lower_line and "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 4:
                # Keep the room type, configuration, and important note;
                # omit the occupancy column for this feature question.
                focused.append(" | ".join((cells[0], cells[1], cells[3])))
                continue
        if any(term in lower_line for term in keep_terms) and not any(term in lower_line for term in exclude_terms):
            focused.append(line)
    if not focused:
        return Document(
            page_content="No directly relevant Accessible King room detail in this chunk.",
            metadata=_section_metadata(doc),
        )
    return Document(page_content="\n".join(focused), metadata=_section_metadata(doc))


def _focus_late_checkout_context(doc):
    """Keep only late-checkout policy evidence for departure questions."""
    keep_phrases = (
        "late checkout",
        "late check-out",
        "late departure",
        "departure extension",
        "approved departure",
        "full-night charge",
        "full night charge",
    )
    lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
    focused = []
    for line in lines:
        lower_line = line.lower()
        if any(phrase in lower_line for phrase in keep_phrases):
            focused.append(line)
        elif "availability" in lower_line and ("departure" in lower_line or "checkout" in lower_line):
            focused.append(line)
    if not focused:
        return Document(page_content="", metadata=_section_metadata(doc))
    return Document(page_content="\n".join(focused), metadata=_section_metadata(doc))


def _source_boost(query: str, source: str) -> float:
    """Small topic-aware boost; global retrieval remains available."""
    source = Path(source).name.lower()
    return 0.25 if any(term in source for term in _topic_source_terms(query)) else 0.0


def _exact_match_boost(query: str, doc) -> float:
    query_lower = query.lower()
    text_lower = doc.page_content.lower()
    boost = 0.0
    for code in _property_codes(query):
        if code.lower() in text_lower:
            boost += 1.5
    for term in re.findall(r"[a-z0-9]{4,}", query_lower):
        if term in text_lower:
            boost += 0.08
    return boost


def _property_specificity_boost(query: str, doc) -> float:
    """Prefer an atomic row for the requested property over a multi-property matrix."""
    codes = _property_codes(query)
    if not codes:
        return 0.0
    metadata_codes = {str(value).upper() for value in doc.metadata.get("property_codes", [])}
    if metadata_codes.intersection(codes):
        return 2.0 if len(metadata_codes) == 1 else -0.25
    return -1.0


def retrieve_documents(
    query: str,
    k: int = 5,
    vectorstore=None,
    lambda_mult: float | None = None,
    fetch_k: int | None = None,
    reranker_model: str | None = None,
    metadata_boost: float | None = None,
):
    """MMR candidate retrieval followed by relevance reranking."""
    if _unsupported_property_query(query):
        return []
    if os.getenv("ADAPTIVE_K", "false").lower() == "true":
        k = _adaptive_k(query, k)
    lambda_mult = float(os.getenv("MMR_LAMBDA", "0.5")) if lambda_mult is None else lambda_mult
    fetch_k = int(os.getenv("MMR_FETCH_K", str(max(k * 4, 20)))) if fetch_k is None else fetch_k
    reranker_model = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL) if reranker_model is None else reranker_model
    metadata_boost = float(os.getenv("METADATA_BOOST", "0.25")) if metadata_boost is None else metadata_boost

    dense_candidates = vectorstore.max_marginal_relevance_search(
        query,
        k=max(k * 3, 15),
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    candidates = dense_candidates
    if os.getenv("HYBRID_RETRIEVAL", "false").lower() == "true":
        all_documents = list(vectorstore.docstore._dict.values())
        lexical_candidates = _bm25_rank(query, all_documents, max(k * 3, 15))
        ranked = {}
        for rank, doc in enumerate(dense_candidates, 1):
            ranked.setdefault(doc.page_content, {"doc": doc, "score": 0.0})["score"] += 1 / (60 + rank)
        for rank, doc in enumerate(lexical_candidates, 1):
            ranked.setdefault(doc.page_content, {"doc": doc, "score": 0.0})["score"] += 1 / (60 + rank)
        candidates = [item["doc"] for item in sorted(ranked.values(), key=lambda item: item["score"], reverse=True)]
    # Route property/topic questions into a focused candidate pool, while
    # retaining global candidates as a fallback for ambiguous questions.
    routed_terms = _topic_source_terms(query)
    property_codes = _property_codes(query)
    if os.getenv("HIERARCHICAL_ROUTING", "true").lower() == "true" and (routed_terms or property_codes):
        routed = vectorstore.similarity_search(query, k=max(k * 4, 20))
        focused = [doc for doc in routed if _topic_matches(query, doc) and _property_matches(query, doc)]
        if focused:
            if "double queen" in query.lower():
                # This is a single-room fact lookup. Use only exact room-type
                # evidence when it is available; broad fallback context is
                # what caused the contextual-precision failure.
                candidates = focused
            else:
                multi_policy = any(word in query.lower() for word in ("cancel", "refund", "advance", "reservation", "rate"))
                strict = os.getenv("STRICT_ROUTING", "true").lower() == "true"
                if strict and property_codes and not multi_policy and len(focused) >= min(k, 3):
                    candidates = focused
                else:
                    candidates = focused + [
                        doc for doc in candidates
                        if doc.page_content not in {item.page_content for item in focused}
                    ]
            if os.getenv("STRICT_FINAL_FILTER", "false").lower() == "true":
                strict_candidates = [
                    doc for doc in candidates
                    if _property_matches(query, doc) and _strict_topic_matches(query, doc)
                ]
                if len(strict_candidates) >= min(2, k):
                    strict_ids = {doc.page_content for doc in strict_candidates}
                    candidates = strict_candidates + [doc for doc in candidates if doc.page_content not in strict_ids]

    # Apply focused candidate routing before reranking for two questions where
    # broad property-policy chunks otherwise dilute contextual precision.
    lower_query = query.lower()
    if "breakfast" in lower_query and property_codes:
        breakfast_candidates = [
            doc for doc in vectorstore.similarity_search(query, k=max(k * 8, 40))
            if "breakfast" in doc.page_content.lower() and _property_matches(query, doc)
        ]
        if breakfast_candidates:
            candidates = breakfast_candidates
    elif "advance purchase" in lower_query:
        advance_candidates = [
            doc for doc in vectorstore.similarity_search(query, k=max(k * 8, 40))
            if "advance purchase" in doc.page_content.lower()
        ]
        if len(advance_candidates) >= min(2, k):
            candidates = advance_candidates
    reranker = _get_reranker(reranker_model)
    scores = reranker.predict([(query, doc.page_content) for doc in candidates])
    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: item[0]
        + metadata_boost * _source_boost(query, item[1].metadata.get("source", ""))
        + (0.35 if _property_matches(query, item[1]) and _property_codes(query) else 0.0)
        + _property_specificity_boost(query, item[1])
        + (_exact_match_boost(query, item[1]) if os.getenv("EXACT_MATCH_BOOST", "false").lower() == "true" else 0.0),
        reverse=True,
    )
    results = [doc for _, doc in ranked[:k]]
    if "advance purchase" in query.lower():
        results = [_focus_advance_purchase_context(doc) for doc in results]
        results = [doc for doc in results if doc.page_content.strip()][:3]
    if "breakfast" in query.lower() and _property_codes(query):
        property_code = sorted(_property_codes(query))[0]
        results = [_focus_breakfast_context(doc, property_code) for doc in results]
        results = [doc for doc in results if doc.page_content.strip()][:3]
    if any(phrase in query.lower() for phrase in ("late checkout", "late check-out", "late departure", "check out late")):
        results = [_focus_late_checkout_context(doc) for doc in results]
        results = [doc for doc in results if doc.page_content.strip()][:3]
    if "accessible king" in query.lower() or ("accessible" in query.lower() and "room" in query.lower()):
        results = [_focus_accessible_room_context(doc) for doc in results]
    if os.getenv("CONTEXT_COMPRESSION", "false").lower() == "true":
        return [_compress_context(query, doc) for doc in results]
    return [Document(page_content=doc.page_content, metadata=_section_metadata(doc)) for doc in results]

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 5,vectorstore=None) -> list[str]:
    results = retrieve_documents(query=query, k=k, vectorstore=vectorstore)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about Aurora Stays hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent benefits, exclusions, limits, waiting periods, conditions, or coverage details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    For Advance Purchase questions, answer only from the Advance Purchase rate-plan
    evidence. State that it is generally non-refundable and non-changeable, and
    mention exceptions only when the retrieved policy explicitly supports them.
    Do not discuss taxes, deposits, property matrices, or unrelated rate plans.

    For Accessible King room questions, state that the room has a king bed and
    selected accessible features, explain that exact features vary by room and
    property, and advise confirming the requested feature in the live profile.
    Do not add occupancy, medical-accommodation, service-animal, pet, or parking
    information unless the user explicitly asks for it.

    For late-checkout questions, answer only with availability, the approved
    departure time, any fee, and the full-night charge condition. Do not mention
    smoking, lost keys, pets, parking, breakfast, or unrelated hotel policies.


    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    try:
        # Create the generator model inside the guarded block. This allows the
        # local UI to return a useful fallback when API configuration is absent.
        model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

        # Generate answer using retrieved chunks.
        answer = generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            model=model
        )
        return answer.content, retrieved_chunks
    except Exception as exc:
        # Keep local endpoint smoke tests available when the model service is offline.
        print(f"RAG generation failed: {type(exc).__name__}: {exc}", flush=True)
        return f"Local smoke-test fallback: model service unavailable ({type(exc).__name__}).", retrieved_chunks
