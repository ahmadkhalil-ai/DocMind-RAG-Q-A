# generator.py
from anthropic import Anthropic
from groq import Groq
from models import SourceChunk, AskResponse
from enum import Enum
from dotenv import load_dotenv
import os
load_dotenv()

anthropic_client = Anthropic()
groq_client = Groq()  # uses GROQ_API_KEY from env

class ModelChoice(str, Enum):
    CLAUDE = "claude"
    GROQ_LLAMA = "groq-llama"

SYSTEM_PROMPT = """You are DocMind, a precise document Q&A assistant.

Rules:
- Answer ONLY from the provided context chunks. Never use outside knowledge.
- If the answer isn't in the context, say: "I couldn't find that in the uploaded documents."
- Always mention which document your answer comes from.
- Be concise. Avoid filler phrases."""

def generate_answer(query: str, chunks: list[tuple[str, str, float]], model: ModelChoice = ModelChoice.CLAUDE) -> AskResponse:
    if not chunks:
        return AskResponse(
            answer="No relevant content found in the uploaded documents.",
            sources=[],
            confidence=0.0
        )

    context_lines = [
        f"[Chunk {i+1} from '{doc}' | relevance: {score:.2f}]\n{text}"
        for i, (doc, text, score) in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(context_lines)
    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"

    if model == ModelChoice.CLAUDE:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        answer_text = response.content[0].text

    elif model == ModelChoice.GROQ_LLAMA:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast + free tier
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        answer_text = response.choices[0].message.content

    sources = [
        SourceChunk(
            doc=doc,
            text=text[:150] + "..." if len(text) > 150 else text,
            score=round(score, 3)
        )
        for doc, text, score in chunks[:3]
    ]
    avg_confidence = round(sum(s.score for s in sources) / len(sources), 3)

    return AskResponse(answer=answer_text, sources=sources, confidence=avg_confidence)


def stream_answer(query: str, chunks: list[tuple[str, str, float]], model: ModelChoice = ModelChoice.CLAUDE):
    context_lines = [
        f"[Chunk {i+1} from '{doc}']\n{text}"
        for i, (doc, text, _) in enumerate(chunks)
    ]
    context = "\n\n---\n\n".join(context_lines)
    user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

    if model == ModelChoice.CLAUDE:
        with anthropic_client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    elif model == ModelChoice.GROQ_LLAMA:
        stream = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta