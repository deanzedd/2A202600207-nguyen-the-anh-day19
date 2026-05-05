"""Utility functions: load corpus, chunking, normalize entities, LLM calls."""
import os, time, json, re
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Token / cost tracking ----------
total_prompt_tokens = 0
total_completion_tokens = 0

def call_llm(prompt, model="gpt-4o-mini", temperature=0):
    """Call OpenAI and track token usage."""
    global total_prompt_tokens, total_completion_tokens
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if resp.usage:
        total_prompt_tokens += resp.usage.prompt_tokens
        total_completion_tokens += resp.usage.completion_tokens
    return resp.choices[0].message.content.strip()

# ---------- Corpus loading & chunking ----------
def load_chunks(path=None):
    """Read corpus and split into paragraph chunks."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "tech_company_corpus.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [{"chunk_id": f"chunk_{i+1:03d}", "text": p} for i, p in enumerate(paragraphs)]

# ---------- Entity normalization ----------
ENTITY_ALIASES = {
    "Open AI": "OpenAI", "OpenAI Inc.": "OpenAI", "OpenAI Inc": "OpenAI",
    "Google LLC": "Google", "Alphabet": "Google",
    "Microsoft Corporation": "Microsoft", "Microsoft Corp": "Microsoft",
    "Meta Platforms": "Meta", "Facebook": "Meta",
    "Amazon.com": "Amazon", "Amazon Web Services": "AWS",
    "Tesla Inc": "Tesla", "Tesla Motors": "Tesla",
    "NVIDIA Corporation": "NVIDIA", "Nvidia": "NVIDIA",
    "SpaceX": "SpaceX",
    "Samsung Electronics": "Samsung",
    "X.ai": "X.AI", "xAI": "X.AI",
}

def normalize_entity(name):
    """Normalize entity name."""
    name = name.strip()
    name = ENTITY_ALIASES.get(name, name)
    return name

def normalize_triples(triples):
    """Normalize and deduplicate triples."""
    seen, result = set(), []
    for t in triples:
        s = normalize_entity(t["subject"])
        o = normalize_entity(t["object"])
        r = t["relation"].strip().upper()
        cid = t.get("source_chunk_id", "")
        if not s or not o:
            continue
        key = (s, r, o)
        if key not in seen:
            seen.add(key)
            result.append({"subject": s, "relation": r, "object": o, "source_chunk_id": cid})
    return result
