"""Task 3 & 4: Extract triples from corpus using LLM, normalize, save to CSV."""
import json, time, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_chunks, call_llm, normalize_triples, PROJECT_ROOT

PROMPT_TEMPLATE = """Extract factual knowledge triples from the following text.
Return only valid JSON array.
Each triple must have: subject, relation, object.
Use concise uppercase relation names such as FOUNDED_BY, FOUNDED_IN, CREATED_PRODUCT, INVESTED_IN, ACQUIRED, PARTNERED_WITH, COMPETES_WITH, LOCATED_IN, WORKS_IN_FIELD.
Text: {chunk_text}"""


def extract_triples_from_chunk(chunk):
    """Extract triples from a single chunk with retry."""
    prompt = PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
    for attempt in range(3):
        try:
            raw = call_llm(prompt)
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]
            triples = json.loads(raw)
            for t in triples:
                t["source_chunk_id"] = chunk["chunk_id"]
            return triples
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Retry {attempt+1} for {chunk['chunk_id']}: {e}")
            time.sleep(1)
    return []


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    all_triples = []
    t0 = time.time()
    for chunk in chunks:
        print(f"Extracting from {chunk['chunk_id']}...")
        triples = extract_triples_from_chunk(chunk)
        all_triples.extend(triples)
    extraction_time = time.time() - t0

    # Normalize & deduplicate
    all_triples = normalize_triples(all_triples)
    print(f"Total unique triples: {len(all_triples)}")

    df = pd.DataFrame(all_triples, columns=["subject", "relation", "object", "source_chunk_id"])
    out_path = os.path.join(PROJECT_ROOT, "outputs", "triples.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}  (extraction took {extraction_time:.1f}s)")


if __name__ == "__main__":
    main()
