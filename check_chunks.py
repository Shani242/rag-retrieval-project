from dotenv import load_dotenv

load_dotenv()

from src.retrieval_logic import retrieve_context

test_queries = [
    ("tax", "Single word - should find results"),
    ("accounting", "Single word - should find results"),
    ("How to save money on taxes?", "Multi-word query"),
    ("deductions", "Common term"),

    ("", "Empty query"),
    ("   ", "Only spaces"),
    ("xyzabc", "Nonsense word"),
    ("!@#$%", "Special characters"),
    ("123456", "Numbers only"),
    ("savings", "Word not in dataset"),
]

print("=" * 80)
print("TESTING RAG RETRIEVAL SYSTEM")
print("=" * 80)

for query, description in test_queries:
    print(f"\n{'=' * 80}")
    print(f"Query: '{query}'")
    print(f"Description: {description}")
    print(f"{'-' * 80}")

    result = retrieve_context(query)

    print(f"Results: {result.num_results} chunk(s) found")

    if result.num_results > 0:
        for i, chunk in enumerate(result.results):
            print(f"\n  [{i + 1}] Chunk ID: {chunk.id}")
            print(f"      Similarity Score: {chunk.score}")
            print(f"      Text Preview: {chunk.text[:100]}...")
    else:
        if not query.strip():
            print("  ✓ Correctly handled empty input")
        else:
            print("  ✓ No results above threshold (expected)")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)