"""Part D (tasks 1-4) - retrieval-grounded policy assistant over data/policies/*.txt.
Run after:  python generate_data.py
The retrieval functions are reused by partD_agent.py (import), so the evaluation only
runs when this file is executed directly.
"""
import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = "data/policies"
CHUNK_WORDS, OVERLAP = 40, 10


# ------------------------------------------------------------- 1. chunking
def chunk_words(words, size=CHUNK_WORDS, overlap=OVERLAP):
    step, chunks, start = size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches the last word
            return chunks
        start += step


def build_chunks(size=CHUNK_WORDS, overlap=OVERLAP):
    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for text in chunk_words(words, size, overlap):
            chunks.append({"source": os.path.basename(path), "text": text})
    return chunks


# ----------------------------------------------------- 2. vectorise, retrieve
class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in top]


DEFAULT = Retriever(build_chunks())


def retrieve(query, k=3):
    return DEFAULT.retrieve(query, k)


# -------------------------------------------------- 3. augment and answer
SYSTEM = ("You are a POLICY ASSISTANT for Instamart support agents. Answer only from the supplied "
          "context. If the context does not contain the answer, reply that you could not find "
          "the information in the policy documents.")


def answer(question, retriever=None):
    hits = (retriever or DEFAULT).retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


# ------------------------------------------------------------ 4. evaluate
def evaluate(retriever, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    rows = []
    for q in ev.itertuples():
        text, sources = answer(q.question, retriever)
        if q.expected_doc == "NONE":
            ok_doc, ok_fact = None, None
            ok_fallback = text.startswith("I could not find")
        else:
            ok_doc = q.expected_doc in sources
            ok_fact = q.expected_fact in text
            ok_fallback = None
        rows.append({"question": q.question, "expected_doc": q.expected_doc, "doc_in_top3": ok_doc,
                     "fact_in_answer": ok_fact, "fallback": ok_fallback, "answer": text})
    df = pd.DataFrame(rows)
    ans, none = df[df.expected_doc != "NONE"], df[df.expected_doc == "NONE"]
    res = {"setting": label,
           "doc_in_top3": f"{int(ans.doc_in_top3.sum())}/{len(ans)}",
           "fact_in_answer": f"{int(ans.fact_in_answer.sum())}/{len(ans)}",
           "fallback_unanswerable": f"{int(none.fallback.sum())}/{len(none)}"}
    return res, df


if __name__ == "__main__":
    print("=" * 70, "\n1. CHUNKING (40 words, overlap 10)\n" + "=" * 70)
    counts = pd.Series([c["source"] for c in DEFAULT.chunks]).value_counts()
    print(counts.to_string(), "\ntotal:", len(DEFAULT.chunks),
          "(ref 15: refund 4, delivery 3, escalation 3, payment 3, privacy 2)")

    print("\n" + "=" * 70, "\n2. RETRIEVAL\n" + "=" * 70)
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']:22s} score {h['score']:.4f}  {h['text'][:60]}...")

    print("\n" + "=" * 70, "\n3. ANSWER\n" + "=" * 70)
    txt, src = answer("Who must approve a refund above INR 1000?")
    print("answer:", txt, "| sources:", src)

    print("\n" + "=" * 70, "\n4. EVALUATION\n" + "=" * 70)
    r10, d10 = evaluate(DEFAULT, "overlap 10")
    zero = Retriever(build_chunks(overlap=0))
    r0, d0 = evaluate(zero, "overlap 0")
    print(pd.DataFrame([r10, r0]).to_string(index=False), "\n(ref overlap 10: 12/12, 12/12, 2/2)")
    print("zero-overlap chunks:", len(zero.chunks))
    broken = d0[(d0.expected_doc != "NONE") & ~(d0.doc_in_top3.astype(bool) & d0.fact_in_answer.astype(bool))]
    print("\nQuestions that break with overlap 0:")
    for r in broken.itertuples():
        print(f"  - {r.question}\n    doc_in_top3={r.doc_in_top3} fact_in_answer={r.fact_in_answer}\n    answer: {r.answer}")
    assert len(broken) >= 1
