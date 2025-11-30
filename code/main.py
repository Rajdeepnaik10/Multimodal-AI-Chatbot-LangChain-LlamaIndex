# Colab-ready notebook for "Multimodal AI Chatbot with LangChain and LlamaIndex"
# Author: Rajdeep Naik
# Intern ID: GWING082631
# Project: Multimodal AI Chatbot with LangChain and LlamaIndex

# -----------------------
# 1) Install required libraries
# -----------------------
!pip install -q sentence-transformers faiss-cpu transformers datasets nltk llama-index langchain

# -----------------------
# 2) Imports
# -----------------------
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os, json, textwrap
from PIL import Image
import requests
from io import BytesIO

# -----------------------
# 3) Simple utilities
# -----------------------
def pretty_print(title, text):
    print(f"\n==== {title} ====\n")
    print(textwrap.fill(text, 100))

# -----------------------
# 4) Load embedding model and text generation model
# -----------------------
pretty_print("INFO", "Loading embedding model (all-MiniLM-L6-v2) and a small generation model (google/flan-t5-small).")


embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # fast embeddings

# text generation model (small)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer)

# -----------------------
# 5) Example documents to index (replace or extend later)
# -----------------------
docs = [
    {"id": "doc1", "text": "LangChain is a library that helps build applications with LLMs and manage chains of prompts."},
    {"id": "doc2", "text": "LlamaIndex (also known as GPT Index) provides tools to index documents and retrieve context for LLMs."},
    {"id": "doc3", "text": "Multimodal AI combines text and image inputs to produce context-aware responses."},
    {"id": "doc4", "text": "Flan-T5 small is a lightweight sequence-to-sequence model that can be used for simple generation tasks."},
]

pretty_print("Docs indexed (examples)", "\n".join([f"{d['id']}: {d['text']}" for d in docs]))

# -----------------------
# 6) Create embeddings and FAISS index
# -----------------------
texts = [d['text'] for d in docs]
embs = embed_model.encode(texts, convert_to_numpy=True)
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
# normalize vectors for cosine similarity
faiss.normalize_L2(embs)
index.add(embs)

# store mapping id -> text
id_to_text = {i:docs[i]['text'] for i in range(len(docs))}
pretty_print("Index", f"FAISS index with {index.ntotal} vectors created.")

# -----------------------
# 7) Retrieval function
# -----------------------
def retrieve(query, k=2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = [id_to_text[i] for i in I[0]]
    return results

# -----------------------
# 8) Simple answer synthesis using retrieved docs and FLAN-T5
# -----------------------
def generate_answer(query, retrieved_docs):
    # Build a simple prompt providing retrieved context
    context = "\n\n".join(retrieved_docs)
    prompt = f"Use the context below to answer the question. If the answer is not in the context, say 'I don't have enough info.'\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    out = generator(prompt, max_length=200, do_sample=False)[0]['generated_text']
    return out

# -----------------------
# 9) Dummy multimodal image handler (placeholder)
# -----------------------
def handle_image_input(image_url):
    """
    Placeholder image handler: downloads image and returns a short automatic caption placeholder.
    Real multimodal implementation would use CLIP, BLIP, or a multimodal LLM.
    """
    try:
        resp = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # For now return a placeholder description; in a full implementation we'd call an image captioning model
        return "Image received. (Placeholder caption: a sample image used for multimodal testing.)"
    except Exception as e:
        return f"Could not load image: {e}"

# -----------------------
# 10) Chat function combining text and optional image
# -----------------------
def chat(query_text=None, image_url=None):
    combined_info = []
    if image_url:
        img_desc = handle_image_input(image_url)
        combined_info.append("Image Description: " + img_desc)
    if query_text:
        combined_info.append("User Query: " + query_text)
    combined_prompt = "\n".join(combined_info)
    # Use retrieval to get context
    retrieved = retrieve(query_text or img_desc, k=3)
    answer = generate_answer(query_text or img_desc, retrieved)
    return {"retrieved_docs": retrieved, "answer": answer, "image_desc": (img_desc if image_url else None)}

# -----------------------
# 11) Demo: run some example queries
# -----------------------
pretty_print("Demo", "Running demo queries now...")

q1 = "What is LangChain and why use it?"
res1 = chat(query_text=q1)
pretty_print("Query 1", q1)
print("Retrieved docs:", res1['retrieved_docs'])
pretty_print("Answer 1", res1['answer'])

q2 = "Explain LlamaIndex and its use for contextual Q&A."
res2 = chat(query_text=q2)
pretty_print("Query 2", q2)
print("Retrieved docs:", res2['retrieved_docs'])
pretty_print("Answer 2", res2['answer'])

# -----------------------
# 12) Demo: image placeholder (multimodal)
# -----------------------
img_url = "https://images.unsplash.com/photo-1558980664-10f6d8f78f94?w=800&q=80"
res3 = chat(query_text="Describe the image and relate it to multimodal AI.", image_url=img_url)
pretty_print("Image demo", f"Image URL: {img_url}")
print("Image desc:", res3['image_desc'])
print("Retrieved docs:", res3['retrieved_docs'])
pretty_print("Answer (image-related)", res3['answer'])

# -----------------------
# 13) Save notebook artifacts for GitHub
# -----------------------
# Save example outputs to a JSON outputs file for screenshots insertion into report later
outputs = {
    "query1": {"query": q1, "retrieved": res1['retrieved_docs'], "answer": res1['answer']},
    "query2": {"query": q2, "retrieved": res2['retrieved_docs'], "answer": res2['answer']},
    "image_demo": {"image_url": img_url, "image_desc": res3['image_desc'], "answer": res3['answer']}
}
with open("demo_outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

pretty_print("Done", "Demo runs complete. Please take screenshots of the outputs and the notebook cells showing 'Answer 1', 'Answer 2', and 'Image demo' for your report.")
