
import os
import numpy as np
import faiss
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import PyPDF2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def read_resume(file_path: str) -> str:
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if file_path.lower().endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text

    return ""

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        files = request.files.getlist("resumes")

        if not job_description or not files:
            return render_template(
                "index.html",
                error="Please provide a job description and upload at least one resume.",
                results=None,
            )

        resume_texts = []
        resume_names = []

        for file in files:
            if file.filename == "":
                continue

            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            text = read_resume(save_path)
            if text.strip():
                resume_texts.append(text)
                resume_names.append(filename)

        if not resume_texts:
            return render_template(
                "index.html",
                error="No readable resumes found (only .pdf and .txt are supported).",
                results=None,
            )

        resume_embeddings = model.encode(resume_texts)
        resume_embeddings = np.array(resume_embeddings).astype("float32")
        faiss.normalize_L2(resume_embeddings)

        dimension = resume_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(resume_embeddings)

        job_embedding = model.encode([job_description])
        job_embedding = np.array(job_embedding).astype("float32")
        faiss.normalize_L2(job_embedding)

        top_k = min(5, len(resume_texts))
        distances, indices = index.search(job_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            score = float(distances[0][rank]) * 100.0
            results.append(
                {
                    "rank": rank + 1,
                    "filename": resume_names[idx],
                    "score": round(score, 2),
                    "preview": resume_texts[idx][:500],
                }
            )

        return render_template(
            "index.html",
            results=results,
            job_description=job_description,
            error=None,
        )

    return render_template("index.html", results=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
