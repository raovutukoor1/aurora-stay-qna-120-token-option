# Aurora Stays Q&A — 120-Token Option

This repository contains the Aurora Stays RAG application configured with the
native-safe `120` token chunks / `16` token overlap option for the
`HarishMaths/Hotel-Policy-Embedding` model. The Hugging Face tokenizer limit is
128 tokens, so the chunk configuration stays within the model's effective limit.

The latest held-out run used `k=5`, `temperature=0.0`, `top_p=1.0`, and
`max_tokens=512`. It passed 7/10 cases with a weighted composite of 0.867.

## GitHub Codespaces workflow

1. Create a Codespace from the `main` branch.
2. Add `OPENAI_API_KEY` and, if required, `OPENAI_BASE_URL` as Codespaces secrets.
3. Build and run the backend:

```bash
cd backend
docker build -t aurora-stays-backend .
docker run --rm -p 7860:7860 -e OPENAI_API_KEY -e OPENAI_BASE_URL aurora-stays-backend
```

4. Set port 7860 to Public and use its forwarded URL as `BACKEND_URL`.
5. Run the frontend from `frontend/` or use its Dockerfile on port 8501.

The FAISS index is stored at `backend/faiss_index`. Do not commit API keys,
GitHub tokens, local config files, or Hugging Face caches.
