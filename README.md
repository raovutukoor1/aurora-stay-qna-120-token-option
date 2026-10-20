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
3. Build and run both services:

```bash
./docker.sh start
```

The backend uses host networking so it can use the Codespace's working DNS
resolver. The frontend stays on a Docker bridge network and reaches the backend
through `host.docker.internal`. Port 7860 is the backend's fixed listen port;
the frontend port can be changed with `FRONTEND_PORT`.

4. Set port 8501 to Public and open the forwarded frontend URL.
5. Run an end-to-end health check and sample question from the frontend
   container:

```bash
./docker.sh test
```

To stop both containers, run `./docker.sh stop`.

The FAISS index is stored at `backend/faiss_index`. Do not commit API keys,
GitHub tokens, local config files, or Hugging Face caches.
