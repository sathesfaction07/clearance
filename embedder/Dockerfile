# EmbeddingGemma served by Ollama, with the model baked into the image at build time, so the
# pod needs no network access at run time: questions are embedded inside the cluster and
# never leave it. Built for the GKE overlay; CI uses precomputed question vectors instead.
FROM ollama/ollama:0.40.16
ENV OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS=/models
RUN ollama serve & sleep 5 && ollama pull embeddinggemma && kill %1
EXPOSE 11434
