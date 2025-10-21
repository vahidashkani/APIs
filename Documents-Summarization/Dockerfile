FROM python:3.10

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Hugging Face cache + token (build arg)
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p $HF_HOME
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Prefetch gated model (requires token & license accepted on your HF account)
RUN python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "mistralai/Mistral-7B-Instruct-v0.2"
print("Downloading:", m)
AutoTokenizer.from_pretrained(m, use_auth_token=True)
AutoModelForCausalLM.from_pretrained(m, use_auth_token=True)
print("Done.")
PY

# App code last for better cache reuse
COPY . /app

ENV WHISPER_MODEL_SIZE=large-v2
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=int8

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
