# syntax=docker/dockerfile:1

# ---- model download, kept in its own stage so CPU and GPU builds share it ----
FROM python:3.12-slim AS model

ARG MODEL_ID=stabilityai/sdxl-turbo

COPY requirements.txt download_model.py /tmp/
RUN pip install --no-cache-dir $(grep huggingface_hub /tmp/requirements.txt)
# optional token for gated models: docker build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
	python /tmp/download_model.py "$MODEL_ID" /model && rm -rf /model/.cache

# ---- runtime ----
FROM python:3.12-slim

# PyPI torch includes CUDA, use https://download.pytorch.org/whl/cpu for a CPU-only image
ARG TORCH_INDEX_URL=https://pypi.org/simple
ARG TORCH_VERSION=2.14.1
# defaults suitable for SDXL-Turbo, adjust together with MODEL_ID
ARG DEFAULT_STEPS=4
ARG DEFAULT_GUIDANCE=0.0

RUN pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY --from=model /model /model
COPY generate.py /app/

ENV MODEL_PATH=/model \
	DEFAULT_STEPS=$DEFAULT_STEPS \
	DEFAULT_GUIDANCE=$DEFAULT_GUIDANCE \
	HF_HUB_OFFLINE=1 \
	HF_HUB_DISABLE_TELEMETRY=1 \
	HF_HOME=/tmp/huggingface \
	TRANSFORMERS_VERBOSITY=error \
	PYTHONWARNINGS=ignore::FutureWarning

# writable for any --user
RUN mkdir -p /data && chmod 777 /data
WORKDIR /data

ENTRYPOINT ["python", "/app/generate.py"]
