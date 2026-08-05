# Hugging Face Space (Docker SDK) / any container host for the evaluation dashboard.
# Serves the Flask app publicly with code execution OFF by default (ALLOW_EXEC=0).
FROM python:3.11-slim

WORKDIR /app

# jscpd is the only detector that is not a Python package, and duplicate_code is the
# only one of the twelve smells it can find. Without Node the hosted dashboard covers
# eleven; with it, all twelve. Node is pulled from the distro rather than a Node base
# image so the Python side stays exactly as it is.
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs npm \
 && npm install -g jscpd@4.0.5 \
 && apt-get purge -y npm && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Dashboard dependencies only -- NOT torch/transformers/datasets (those are the GPU phase).
COPY deploy/requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt
# codebleu pins an old tree-sitter; install without deps so it can't downgrade the
# others or fail the build (it works against the newer tree-sitter). Optional.
RUN pip install --no-cache-dir --no-deps codebleu || true

# Only what the dashboard reads. The repository also carries the benchmark, the
# generations and the rendered reports -- tens of megabytes the app never opens --
# and copying all of it makes the image slow to build for no benefit.
COPY dashboard/ dashboard/
COPY eval_tool/ eval_tool/
COPY smell_injection/*.py smell_injection/
COPY smell_injection/realworld_clean.jsonl smell_injection/

# Public-demo defaults: listen on all interfaces, HF's port, no code execution, no browser popup.
ENV HOST=0.0.0.0 PORT=7860 ALLOW_EXEC=0 DASH_NO_BROWSER=1
EXPOSE 7860

CMD ["python", "dashboard/app.py"]
