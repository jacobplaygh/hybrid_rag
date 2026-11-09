# Installation

## Requirements

- Python 3.10 or later.
- The packages listed in `requirements_fastapi.txt`.
- An NVIDIA API key only when using NVIDIA-hosted generation or model-backed agent planning.

## Install

Run these commands from the workspace directory that contains the `hybrid_rag` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\hybrid_rag\requirements_fastapi.txt
$env:PYTHONPATH = (Join-Path $PWD 'hybrid_rag')
```

If PowerShell prevents environment activation, either adjust the execution policy for your user or invoke `.venv\Scripts\python.exe` directly.

## Configuration

The application loads `hybrid_rag\.env` when present. There is no checked-in `.env.example`; create that file yourself or set variables in the process environment. Do not commit credentials.

Common settings are declared in `api/config.py`:

| Setting | Default / behavior |
|---|---|
| `NVIDIA_API_KEY` | Unset; model-backed generation is unavailable without credentials |
| `DEFAULT_CHAT_MODEL` | Uses `NVIDIA_MODEL` when set, otherwise `meta/llama3-8b-instruct` |
| `VECTOR_STORE_TYPE` | `chroma` |
| `CHROMA_PERSIST_DIR` | `hybrid_rag/data/chroma` |
| `UPLOAD_DIR` | `hybrid_rag/uploaded_docs` |
| `AUTH_REQUIRED` | `false` |
| `API_KEY` | Unset |
| `AGENTIC_LOOP_ENABLED` | `false`; set to `true` to opt in to iterative retrieval |
| `DYNAMIC_ROUTING_ENABLED` | `true` |
| `RETRIEVAL_RERANK` | `true` |

Relative data paths resolve from the process working directory. Review the settings class for additional upload, cache, CRAG, tracing, and Prometheus options.

## Verify

```powershell
python -m pytest -q
```

Then see the [quick start](./quick-start.md) to launch the API and try it.
