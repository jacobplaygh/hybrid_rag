# Installation Guide

Set up the Hybrid RAG FastAPI backend for development or production use.

## Prerequisites

- Python 3.10 or higher
- Virtual environment tool (venv, conda, or pyenv)
- ~2GB disk space for dependencies
- Optional: NVIDIA API key for LLM features

## Step 1: Clone/Access the Repository

```powershell
# Navigate to the backend directory
cd d:\projects\ai_ws\rag\multimodel_rag\hybrid_rag
```

## Step 2: Create Virtual Environment

### Using Python venv
```powershell
# Create virtual environment
python -m venv genai

# Activate it
.\genai\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Using Conda
```powershell
conda create -n hybrid-rag python=3.10
conda activate hybrid-rag
```

## Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install FastAPI backend dependencies
pip install -r requirements_fastapi.txt
```

## Step 4: Configure Environment

```powershell
# Copy the example environment file
Copy-Item .env.example .env

# Edit .env with your settings (or see defaults)
```

### Configuration Options

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEBUG` | `false` | Enable verbose logging |
| `AUTH_REQUIRED` | `false` | Require API key |
| `API_KEY` | `none` | API key for auth |
| `NVIDIA_API_KEY` | `none` | LLM API key (optional) |
| `RATE_LIMIT_ENABLED` | `true` | Rate limiting |
| `UPLOAD_DIR` | `./uploaded_docs` | Document storage |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store location |
| `VECTOR_STORE_TYPE` | `chroma` | `chroma` or `in-memory` |

### Example Configuration

For development:
```powershell
$env:DEBUG = "true"
$env:AUTH_REQUIRED = "false"
$env:VECTOR_STORE_TYPE = "chroma"
```

For production:
```powershell
$env:DEBUG = "false"
$env:AUTH_REQUIRED = "true"
$env:API_KEY = "your-secure-key"
$env:NVIDIA_API_KEY = "your-nvidia-api-key"
```

## Step 5: Verify Installation

```powershell
# Check Python version
python --version  # Should be 3.10+

# Check key dependencies
python -c "import fastapi; import chromadb; import langchain; print('✅ All core dependencies installed')"

# Run test suite (optional but recommended)
python -m pytest tests/ -v

# Expected: 35/35 tests passing
```

## Step 6: (Optional) Set Up IDE

### VS Code Setup
1. Install Python extension by Microsoft
2. Select interpreter: `.venv/bin/python` (or `genai\Scripts\python.exe` on Windows)
3. Install extensions:
   - Python
   - Pylance
   - FastAPI (optional)

### PyCharm Setup
1. Go to Settings → Project → Python Interpreter
2. Click gear → Add → Existing Environment
3. Select `genai\Scripts\python.exe`

## Installation Verification

Run this command to verify everything is set up correctly:

```powershell
python -c "
import sys
print(f'Python version: {sys.version}')
import fastapi
print(f'FastAPI: {fastapi.__version__}')
import chromadb
print(f'ChromaDB: {chromadb.__version__}')
import langchain
print(f'LangChain: {langchain.__version__}')
print('✅ Installation successful!')
"
```

Expected output:
```
Python version: 3.10.x ...
FastAPI: 0.104.1
ChromaDB: 0.4.0
LangChain: 1.2.17
✅ Installation successful!
```

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'fastapi'
```
→ Run `pip install -r requirements_fastapi.txt`

### Virtual Environment Not Activating
```powershell
# Windows PowerShell policy issue
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\genai\Scripts\Activate.ps1
```

### ChromaDB Issues
```
ImportError: No module named 'hnswlib'
```
→ Run `pip install chroma-hnswlib`

### Permission Denied on Linux/Mac
```bash
# Fix file permissions
chmod +x venv/bin/activate
source venv/bin/activate
```

## Next Steps

✅ Installation complete!

1. **Ready to run the server?** → See [Quick Start](./quick-start.md)
2. **Want to understand the system?** → See [Architecture](../architecture/)
3. **Ready to develop?** → See [Testing Guide](../development/testing.md)

---

**Need Help?**
- Check [Quick Start](./quick-start.md) for common issues
- Review [Troubleshooting](#troubleshooting) section above
- See [../README.md](../README.md) for more info
