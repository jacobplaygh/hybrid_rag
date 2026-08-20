from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hybrid RAG UI</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f6fb; }
    .page { max-width: 980px; margin: 0 auto; padding: 24px; }
    h1 { margin-top: 0; }
    .panel { background: #fff; border-radius: 12px; box-shadow: 0 2px 18px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 20px; overflow: hidden; }
    .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; align-items: start; }
    label { display: block; font-weight: 600; margin-bottom: 8px; }
    input, select, textarea { width: 100%; padding: 10px 12px; border: 1px solid #cbd6e2; border-radius: 8px; font-size: 14px; min-width: 0; }
    textarea { min-height: 140px; resize: vertical; }
    button { cursor: pointer; border: none; color: white; background: #2563eb; padding: 12px 18px; border-radius: 8px; font-size: 15px; min-width: 0; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    pre { background: #0f8fafc; color: #1e293b; padding: 16px; border-radius: 12px; overflow: auto; max-height: 420px; border: 1px solid #cbd6e2; white-space: pre-wrap; word-wrap: break-word; }
    .streaming-active { border: 2px solid #2563eb !important; background: #eff6ff !important; }
    .full { grid-column: 1 / -1; }
    .hint { color: #475569; font-size: 14px; margin-top: 4px; }
    @media (max-width: 740px) {
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <h1>Hybrid RAG Web UI</h1>
    <p>Use this page to query the backend, upload documents, and inspect indexed files.</p>

    <div class="panel">
      <div class="full">
        <label for="apiKey">API Key</label>
        <input id="apiKey" type="password" placeholder="Enter API key if required" />
      </div>
      <div class="row">
        <div>
          <label for="queryText">RAG Query</label>
          <textarea id="queryText" rows="4" placeholder="Enter a question or prompt"></textarea>
          <label for="mode">Mode</label>
          <select id="mode">
            <option value="simple">simple</option>
            <option value="multi-turn">multi-turn</option>
            <option value="decompose">decompose</option>
            <option value="structured">structured</option>
          </select>
        </div>
        <div>
          <label for="topK">Top K</label>
          <input id="topK" type="number" min="1" value="3" />
          <label for="model">Model Override</label>
          <input id="model" type="text" placeholder="Optional model override" />
          <div style="margin-top: 12px; display: flex; align-items: center; gap: 8px;">
            <input id="streamToggle" type="checkbox" style="width: auto;" />
            <label for="streamToggle" style="margin-bottom: 0;">Enable Streaming</label>
          </div>
          <button id="queryButton" style="margin-top: 12px;">Run Query</button>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="full">
        <label for="uploadFile">Upload Document</label>
        <input id="uploadFile" type="file" />
        <p class="hint">Supported document upload endpoints use the same backend allowed extensions.</p>
        <button id="uploadButton">Upload File</button>
      </div>
    </div>

    <div class="panel row">
      <div class="full">
        <button id="listButton">Refresh Document List</button>
        <button id="healthButton">Check Health</button>
      </div>
      <div class="full">
        <h2>Response</h2>
        <pre id="responseArea">Ready.</pre>
      </div>
    </div>
  </div>

  <script>
    const apiKeyInput = document.getElementById('apiKey');
    const queryText = document.getElementById('queryText');
    const modeInput = document.getElementById('mode');
    const topKInput = document.getElementById('topK');
    const modelInput = document.getElementById('model');
    const streamToggle = document.getElementById('streamToggle');
    const responseArea = document.getElementById('responseArea');
    const uploadFile = document.getElementById('uploadFile');

    function buildHeaders() {
      const headers = {};
      const apiKey = apiKeyInput.value.trim();
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      return headers;
    }

    async function renderResult(result) {
      responseArea.textContent = JSON.stringify(result, null, 2);
    }

    function appendStreamContent(content) {
      const text = String(content ?? '');
      if (!text) return;

      const previous = responseArea.textContent;
      const needsSpace = previous && /[A-Za-z0-9]$/.test(previous) && /^[A-Za-z0-9]/.test(text);
      responseArea.textContent += (needsSpace ? ' ' : '') + text;
    }

    function appendSsePayload(rawData) {
      try {
        const data = JSON.parse(rawData);
        if (data && data.error) throw new Error(data.error);
        appendStreamContent(data && data.content !== undefined ? data.content : data);
      } catch (error) {
        if (!(error instanceof SyntaxError)) {
          throw error;
        }
        appendStreamContent(rawData);
      }
    }

    function processSseLine(line) {
      if (!line.startsWith('data: ')) return;
      const rawData = line.substring(6);
      if (rawData) appendSsePayload(rawData);
    }

    async function callQuery() {
      try {
        responseArea.textContent = 'Running query...';
        const isStreaming = streamToggle.checked;
        const payload = {
          query: queryText.value,
          mode: modeInput.value,
          top_k: Number(topKInput.value) || 3,
          model: modelInput.value || null,
          stream: isStreaming
        };

        const endpoint = isStreaming ? '/api/query/stream' : '/api/query/';
        
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...buildHeaders() },
          body: JSON.stringify(payload)
        });

        if (isStreaming) {
          if (!res.ok) {
            const errorBody = await res.text();
            throw new Error(`HTTP ${res.status}: ${errorBody}`);
          }
          if (!res.body) {
            throw new Error('The browser did not provide a response stream.');
          }

          responseArea.textContent = '';
          responseArea.classList.add('streaming-active');
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          try {
            while (true) {
              const { value, done } = await reader.read();
              buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
              const lines = buffer.split(/\\r?\\n/);
              buffer = lines.pop() || '';

              for (const line of lines) {
                processSseLine(line);
                responseArea.scrollTop = responseArea.scrollHeight;
              }

              if (done) break;
            }

            processSseLine(buffer.trim());
          } finally {
            responseArea.classList.remove('streaming-active');
          }
        } else {
          const body = await res.json();
          await renderResult({ status: res.status, body });
        }
      } catch (err) {
        responseArea.textContent = `Query failed: ${err}`;
      }
    }

    async function uploadDocument() {
      const file = uploadFile.files[0];
      if (!file) {
        responseArea.textContent = 'Please select a file to upload.';
        return;
      }
      try {
        responseArea.textContent = 'Uploading file...';
        const formData = new FormData();
        formData.append('files', file, file.name);
        const res = await fetch('/api/documents/upload', {
          method: 'POST',
          headers: buildHeaders(),
          body: formData
        });
        const body = await res.json();
        await renderResult({ status: res.status, body });
      } catch (err) {
        responseArea.textContent = `Upload failed: ${err}`;
      }
    }

    async function listDocuments() {
      try {
        responseArea.textContent = 'Loading documents...';
        const res = await fetch('/api/documents', { headers: buildHeaders() });
        const body = await res.json();
        await renderResult({ status: res.status, body });
      } catch (err) {
        responseArea.textContent = `List failed: ${err}`;
      }
    }

    async function checkHealth() {
      try {
        responseArea.textContent = 'Checking health...';
        const res = await fetch('/health');
        const body = await res.json();
        await renderResult({ status: res.status, body });
      } catch (err) {
        responseArea.textContent = `Health check failed: ${err}`;
      }
    }

    document.getElementById('queryButton').addEventListener('click', callQuery);
    document.getElementById('uploadButton').addEventListener('click', uploadDocument);
    document.getElementById('listButton').addEventListener('click', listDocuments);
    document.getElementById('healthButton').addEventListener('click', checkHealth);
  </script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def ui_page():
  return HTMLResponse(
    content=HTML_PAGE,
    headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
  )
