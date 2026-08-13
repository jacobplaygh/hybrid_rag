import io
import shutil
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from api.routes import documents as documents_module


def reset_document_route_state():
    documents_module._indexer = None
    documents_module._vector_store = None


def test_upload_list_and_delete_document_flow():
    upload_dir = Path("./uploaded_docs")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    reset_document_route_state()

    client = TestClient(app)

    payload = {"files": ("sample.txt", b"hello from the backend", "text/plain")}

    response = client.post(
        "/api/documents/upload",
        files=payload,
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["uploaded"] == 1

    list_response = client.get("/api/documents/list", headers={"X-API-Key": "test-key"})
    assert list_response.status_code == 200
    docs = list_response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "sample.txt"

    stats_response = client.get("/api/documents/stats", headers={"X-API-Key": "test-key"})
    assert stats_response.status_code == 200
    assert stats_response.json()["total_documents"] >= 1

    doc_id = docs[0]["doc_id"]
    delete_response = client.delete(f"/api/documents/{doc_id}", headers={"X-API-Key": "test-key"})
    assert delete_response.status_code == 200

    after_delete = client.get("/api/documents/list", headers={"X-API-Key": "test-key"})
    assert after_delete.status_code == 200
    assert len(after_delete.json()["documents"]) == 0


def test_document_list_includes_num_chunks():
    upload_dir = Path("./uploaded_docs")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    reset_document_route_state()

    client = TestClient(app)

    payload = {"files": ("sample.txt", b"hello from the backend " * 20, "text/plain")}
    response = client.post(
        "/api/documents/upload",
        files=payload,
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["uploaded"] == 1

    list_response = client.get("/api/documents/list", headers={"X-API-Key": "test-key"})
    assert list_response.status_code == 200
    docs = list_response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "sample.txt"
    assert "num_chunks" in docs[0]
    assert docs[0]["num_chunks"] >= 1


def test_collection_root_lists_documents():
    upload_dir = Path("./uploaded_docs")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    reset_document_route_state()

    client = TestClient(app)

    response = client.get("/api/documents", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
    body = response.json()
    assert "documents" in body
    assert body["total"] == 0


def test_upload_epub_document():
    upload_dir = Path("./uploaded_docs")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    reset_document_route_state()

    # Create a minimal EPUB file in memory
    epub_bytes = io.BytesIO()
    with zipfile.ZipFile(epub_bytes, mode="w") as zf:
        zf.writestr("META-INF/container.xml", "<?xml version=\"1.0\"?><container version=\"1.0\" xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\"><rootfiles><rootfile full-path=\"OEBPS/content.opf\" media-type=\"application/oebps-package+xml\"/></rootfiles></container>")
        zf.writestr("OEBPS/content.opf", "<?xml version=\"1.0\"?><package version=\"2.0\" xmlns=\"http://www.idpf.org/2007/opf\"><manifest><item id=\"chap1\" href=\"chapter1.xhtml\" media-type=\"application/xhtml+xml\"/></manifest><spine><itemref idref=\"chap1\"/></spine></package>")
        zf.writestr("OEBPS/chapter1.xhtml", "<html><body><p>Hello EPUB world!</p></body></html>")
    epub_bytes.seek(0)

    client = TestClient(app)

    response = client.post(
        "/api/documents/upload",
        files={"files": ("sample.epub", epub_bytes.read(), "application/epub+zip")},
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["uploaded"] == 1
    assert body["failed"] == 0
    assert body["files"][0]["filename"] == "sample.epub"
