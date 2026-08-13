import asyncio
import shutil
from pathlib import Path
from api.config import get_settings
from rag.indexing import DocumentIndexer
from tools import eval_retrieval


async def seed_and_run():
    settings = get_settings()
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Candidate files from repo to seed (fallback)
    repo_docs = [
        Path(__file__).parent.parent / 'PHASE_1_SETUP.md',
        Path(__file__).parent.parent / 'ARCHITECTURE.md',
    ]

    seeded = []
    for src in repo_docs:
        try:
            if src.exists():
                dest = upload_dir / src.name
                # Copy if not already present
                if not dest.exists():
                    shutil.copy(src, dest)
                seeded.append(dest)
        except Exception as e:
            print(f"Failed to seed {src}: {e}")

    indexer = DocumentIndexer(str(upload_dir), chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # Register seeded files in metadata
    if seeded:
        indexer.add_documents([str(p) for p in seeded])

    # Build queries file from these documents and run eval
    await eval_retrieval.run_eval()


if __name__ == '__main__':
    asyncio.run(seed_and_run())
