import asyncio
import json
import os
from pathlib import Path
from api.config import get_settings
from rag.indexing import DocumentIndexer
from rag.retrieval import HybridRetriever
from data.vector_store import get_vector_store
from tools.eval_utils import precision_at_k, recall_at_k, mrr, ndcg_at_k


async def build_or_load_queries(indexer: DocumentIndexer, queries_path: Path):
    if queries_path.exists():
        with open(queries_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    # Auto-generate simple queries from indexed metadata (first sentence of file)
    queries = []
    for meta in indexer.get_documents():
        fp = meta.get('file_path')
        doc_id = meta.get('doc_id')
        if not fp:
            continue
        try:
            text = Path(fp).read_text(encoding='utf-8', errors='ignore')
            snippet = text.strip().split('.')
            query_text = (snippet[0] + '.') if snippet and snippet[0] else meta.get('file_name', doc_id)
            queries.append({
                'query': query_text.strip(),
                'relevant': [doc_id],
            })
        except Exception:
            continue

    # Save generated queries
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queries_path, 'w', encoding='utf-8') as f:
        for q in queries:
            f.write(json.dumps(q) + '\n')
    return queries


async def run_eval(queries_file: str = './tools/eval_queries.jsonl', top_k: int = 10, alpha: float = 0.5):
    settings = get_settings()
    indexer = DocumentIndexer(settings.UPLOAD_DIR, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # Load documents and index into vector store
    docs = indexer.load_documents(settings.UPLOAD_DIR)

    vector_store = get_vector_store(store_type=settings.VECTOR_STORE_TYPE, persist_dir=settings.CHROMA_PERSIST_DIR)

    retriever = HybridRetriever(vector_store, use_reranker=False, reranker=None)
    await retriever.index_documents(docs)

    queries_path = Path(queries_file)
    queries = await build_or_load_queries(indexer, queries_path)

    results = []

    for q in queries:
        query_text = q['query']
        relevant = q.get('relevant', [])

        sem = await retriever.semantic_retrieve(query_text, top_k=top_k)
        bm = await retriever.keyword_retrieve(query_text, top_k=top_k)
        hybrid_list, _ = await retriever.retrieve(query_text, top_k=top_k, alpha=alpha)

        # Robust relevance matching: consider doc_id, metadata.file_name, or source
        def is_relevant_doc(doc, relevant_tokens):
            doc_id = getattr(doc, 'doc_id', '')
            meta = getattr(doc, 'metadata', {}) or {}
            file_name = meta.get('file_name', '') or meta.get('filename', '')
            source = meta.get('file_path', '') or meta.get('source', '')
            for token in relevant_tokens:
                if not token:
                    continue
                if token in doc_id or token in file_name or token in source:
                    return True
            return False

        sem_ids = [d.doc_id for d in sem]
        bm_ids = [d.doc_id for d in bm]
        hybrid_ids = [d.doc_id for d in hybrid_list]

        # Convert retrieved lists to include metadata-aware relevance for metrics
        sem_relevant_flags = [is_relevant_doc(d, relevant) for d in sem]
        bm_relevant_flags = [is_relevant_doc(d, relevant) for d in bm]
        hybrid_relevant_flags = [is_relevant_doc(d, relevant) for d in hybrid_list]

        res = {
            'query': query_text,
            'relevant': relevant,
            'semantic': sem_ids,
            'bm25': bm_ids,
            'hybrid': hybrid_ids,
            'metrics': {
                'semantic': {
                    'p@5': sum(1 for x in sem_relevant_flags[:5] if x) / 5,
                    'recall@5': (sum(1 for x in sem_relevant_flags[:5] if x) / len(relevant)) if relevant else 0.0,
                    'mrr': next((1.0/(i+1) for i, x in enumerate(sem_relevant_flags) if x), 0.0),
                    'ndcg@5': ndcg_at_k(sem_ids, relevant, 5),
                },
                'bm25': {
                    'p@5': sum(1 for x in bm_relevant_flags[:5] if x) / 5,
                    'recall@5': (sum(1 for x in bm_relevant_flags[:5] if x) / len(relevant)) if relevant else 0.0,
                    'mrr': next((1.0/(i+1) for i, x in enumerate(bm_relevant_flags) if x), 0.0),
                    'ndcg@5': ndcg_at_k(bm_ids, relevant, 5),
                },
                'hybrid': {
                    'p@5': sum(1 for x in hybrid_relevant_flags[:5] if x) / 5,
                    'recall@5': (sum(1 for x in hybrid_relevant_flags[:5] if x) / len(relevant)) if relevant else 0.0,
                    'mrr': next((1.0/(i+1) for i, x in enumerate(hybrid_relevant_flags) if x), 0.0),
                    'ndcg@5': ndcg_at_k(hybrid_ids, relevant, 5),
                },
            }
        }
        results.append(res)

    out_path = Path('./tools/eval_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print summary
    summary = {'semantic': {'p@5': [], 'mrr': []}, 'bm25': {'p@5': [], 'mrr': []}, 'hybrid': {'p@5': [], 'mrr': []}}
    for r in results:
        for m in ('semantic', 'bm25', 'hybrid'):
            summary[m]['p@5'].append(r['metrics'][m]['p@5'])
            summary[m]['mrr'].append(r['metrics'][m]['mrr'])

    final = {}
    for m in summary:
        final[m] = {
            'mean_p@5': sum(summary[m]['p@5']) / len(summary[m]['p@5']) if summary[m]['p@5'] else 0,
            'mean_mrr': sum(summary[m]['mrr']) / len(summary[m]['mrr']) if summary[m]['mrr'] else 0,
        }

    print('Evaluation complete. Summary:')
    print(json.dumps(final, indent=2))
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    asyncio.run(run_eval())
