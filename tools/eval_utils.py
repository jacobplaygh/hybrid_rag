import math

def precision_at_k(retrieved, relevant, k):
    ret = retrieved[:k]
    if k == 0:
        return 0.0
    return len(set(ret) & set(relevant)) / k


def recall_at_k(retrieved, relevant, k):
    ret = retrieved[:k]
    if not relevant:
        return 0.0
    return len(set(ret) & set(relevant)) / len(relevant)


def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains):
    return sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(retrieved, relevant, k):
    gains = [1 if d in relevant else 0 for d in retrieved[:k]]
    if not any(gains):
        return 0.0
    ideal = sorted(gains, reverse=True)
    return dcg(gains) / dcg(ideal)
