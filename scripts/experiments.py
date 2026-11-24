"""Create the predeclared ablation configs. No benchmark scores are consulted."""

from pathlib import Path
import json

configs = {
    "R00_bm25_latest_rerank": dict(
        backends=["bm25"],
        query_mode="latest",
        reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
        retrieval_rank_weight=0,
    ),
    "E00_bm25_latest": dict(backends=["bm25"], query_mode="latest"),
    "E01_bm25_history": dict(backends=["bm25"], query_mode="history"),
    "E02_bm25_rewrite": dict(backends=["bm25"], query_mode="rewrite"),
    "E03_splade_latest": dict(backends=["splade"], query_mode="latest"),
    "E04_splade_rewrite": dict(backends=["splade"], query_mode="rewrite"),
    "E05_splade_fusion": dict(backends=["splade"], query_mode="fusion"),
    "E06_splade_rerank": dict(
        backends=["splade"],
        query_mode="fusion",
        reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
        retrieval_rank_weight=0,
    ),
    "E07_splade_blend": dict(
        backends=["splade"],
        query_mode="fusion",
        reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
        retrieval_rank_weight=0.5,
    ),
    "E08_hybrid_fusion": dict(backends=["splade", "dense"], query_mode="fusion"),
    "E10_keyword_fusion": dict(backends=["splade"], query_mode="keyword"),
    "D00_dense_latest": dict(backends=["dense"], query_mode="latest"),
    "D01_dense_fusion": dict(backends=["dense"], query_mode="fusion"),
    "B02_bm25_fusion": dict(backends=["bm25"], query_mode="fusion"),
    "B03_bm25_rerank": dict(
        backends=["bm25"],
        query_mode="fusion",
        reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
        retrieval_rank_weight=0,
    ),
    "B04_bm25_blend": dict(
        backends=["bm25"],
        query_mode="fusion",
        reranker="cross-encoder/ms-marco-MiniLM-L6-v2",
        retrieval_rank_weight=0.5,
    ),
}
if __name__ == "__main__":
    Path("configs").mkdir(exist_ok=True)
    for name, settings in configs.items():
        cfg = dict(
            id=name,
            dataset="mtrag",
            split="validation",
            retrieve_k=100,
            pool_size=100,
            rrf_k=60,
            **settings,
        )
        if cfg["query_mode"] not in ("latest", "history"):
            cfg["rewrites"] = "cache/rewrites/mtrag.jsonl"
        Path("configs", name + ".json").write_text(json.dumps(cfg, indent=2) + "\n")
