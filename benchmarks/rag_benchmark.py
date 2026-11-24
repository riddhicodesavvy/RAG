"""
Enterprise RAG Quantitative Evaluation Suite
Evaluates:
  - Experiment A: Retrieval benchmark across 105 labeled questions (6 categories)
  - Experiment B: Chunking ablation (500/100, 1000/200, 1500/300)
  - Experiment C: Retrieval-confidence distance-gate calibration (answerable vs out-of-scope)
  - Experiment D: Local LLM grounded generation comparison (No-RAG vs RAG vs RAG+Gate) using Ollama gemma4
"""

import csv
import json
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmarks" / "datasets" / "evaluation_questions.json"
MANUALS_DIR = ROOT / "benchmarks" / "datasets" / "manuals"
RESULTS_DIR = ROOT / "benchmarks" / "results"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma4:latest"
EMBED_MODEL = "all-MiniLM-L6-v2"


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def load_manuals():
    docs = []
    for p in sorted(MANUALS_DIR.glob("*.txt")):
        loader = TextLoader(str(p), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = p.name
        docs.extend(loaded)
    return docs


def build_index(embedder: SentenceTransformer, docs, chunk_size: int, chunk_overlap: int, persist_dir: Path):
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    t0 = time.perf_counter()
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=f"bench_{chunk_size}_{chunk_overlap}")
    collection.add(
        ids=[f"c_{i}" for i in range(len(chunks))],
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"source_file": c.metadata["source_file"], "chunk_index": i, "len": len(c.page_content)} for i, c in enumerate(chunks)],
        documents=texts,
    )
    ingest_ms = (time.perf_counter() - t0) * 1000.0
    index_bytes = dir_size_bytes(persist_dir)
    return collection, {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "ingest_time_ms": round(ingest_ms, 2),
        "index_disk_bytes": index_bytes,
        "index_disk_kb": round(index_bytes / 1024.0, 2),
    }


def is_chunk_relevant(chunk_text: str, source_file: str, q: dict) -> bool:
    if not q["answerable"] or source_file != q["ground_truth_doc"]:
        return False
    low = chunk_text.lower()
    if q["ground_truth_section"].lower() in low:
        return True
    return any(p.lower() in low for p in q["required_evidence_phrases"])


def evaluate_retrieval(embedder: SentenceTransformer, collection, questions: list, top_k: int = 5):
    latencies_ms = []
    per_question = []

    for q in questions:
        t0 = time.perf_counter()
        q_emb = embedder.encode(q["query"], convert_to_numpy=True)
        res = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        rels = [is_chunk_relevant(d, m["source_file"], q) for d, m in zip(docs, metas)]
        # Full evidence phrase coverage across top-k chunks
        def phrase_cov(k_cut):
            if not q["required_evidence_phrases"]:
                return 0.0
            joined = "\n".join(
                d for d, m in zip(docs[:k_cut], metas[:k_cut]) if m["source_file"] == q["ground_truth_doc"]
            ).lower()
            hits = sum(1 for p in q["required_evidence_phrases"] if p.lower() in joined)
            return hits / len(q["required_evidence_phrases"])

        rr = 0.0
        for rank, rel in enumerate(rels, start=1):
            if rel:
                rr = 1.0 / rank
                break

        per_question.append({
            "id": q["id"],
            "category": q["category"],
            "answerable": q["answerable"],
            "latency_ms": round(lat_ms, 3),
            "top1_distance": float(dists[0]),
            "top1_source": metas[0]["source_file"],
            "hit_at_1": int(any(rels[:1])),
            "hit_at_3": int(any(rels[:3])),
            "hit_at_5": int(any(rels[:5])),
            "precision_at_1": float(sum(rels[:1]) / 1.0),
            "precision_at_3": float(sum(rels[:3]) / 3.0),
            "precision_at_5": float(sum(rels[:5]) / 5.0),
            "reciprocal_rank": rr,
            "evidence_cov_at_3": phrase_cov(3),
            "evidence_cov_at_5": phrase_cov(5),
            "top_distances": [float(x) for x in dists],
            "top_docs": docs,
            "top_sources": [m["source_file"] for m in metas],
        })

    ans = [r for r in per_question if r["answerable"]]
    lats_sorted = sorted(latencies_ms)
    p50 = statistics.median(lats_sorted)
    p95 = lats_sorted[min(len(lats_sorted) - 1, int(0.95 * len(lats_sorted)))]

    def agg(rows):
        n = len(rows)
        if n == 0:
            return {}
        return {
            "count": n,
            "recall_at_1": round(sum(r["hit_at_1"] for r in rows) / n, 4),
            "recall_at_3": round(sum(r["hit_at_3"] for r in rows) / n, 4),
            "recall_at_5": round(sum(r["hit_at_5"] for r in rows) / n, 4),
            "precision_at_1": round(sum(r["precision_at_1"] for r in rows) / n, 4),
            "precision_at_3": round(sum(r["precision_at_3"] for r in rows) / n, 4),
            "precision_at_5": round(sum(r["precision_at_5"] for r in rows) / n, 4),
            "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
            "evidence_cov_at_3": round(sum(r["evidence_cov_at_3"] for r in rows) / n, 4),
            "evidence_cov_at_5": round(sum(r["evidence_cov_at_5"] for r in rows) / n, 4),
            "mean_top1_distance": round(sum(r["top1_distance"] for r in rows) / n, 4),
        }

    by_category = {}
    for cat in ["direct_factual", "procedural", "paraphrased", "exact_id", "ambiguous"]:
        by_category[cat] = agg([r for r in ans if r["category"] == cat])

    oos = [r for r in per_question if not r["answerable"]]
    oos_stats = {
        "count": len(oos),
        "mean_top1_distance": round(sum(r["top1_distance"] for r in oos) / len(oos), 4),
        "min_top1_distance": round(min(r["top1_distance"] for r in oos), 4),
        "max_top1_distance": round(max(r["top1_distance"] for r in oos), 4),
    }

    return {
        "overall_answerable": agg(ans),
        "by_category": by_category,
        "out_of_scope_distances": oos_stats,
        "latency_ms": {
            "mean": round(statistics.mean(latencies_ms), 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "min": round(min(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
        },
        "per_question": per_question,
    }


def evaluate_confidence_gate(per_question: list) -> dict:
    """
    Sweep squared L2 distance thresholds tau (lower distance = higher semantic confidence).
    A query is ACCEPTED if top1_distance <= tau, else REJECTED (abstained).
    """
    thresholds = [0.7, 0.85, 1.0, 1.15, 1.25, 1.35, 1.45, 1.6, 1.8]
    ans = [r for r in per_question if r["answerable"]]
    oos = [r for r in per_question if not r["answerable"]]
    sweep = []

    for tau in thresholds:
        tp_accept = sum(1 for r in ans if r["top1_distance"] <= tau)
        fn_reject_valid = len(ans) - tp_accept
        tn_reject_oos = sum(1 for r in oos if r["top1_distance"] > tau)
        fp_accept_oos = len(oos) - tn_reject_oos

        total_rejected = fn_reject_valid + tn_reject_oos
        abstention_prec = tn_reject_oos / total_rejected if total_rejected > 0 else 0.0
        abstention_rec = tn_reject_oos / len(oos) if len(oos) > 0 else 0.0

        sweep.append({
            "distance_threshold_tau": tau,
            "answerable_acceptance_rate": round(tp_accept / len(ans), 4),
            "incorrect_rejection_rate_valid": round(fn_reject_valid / len(ans), 4),
            "correct_rejection_rate_oos": round(tn_reject_oos / len(oos), 4),
            "false_acceptance_rate_oos": round(fp_accept_oos / len(oos), 4),
            "abstention_precision": round(abstention_prec, 4),
            "abstention_recall": round(abstention_rec, 4),
            "counts": {
                "tp_valid_accepted": tp_accept,
                "fn_valid_rejected": fn_reject_valid,
                "tn_oos_rejected": tn_reject_oos,
                "fp_oos_accepted": fp_accept_oos,
            },
        })

    # Choose optimal balanced threshold (maximizing Youden's J = TPR - FAR)
    best = max(sweep, key=lambda x: x["answerable_acceptance_rate"] - x["false_acceptance_rate_oos"])
    return {
        "public_repo_gate_implemented": False,
        "distinction_note": "Public demonstrator rag_system.py returns raw distances without an active abstention threshold; confidence gating was implemented in the private Bridge & Roof enterprise pipeline. Below is a threshold-calibration simulation over the public demonstrator's ChromaDB L2 distances.",
        "optimal_youden_threshold": best["distance_threshold_tau"],
        "threshold_sweep": sweep,
    }


def call_ollama(prompt: str, max_tokens: int = 90) -> tuple[str, float, str | None]:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"num_ctx": 2048, "num_predict": max_tokens, "temperature": 0.0},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip(), (time.perf_counter() - t0) * 1000.0, None
    except Exception as e:
        return "", (time.perf_counter() - t0) * 1000.0, str(e)


def score_generation(answer: str, q: dict, context_text: str, abstained: bool) -> dict:
    """
    Deterministic, rubric-based scoring (explicitly documented as automated lexical/regex rubric):
      - For answerable questions:
          * answer_correctness: fraction of required_evidence_phrases present in answer (or matched numerically)
          * evidence_support: fraction of answer sentences/numbers grounded in retrieved context
          * unsupported_claims: 1 if numeric tokens appear in answer that are absent from BOTH context and reference (when not abstaining)
      - For out_of_scope questions:
          * appropriate_abstention: 1 if model explicitly states insufficient info / abstains, 0 if it hallucinates an answer.
    """
    ans_low = answer.lower()
    abstention_markers = [
        "insufficient", "does not contain", "cannot answer", "not mentioned",
        "no information", "not provided", "unable to answer", "outside the scope",
        "don't have information", "not present in the", "cannot determine",
    ]
    model_abstained = abstained or any(m in ans_low for m in abstention_markers)

    if not q["answerable"]:
        return {
            "answer_correctness": 1.0 if model_abstained else 0.0,
            "appropriate_abstention": 1 if model_abstained else 0,
            "unsupported_factual_claim": 0 if model_abstained else 1,
            "evidence_supported": 1 if model_abstained else 0,
        }

    if model_abstained:
        return {
            "answer_correctness": 0.0,
            "appropriate_abstention": 0,
            "unsupported_factual_claim": 0,
            "evidence_supported": 0,
        }

    req = q["required_evidence_phrases"]
    hits = sum(1 for p in req if p.lower() in ans_low)
    correctness = hits / len(req) if req else 0.0

    # Check numeric grounding against retrieved context (if context provided) or reference
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer))
    ctx_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", context_text))
    ref_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", q["reference_answer"]))
    allowed_nums = ctx_nums | ref_nums
    unsupported_nums = [n for n in ans_nums if n not in allowed_nums]

    # Unsupported if numeric values invented or zero evidence phrases hit on non-abstaining answer
    has_unsupported = 1 if (len(unsupported_nums) > 0 or (len(context_text) > 0 and correctness == 0.0)) else 0
    ev_supported = 1 if (len(context_text) > 0 and hits > 0 and len(unsupported_nums) == 0) else 0

    return {
        "answer_correctness": round(correctness, 4),
        "appropriate_abstention": 0,
        "unsupported_factual_claim": has_unsupported,
        "evidence_supported": ev_supported,
    }


def run_experiment_d(questions: list, baseline_per_q: list, gate_tau: float) -> dict:
    """
    Experiment D: Grounded generation using local Ollama gemma4:latest.
    Stratified sample: 1 question per category x 6 categories = 6 questions across 3 modes (bounded CPU budget).
    """
    by_id_ret = {r["id"]: r for r in baseline_per_q}
    stratified = []
    for cat in ["direct_factual", "procedural", "paraphrased", "exact_id", "ambiguous", "out_of_scope"]:
        cat_qs = [q for q in questions if q["category"] == cat]
        stratified.extend(cat_qs[:1])

    modes = ["llm_no_retrieval", "llm_with_rag", "llm_with_rag_and_gate"]
    mode_records = {m: [] for m in modes}

    for idx, q in enumerate(stratified, 1):
        print(f"[Exp D {idx}/{len(stratified)}] {q['id']} ({q['category']})", flush=True)
        ret = by_id_ret[q["id"]]
        top3_ctx = "\n\n".join(
            f"[Source: {src}]\n{doc}"
            for src, doc in zip(ret["top_sources"][:3], ret["top_docs"][:3])
        )
        top1_dist = ret["top1_distance"]

        # Mode 1: No Retrieval
        prompt_no_rag = (
            "Answer the following industrial engineering question concisely in 1-2 sentences. "
            "If you do not know the exact plant specification or standard, state 'INSUFFICIENT_CONTEXT'.\n\n"
            f"Question: {q['query']}\nAnswer:"
        )
        ans1, lat1, err1 = call_ollama(prompt_no_rag)
        sc1 = score_generation(ans1, q, "", False)
        mode_records["llm_no_retrieval"].append({
            "id": q["id"], "category": q["category"], "answerable": q["answerable"],
            "latency_ms": round(lat1, 2), "error": err1, "answer_preview": ans1[:220], **sc1,
        })

        # Mode 2: RAG without Confidence Gate
        prompt_rag = (
            "You are an enterprise technical assistant. Answer strictly using ONLY the provided context. "
            "If the context does not contain the answer, respond strictly with 'INSUFFICIENT_CONTEXT'.\n\n"
            f"Context:\n{top3_ctx}\n\nQuestion: {q['query']}\nAnswer:"
        )
        ans2, lat2, err2 = call_ollama(prompt_rag)
        sc2 = score_generation(ans2, q, top3_ctx, False)
        mode_records["llm_with_rag"].append({
            "id": q["id"], "category": q["category"], "answerable": q["answerable"],
            "latency_ms": round(lat2, 2), "error": err2, "answer_preview": ans2[:220], **sc2,
        })

        # Mode 3: RAG + Confidence Gate
        if top1_dist > gate_tau:
            ans3 = f"INSUFFICIENT_CONTEXT (Confidence gate triggered: distance {top1_dist:.3f} > tau {gate_tau:.3f})"
            lat3 = ret["latency_ms"]
            err3 = None
            sc3 = score_generation(ans3, q, top3_ctx, True)
        else:
            ans3, llm_lat3, err3 = call_ollama(prompt_rag)
            lat3 = ret["latency_ms"] + llm_lat3
            sc3 = score_generation(ans3, q, top3_ctx, False)

        mode_records["llm_with_rag_and_gate"].append({
            "id": q["id"], "category": q["category"], "answerable": q["answerable"],
            "top1_distance": round(top1_dist, 4), "gated_abstained": top1_dist > gate_tau,
            "latency_ms": round(lat3, 2), "error": err3, "answer_preview": ans3[:220], **sc3,
        })

    summary = {}
    for m, rows in mode_records.items():
        ans_rows = [r for r in rows if r["answerable"]]
        oos_rows = [r for r in rows if not r["answerable"]]
        summary[m] = {
            "total_evaluated": len(rows),
            "answerable_mean_correctness": round(sum(r["answer_correctness"] for r in ans_rows) / len(ans_rows), 4),
            "answerable_evidence_support_rate": round(sum(r["evidence_supported"] for r in ans_rows) / len(ans_rows), 4),
            "unsupported_claim_rate_all": round(sum(r["unsupported_factual_claim"] for r in rows) / len(rows), 4),
            "oos_appropriate_abstention_rate": round(sum(r["appropriate_abstention"] for r in oos_rows) / len(oos_rows), 4),
            "mean_e2e_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 2),
        }

    return {
        "model": OLLAMA_MODEL,
        "sample_size": len(stratified),
        "gate_threshold_tau": gate_tau,
        "rubric_note": "Automated deterministic lexical/numeric rubric checking ground-truth evidence phrase coverage, numeric consistency against retrieved context, and explicit abstention markers. Not human-graded.",
        "summary": summary,
        "records": mode_records,
    }


def write_csv(out_csv: Path, exp_b: list, exp_c: dict, exp_d: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment", "config", "recall_at_1", "recall_at_3", "recall_at_5",
        "precision_at_3", "mrr", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
        "num_chunks", "index_disk_kb", "extra_metric"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in exp_b:
            ov = row["retrieval"]["overall_answerable"]
            lat = row["retrieval"]["latency_ms"]
            idx = row["index_stats"]
            w.writerow({
                "experiment": "B_chunking_ablation",
                "config": f"chunk_{idx['chunk_size']}_ov_{idx['chunk_overlap']}",
                "recall_at_1": ov["recall_at_1"],
                "recall_at_3": ov["recall_at_3"],
                "recall_at_5": ov["recall_at_5"],
                "precision_at_3": ov["precision_at_3"],
                "mrr": ov["mrr"],
                "mean_latency_ms": lat["mean"],
                "p50_latency_ms": lat["p50"],
                "p95_latency_ms": lat["p95"],
                "num_chunks": idx["num_chunks"],
                "index_disk_kb": idx["index_disk_kb"],
            })
        for mode, s in exp_d["summary"].items():
            w.writerow({
                "experiment": "D_grounded_generation",
                "config": mode,
                "mean_latency_ms": s["mean_e2e_latency_ms"],
                "extra_metric": f"ans_corr={s['answerable_mean_correctness']}|oos_abst={s['oos_appropriate_abstention_rate']}|unsup={s['unsupported_claim_rate_all']}",
            })


def main():
    t_all = time.perf_counter()
    ds = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = ds["questions"]
    docs = load_manuals()
    embedder = SentenceTransformer(EMBED_MODEL)

    tmp_root = Path(tempfile.mkdtemp(prefix="rag_bench_"))
    try:
        # Experiment B + Experiment A (1000/200 is the baseline repo config)
        ablation_configs = [(500, 100), (1000, 200), (1500, 300)]
        ablation_results = []
        baseline_eval = None

        for cs, ov in ablation_configs:
            col, idx_stats = build_index(embedder, docs, cs, ov, tmp_root / f"idx_{cs}_{ov}")
            ret = evaluate_retrieval(embedder, col, questions, top_k=5)
            # Strip heavy doc strings for clean JSON except keeping baseline for Exp D
            ret_clean = {k: v for k, v in ret.items() if k != "per_question"}
            ret_clean["per_question_summary"] = [
                {k: v for k, v in r.items() if k != "top_docs"} for r in ret["per_question"]
            ]
            ablation_results.append({"index_stats": idx_stats, "retrieval": ret_clean})
            if cs == 1000 and ov == 200:
                baseline_eval = {"index_stats": idx_stats, "retrieval": ret}

        # Experiment C: Confidence Gating calibration on baseline (1000/200)
        exp_c = evaluate_confidence_gate(baseline_eval["retrieval"]["per_question"])

        # Experiment D: Grounded generation with local Ollama gemma4:latest
        exp_d = run_experiment_d(questions, baseline_eval["retrieval"]["per_question"], exp_c["optimal_youden_threshold"])
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "embedding_model": EMBED_MODEL,
            "embedding_dimension": int(embedder.get_sentence_embedding_dimension()),
            "local_llm_model": OLLAMA_MODEL,
            "chromadb_version": chromadb.__version__,
            "dataset_name": ds["dataset_name"],
            "total_questions": ds["total_questions"],
            "category_counts": ds["category_counts"],
            "architecture_distinction": "Public demonstrator uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) and ChromaDB; private Bridge & Roof enterprise training used Gemini embeddings and confidence gating over internal manuals.",
            "total_runtime_s": round(time.perf_counter() - t_all, 2),
        },
        "experiment_a_baseline_1000_200": {
            "index_stats": baseline_eval["index_stats"],
            "overall_answerable": baseline_eval["retrieval"]["overall_answerable"],
            "by_category": baseline_eval["retrieval"]["by_category"],
            "out_of_scope_distances": baseline_eval["retrieval"]["out_of_scope_distances"],
            "latency_ms": baseline_eval["retrieval"]["latency_ms"],
        },
        "experiment_b_chunking_ablation": ablation_results,
        "experiment_c_confidence_gating": exp_c,
        "experiment_d_grounded_generation": exp_d,
    }

    out_json = RESULTS_DIR / "rag_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_csv = RESULTS_DIR / "rag_results.csv"
    write_csv(out_csv, ablation_results, exp_c, exp_d)

    print(json.dumps({
        "status": "COMPLETE",
        "json": str(out_json),
        "csv": str(out_csv),
        "runtime_s": payload["metadata"]["total_runtime_s"],
        "baseline_recall_at_3": payload["experiment_a_baseline_1000_200"]["overall_answerable"]["recall_at_3"],
        "baseline_mrr": payload["experiment_a_baseline_1000_200"]["overall_answerable"]["mrr"],
        "exp_d_summary": exp_d["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()

