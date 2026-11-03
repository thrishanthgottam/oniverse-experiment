# ONIVERSE: near-duplicate funnel experiment

Code and raw results for the paper "ONIVERSE: A Hierarchical Architecture for
Vector Storage with Ingestion-Time Near-Duplicate Rejection".

## Reproduce

```bash
pip install -r requirements.txt

# 1. Get the Reuters-21578 corpus as distributed with NLTK
curl -L -o reuters.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip
unzip reuters.zip          # creates reuters/training and reuters/test

# 2. Run the experiment (5 seeds at N=5000, then a scaling sweep); writes results.json
python oniverse_experiment.py

# 3. Aggregate and draw figures; writes summary.json, fig_recall.png, fig_latency.png
python analyze.py
```

Runtime is a few minutes on a laptop-class CPU. Latency numbers depend on hardware;
the paper's numbers were measured on an Intel Xeon 2.3 GHz, Python 3.12.3.

## Files

| File | Purpose |
|---|---|
| oniverse_experiment.py | Builds Bloom filter, MinHash LSH, Infospheres; injects duplicates; evaluates configurations A to E |
| analyze.py | Aggregates results across seeds, prints tables, draws figures |
| results.json | Raw per-seed metrics, latencies and diagnostics |
| summary.json | Mean and standard deviation across seeds |

## Configurations

A Bloom only; B Bloom, MinHash, cosine; C MinHash LSH on all items;
D1 ONIVERSE as specified (centroid gate 0.95); D2 ONIVERSE revised (top-5 centroid routing);
E brute-force cosine.

## Notes

Embeddings are LSA (TF-IDF + 256-d truncated SVD) fitted on the store. Replacing the
`embed` step with a neural model (e.g. Sentence-BERT) is the main planned extension.
Dataset: Lewis, D. D. (1997), Reuters-21578 text categorization test collection.
