# Keyword, semantic and hybrid retrieval, and filtering in pgvector

Corpus: 7544 `headings_titled` chunks from 774 handbook pages (the handbook pinned to one commit). Embeddings: `embeddinggemma`, 768 dimensions, computed once locally and committed. Each question is asked by the least-privileged user allowed to read its source. Regenerate with `python scripts/evaluate_semantic.py`.

## Does the right page come back?

Semantic: exact cosine search over the chunks the asker may read. Hybrid: reciprocal rank fusion of the top 20 keyword and top 20 semantic chunks.

| Questions | Retrieval | Source page in top 1 | Top 3 | Top 5 | Mean reciprocal rank |
|:--|:--|--:|--:|--:|--:|
| As written (33) | keyword | 79% | 97% | 97% | 0.88 |
| As written (33) | semantic | 88% | 100% | 100% | 0.94 |
| As written (33) | hybrid | 85% | 100% | 100% | 0.92 |
| Paraphrased (33) | keyword | 9% | 24% | 36% | 0.22 |
| Paraphrased (33) | semantic | 58% | 85% | 88% | 0.72 |
| Paraphrased (33) | hybrid | 33% | 70% | 79% | 0.51 |

Paired on the same questions (top 5): how often one finds the page and the other does not, with a two-sided sign test.

| Questions | Comparison | Only the first finds it | Only the second | p |
|:--|:--|--:|--:|--:|
| As written | semantic vs keyword | 1 | 0 | 1.00 |
| As written | hybrid vs keyword | 1 | 0 | 1.00 |
| As written | hybrid vs semantic | 0 | 0 | 1.00 |
| Paraphrased | semantic vs keyword | 18 | 1 | 0.00 |
| Paraphrased | hybrid vs keyword | 15 | 1 | 0.00 |
| Paraphrased | hybrid vs semantic | 1 | 4 | 0.38 |

## Whether unreadable pages change hybrid results

Every question as every user (396 searches, top 5), full index against an index of only the pages that user may read.

**0 of 396** hybrid searches differ. Keyword statistics are scoped to readable pages and a cosine similarity depends on nothing but the question and the chunk, so the fused ranking is exactly what the user would get if restricted pages did not exist.

## Filtering inside pgvector

pgvector 0.6.2; 7544 chunks loaded and indexed (HNSW, m 16, ef_construction 64) in 2.4 s. Every question as every user, top 5, `hnsw.ef_search` 40 (the default). The filter is a `WHERE` clause on the chunk's group in every mode, so nothing restricted can be returned; what differs is whether users get their full results.

| Mode | Searches | Restricted returned | Fewer than 5 results | Same results as exact | Median ms | p95 ms |
|:--|--:|--:|--:|--:|--:|--:|
| Exact (no vector index) | 396 | 0 | 0 | 100.0% | 13.5 | 36.2 |
| Postgres's own choice | 396 | 0 | 154 | 63.5% | 0.8 | 1.0 |
| HNSW index, filter applied to what it finds | 396 | 0 | 154 | 63.5% | 0.9 | 1.1 |

The HNSW graph search finds the 40 nearest chunks of any group, and the filter then removes those the user may not read. Only 8% of chunks are readable by everyone, so for most questions the nearest 40 hold too few readable chunks: users are starved exactly as with filtering after ranking. The iterative scan keeps walking the graph until 5 readable chunks are found.

Plan Postgres chose by itself for the least-privileged user:

```
Limit  (cost=132.48..138.56 rows=5 width=73)
  ->  Index Scan using chunks_embedding_idx on chunks  (cost=132.48..892.48 rows=625 width=73)
        Order By: (embedding <=> '[question vector]'::vector)
        Filter: (grp = ANY ('{all-staff}'::text[]))
```
