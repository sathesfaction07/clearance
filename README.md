# Clearance

An enterprise handbook assistant that only ever retrieves what the person asking is allowed
to read. Built on 774 real pages from GitLab's public handbook (people policies, total
rewards, finance, legal, security and support), with a deliberately unforgiving rule: a
restricted page must not reach the wrong person's results, not even as a hint.

**Status: phase 1 of 3.** Ingestion, chunking, keyword retrieval with permissions and the
leak tests are built and measured. Next: semantic retrieval (Gemini embeddings in pgvector),
the answering agent on ADK, and deployment on Kubernetes (GKE, tested on a kind cluster in
CI). This README states only what is built and measured.

## The problem

A composite scenario, not a named company. The customer's first assistant indexed everything and answered everyone. In testing it
quoted a pending legal settlement to a support engineer. The fix that had been proposed,
removing restricted results after search, looked safe in a demo.

## What is measured so far

[Full report](reports/retrieval.md), regenerated in CI from the live handbook on every push.

**Where the permission check sits** (33 questions × 6 users, top 5 results):

| Filter | Restricted chunks returned | Searches cut short |
|:--|--:|--:|
| Before ranking (shipped) | **0** | **0** |
| After ranking (the proposed fix) | 0 | **116 of 198** |
| None (the first assistant) | **480** | 0 |

Removing results after ranking does not leak text, but in 116 of 198 searches restricted pages
took the top places and were then removed, so users with every right to an answer got fewer
results. The shortened list also reveals that something matched. Filtering before ranking
means restricted chunks are never scored at all.

**Chunking** (same questions, BM25): the source page is in the top 5 for 94-97% of questions
under all three strategies. Titled heading chunks score highest (97%, MRR 0.86), but with 33
questions a difference of one or two questions is noise. The question set was written from
page headings, which favours keyword search. Paraphrased questions and semantic retrieval are
the next comparison.

**Messy data:** 149 of 923 pages (16%) are stubs that only link to an internal handbook. They
are dropped at ingestion, so they cannot fill results with titles and no content.

## Access groups

The handbook is public, so the groups are invented for testing (`config/access.yaml`). Each
section belongs to one group: people policies and total rewards to all staff, and support,
security, legal and finance to their own teams. Six test users hold different combinations.
Unknown users can read nothing.

## Running it

```bash
pip install -e ".[dev]"
pytest                               # fixture corpus, no download needed
python scripts/fetch_handbook.py     # six handbook sections, about 2 MB compressed
python scripts/evaluate.py           # writes reports/retrieval.md; fails if anything leaks
```

## Data

GitLab handbook content, from gitlab.com/gitlab-com/content-sites/handbook. It is downloaded
at run time and not redistributed here. Only the question set and page paths are committed.
