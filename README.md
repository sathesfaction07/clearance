# Clearance

An enterprise handbook assistant that only ever retrieves what the person asking is allowed
to read. Built on 774 real pages from GitLab's public handbook (people policies, total
rewards, finance, legal, security and support), with a deliberately unforgiving rule: a
restricted page must not reach the wrong person's results, not even as a hint.

**Engagement readout:** [ENGAGEMENT.md](ENGAGEMENT.md) covers discovery, the rollout plan and
what is still unproven.

**Status: phases 1 and 3 of 3 built.**
- **Built and measured:** ingestion, chunking, keyword retrieval with permissions, the leak
  tests, and the retrieval service on Kubernetes, tested on a kind cluster in CI.
- **Next:** semantic retrieval (Gemini embeddings in pgvector) and the answering agent on ADK.

This README states only what is built and measured.

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

**Whether restricted pages still shape the results.** BM25 weighs each word by how rare it is
in the corpus. Computed over everything, those weights include restricted pages:

| Ranking statistics from | Searches whose scores differ from an index without restricted pages | Order differs |
|:--|--:|--:|
| Readable pages only (shipped) | **0** of 198 | **0** |
| Whole corpus | 165 | 136 |

So filtering before ranking was not enough. Restricted pages still moved allowed results up
and down, which would let someone probe for words in pages they cannot open. The index now
keeps word counts per access group and adds up only the asker's groups. Every result is
exactly what that user would get if the restricted pages did not exist.

**Chunking** (same questions, BM25): the source page is in the top 5 for 97-100% of questions
under all three strategies. Fixed windows and titled heading chunks score about the same (MRR
0.87 and 0.88), and with 33 questions a difference of one question is noise. Titled heading
chunks are shipped because each one says which page and section it came from. The question set was written from
page headings, which favours keyword search. Paraphrased questions and semantic retrieval are
the next comparison.

**Messy data:** 149 of 923 pages (16%) are stubs that only link to an internal handbook. They
are dropped at ingestion, so they cannot fill results with titles and no content.

## The service on Kubernetes

`clearance.service` serves `/search` over HTTP. The caller's identity comes from one header
set by the gateway. Groups are looked up on the server, never taken from the request.
- A request with no identity gets 401.
- An unknown user gets an empty list.
- Query text is not logged, and uvicorn's access log is off because it would record it.

`k8s/` deploys it under the restricted Pod Security standard:
- a read-only filesystem, no service-account token and no Linux capabilities;
- a network policy that admits only pods labelled as the gateway and allows no outbound
  traffic;
- readiness that waits for the index to be built, a zero-unavailable rolling update and a
  disruption budget.

The CI `kubernetes` job builds the image with the real handbook inside and deploys it to a kind
cluster. It then checks four things:
1. A pod without the gateway label cannot connect.
2. An in-cluster probe asks every question as every user, and gets 0 restricted results.
3. The deployment is restarted while the probe runs, and no request fails.
4. The GKE overlay (three zones, autoscaling) passes a server-side dry run.

On the last run the probe made 20,196 searches through the restart: 0 failed, and 0 restricted
results among the 100,980 returned.

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

# the service, locally
CLEARANCE_CORPUS=data/raw/handbook/md CLEARANCE_ACCESS=config/access.yaml   uvicorn --factory clearance.service:from_env --port 8080 --no-access-log
curl -H "x-authenticated-user: priya" "localhost:8080/search?q=refund+policy"
```

Docker and kind reproduce the CI job: `docker build -t clearance:dev .`, `kind load
docker-image clearance:dev`, `kubectl apply -k k8s/overlays/kind`.

## Data

GitLab handbook content, from gitlab.com/gitlab-com/content-sites/handbook. It is downloaded
at run time and not redistributed here. Only the question set and page paths are committed.
