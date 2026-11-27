# Engagement readout: Clearance

This readout treats the project as a customer engagement: how it was scoped, what was built,
how it was proven and how it rolls out. The customer is a composite, a company of a few
thousand people, not a named one. The documents are real (774 pages of GitLab's public
handbook); the access groups and users are invented. Numbers marked **measured** come from
this repository and can be reproduced; everything else is labelled as a plan or an estimate.

## The customer and what they asked for

The company built an internal assistant over its handbook and policy documents. It indexed
everything and answered everyone. In testing, a support engineer asked about contract terms
and got a paragraph about a pending legal settlement. The launch was stopped. The fix on the
table was to remove restricted results after search.

**The ask:** "Make the assistant safe to launch to the whole company, without making it useless
to the teams that own restricted documents."

## What discovery found

| Observed | What it meant |
|:--|:--|
| Restricted and open documents use the same words (contracts, incidents, compensation) | Restricted pages rank highly for ordinary questions. Removing them after ranking leaves users with short or empty results, which also shows that something matched |
| 16% of pages (149 of 923) were stubs: a title and a link to an internal handbook | Stubs match on their title and return nothing useful. They have to be removed at ingestion, and counted |
| The first prototype took the user's groups from the client application | Anyone who could call the service could claim any group. Groups must be looked up on the server from the verified identity |
| Keyword ranking weighs words by how rare they are across the whole corpus | Restricted pages changed the scores and order of results for users who could not read them. Nothing restricted was returned, but someone could still probe for words in restricted pages |
| The service's web server logged full request URLs | Every question about leave, pay or health would sit in the logs. Query text must not be logged |
| The platform team runs everything on GKE and wanted no new runtime | The service had to be a normal Kubernetes deployment, with the cluster's own controls doing part of the enforcement |

**Deliberately not built in this phase:**
- **Answer generation.** Retrieval has to be right first. An answer can only be as safe as the
  passages it was given.
- **Document-level exceptions.** Access here is per section, matching how the customer's
  document owners actually grant it. Per-document exceptions were requested once and can wait.
- **Syncing groups from the identity provider.** The access file stands in for it, with the same
  shape: users to groups.

## What was delivered

- **Permission-aware retrieval.** Every chunk carries its document's group from ingestion. The
  index scores only chunks the asker may read, and computes its ranking statistics from those
  chunks only. Restricted pages have no effect at all on another user's results.
- **Measured chunking.** Three strategies compared on the same labelled questions, with the
  strategy shipped chosen on evidence and the margin of error stated.
- **A retrieval service** that takes identity from one gateway-set header, looks up groups
  itself, refuses requests without identity, gives unknown users nothing, and does not log
  query text.
- **Kubernetes deployment:**
  - two replicas behind a Service, under the restricted Pod Security standard;
  - a read-only filesystem, no Kubernetes API token and no Linux capabilities;
  - a network policy that admits only gateway pods and allows no outbound traffic;
  - readiness tied to the index being built, plus a disruption budget;
  - a GKE overlay with three replicas spread across zones and autoscaling.
- **Checks that fail the build.** A leak, any influence of restricted pages, a pod that can
  reach the service without the gateway label, or a dropped request during a rollout each fail
  CI.

## Proof that it works

| Claim | Evidence | Reproduce |
|:--|:--|:--|
| Restricted text never reaches the wrong user | **Measured:** 0 restricted chunks in 198 searches (33 questions × 6 users) on the real handbook. With no check, 480 | `python scripts/evaluate.py` (CI: handbook job) |
| Filtering after ranking, the proposed fix, starves users | **Measured:** 116 of 198 searches returned fewer results than the user was entitled to | same |
| Restricted pages do not move anyone else's results | **Measured:** 0 of 198 searches differ from an index without the restricted pages. With corpus-wide statistics: 165 differ in score and 136 in order | same |
| Retrieval finds the right page | **Measured:** the source page is in the top 5 for 97% of questions (titled heading chunks, MRR 0.88). One question is 3%, so the strategies are within noise of each other | same |
| The deployed service enforces the same rules | Every question as every user over HTTP, inside the cluster: 0 restricted results, a request without identity refused (401), and an unknown user given nothing | CI: kubernetes job, `k8s/tests/probe-job.yaml` |
| Only the gateway can reach the service | A pod in the same namespace without the gateway label resolves the Service but cannot connect | CI: kubernetes job, `k8s/tests/outsider.yaml` |
| A rolling restart drops no requests | The probe keeps querying through `kubectl rollout restart`, and any failed request fails the job | CI: kubernetes job |
| The production overlay is valid | The GKE overlay passes a server-side dry run against a real API server | CI: kubernetes job |
| The service is light | **Measured on a laptop:** 774 pages indexed into 7,544 chunks in about 1 second; about 95 MB of memory; a search takes 1.1 ms at the median and 5.6 ms at p95 | `uvicorn --factory clearance.service:from_env` |

## What is not yet proven

- **Paraphrased questions and semantic retrieval.** The questions were written by someone who
  had read the page headings, which favours keyword search. The next phase adds Gemini
  embeddings in pgvector and real users' wording, and repeats every check above. Semantic
  search brings the same placement problem: an approximate-nearest-neighbour index filtered
  after search starves users the same way.
- **Identity on GKE.** In the cluster, identity comes from a header set by the gateway, and the
  network policy is what makes that header trustworthy. On GKE, the gateway is Identity-Aware
  Proxy, and the service must verify IAP's signed header rather than trust the plain one. That
  verification is not built yet.
- **Answer quality.** No answers are generated yet, so nothing here measures whether an answer
  is correct or cites the right passage.
- **Real load.** The rolling restart was tested under the probe's sequential load, not under the
  company's peak traffic.

## Rollout plan

| Phase | What changes | Gate to move on |
|:--|:--|:--|
| 1. Retrieval only, one team | Support uses search results directly, with no generated answers | Leak checks stay at 0 on their real questions; support engineers find the right page in the top 5 for most questions |
| 2. Answers, open sections | Generated answers with citations, for people policies and total rewards only | Every answer cites only passages the asker may read; a weekly sample is reviewed by the people team |
| 3. Restricted sections | Legal, security and finance answer for their own teams | Each document owner signs off on a sample of answers from their section |
| 4. Company-wide | All staff, groups synced from the identity provider | Group changes reach the service within the agreed time; an access review each quarter |

## Risks and open items

- **Stale groups.** Someone who leaves the legal team keeps access until the access file
  changes. With identity-provider sync, the delay must be measured and agreed with security.
- **One index in memory per pod.** Fine at 7,544 chunks. At hundreds of thousands, the index
  moves to a database (the pgvector phase), with the same filter-before-ranking rule.
- **Whole-section access.** A single sensitive page in an open section is visible to everyone in
  that group. Document owners need a way to flag pages, and the access model then needs
  per-document exceptions.

## Feedback for the product team

- **Ranking statistics are part of access control.** Any retrieval with keyword or hybrid
  ranking computes term statistics. Managed retrieval products that support per-document
  access should say whether those statistics are computed per access scope, because otherwise
  restricted documents shape other users' results.
- **Filtered vector search needs a documented recall story.** Teams reach for "search, then
  filter" because it is the default in most vector examples. Guidance and benchmarks for
  filtering inside the search, at realistic restriction rates, would stop the starvation
  problem before it reaches production.
- **Identity headers from IAP:** a small, official library to verify IAP's signed header in a
  backend service, with a local test mode, would make "verify, don't trust" the easy path for
  services behind IAP.
