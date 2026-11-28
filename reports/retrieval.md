# Retrieval and permissions report

Corpus: 774 handbook pages in 6 sections; 149 stub pages (a link to an internal handbook and little else) dropped at ingestion.
Questions: 33, each asked by the least-privileged user allowed to read its source. Retrieval: BM25 keyword search. Regenerate with `python scripts/evaluate.py`.

## Chunking strategies

| Strategy | Chunks | Source page in top 1 | Top 3 | Top 5 | Mean reciprocal rank |
|:--|--:|--:|--:|--:|--:|
| fixed | 6484 | 79% | 97% | 100% | 0.87 |
| headings | 7544 | 67% | 91% | 97% | 0.80 |
| headings_titled | 7544 | 79% | 97% | 97% | 0.88 |

With 33 questions, one question is 3%: differences of a question or two are noise, not a ranking. The questions were written by someone who had read the page headings, which favours keyword retrieval; paraphrased questions from real users are the harder test.

## Where the permission check sits

Every question asked by every user (198 searches, top 5, `headings_titled` chunks).

| Filter | Restricted chunks returned | Searches cut short |
|:--|--:|--:|
| Before ranking (shipped) | 0 | 0 |
| After ranking | 0 | 116 |
| None | 480 | 0 |

*Cut short:* the user could read relevant pages, but got fewer than 5 results because restricted pages took the top places before being removed. Post-filtering does not leak text, but it starves users of answers and its short result lists reveal that something matched.

## Whether unreadable pages change what a user sees

BM25 weighs each word by how rare it is in the corpus. Compared here: each user's results from the full index against an index of only the pages they may read, for every question and user (198 searches, top 5).

| Ranking statistics from | Searches with different scores | Different order |
|:--|--:|--:|
| Readable pages only (shipped) | 0 | 0 |
| Whole corpus | 165 | 136 |

With whole-corpus statistics no restricted text is returned, but restricted pages still move allowed results up and down, so a user could probe for words in pages they cannot open. Scoped statistics remove that channel: the results are exactly what the user would get if restricted pages did not exist.
