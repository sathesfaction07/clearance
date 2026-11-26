# Retrieval and permissions report

Corpus: 774 handbook pages in 6 sections; 149 stub pages (a link to an internal handbook and little else) dropped at ingestion.
Questions: 33, each asked by the least-privileged user allowed to read its source. Retrieval: BM25 keyword search. Regenerate with `python scripts/evaluate.py`.

## Chunking strategies

| Strategy | Chunks | Source page in top 1 | Top 3 | Top 5 | Mean reciprocal rank |
|:--|--:|--:|--:|--:|--:|
| fixed | 6481 | 79% | 94% | 94% | 0.86 |
| headings | 7544 | 67% | 91% | 94% | 0.80 |
| headings_titled | 7544 | 76% | 94% | 97% | 0.86 |

With 33 questions, one question is 3%: differences of a question or two are noise, not a ranking. The questions were written by someone who had read the page headings, which favours keyword retrieval; paraphrased questions from real users are the harder test.

## Where the permission check sits

Every question asked by every user (198 searches, top 5, `headings_titled` chunks).

| Filter | Restricted chunks returned | Searches cut short |
|:--|--:|--:|
| Before ranking (shipped) | 0 | 0 |
| After ranking | 0 | 116 |
| None | 480 | 0 |

*Cut short:* the user could read relevant pages, but got fewer than 5 results because restricted pages took the top places before being removed. Post-filtering does not leak text, but it starves users of answers and its short result lists reveal that something matched.
