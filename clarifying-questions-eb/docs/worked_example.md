# One item, end to end, through E-B and E-C

Every prompt and every response below is copied from the committed records for
item `7453736095829480185`, Llama-3-70B. Nothing is paraphrased or invented.

- E-B record: `scan_results/full_eb_llama3_70b_results.jsonl`
- E-C record: `scan_results/ec_llama3_70b_results.jsonl`

The prompt templates live in `eb/conditions.py`, `eb/screening.py`, and
`ec/run_experiment.py`; the grading rules in `eb/grading.py`.

## The item

**Ambiguous question:** *What is it called when you mix up the letters of a word?*

AmbigQA supplies exactly two disambiguated readings, which is the eligibility
rule for Set A:

| reading | rewritten question | answer aliases |
|---|---|---|
| `reading_a` | What is it called when a word or phrase is formed by mixing up or rearranging the letters of a word? | Anagram |
| `reading_b` | What is the disorder characterized by mixing up or reversing the order of letters of words? | Dyslexia |

The intended reading for E-B is fixed at **`a`**, chosen uniformly at random
with seed 0 **before any call is made**.

---

# E-B: five stateless calls, temperature 0

Every condition is an independent completion. No conversation state carries
across conditions or across items.

## 1. Screening answer

```
system: Answer the question as concisely as possible: a short phrase, no explanation.
user:   Question: What is it called when you mix up the letters of a word?
```

> `Anagram.`

## 2. Confidence in that answer

This is the post-answer elicitation that replaced the blind pre-answer
procedure in the original design (see `pre_registration.md`).

```
system: You will be shown a question and a proposed answer to it. State only your
        confidence that the proposed answer is correct, as a single integer
        percentage from 0 to 100. Respond with only the number, nothing else.
user:   Question: What is it called when you mix up the letters of a word?
        Proposed answer: Anagram.
```

> `50`

50 falls inside the 50–60 matching band, so the item enters the sample. **This
screening answer becomes condition 1 (answer-now)** rather than being
regenerated, because it is the answer whose confidence was rated.

## 3. Oracle clarification

The ambiguous question is replaced outright by the intended reading's rewrite.
No dialogue, no model-written question — this measures the value available from
disambiguation, independently of whether the model could obtain it.

```
system: Answer the question as concisely as possible: a short phrase, no explanation.
user:   Question: What is it called when a word or phrase is formed by mixing up
        or rearranging the letters of a word?
```

> `An anagram.`

## 4. Self-ask — three chained calls

**4a. The model writes one clarifying question.**

```
system: You will be asked a question. Before answering, you may ask exactly one
        clarifying question. Ask it now, as a single sentence, and nothing else.
user:   Question: What is it called when you mix up the letters of a word?
```

> `Are you referring to a specific type of wordplay or linguistic phenomenon, or just a general mistake in spelling?`

**4b. A simulated user answers it.** This is where the ambiguity is settled.
The simulator's context contains the ambiguous question, the rewrite, and the
clarifying question — **never the answer field**. That is the leak firewall: the
rewrite disambiguates without stating the answer, so leakage is prevented by
construction rather than by instruction.

```
system: You are a helpful user answering a clarifying question about what you meant.
user:   You are the person who asked: "What is it called when you mix up the letters of a word?"
        What you meant was: "What is it called when a word or phrase is formed by mixing up
        or rearranging the letters of a word?"

        The assistant has asked you: "Are you referring to a specific type of wordplay or
        linguistic phenomenon, or just a general mistake in spelling?"

        Reply in one short sentence. Say only which of the two you meant.
```

> `I meant a specific type of wordplay or linguistic phenomenon.`

The reply is then audited against the intended answer's aliases. `Anagram` does
not appear, so `leaked: false`. (One Llama item failed this audit and was
dropped; see `EB_EXPERIMENT_SUMMARY.md`.)

**4c. The model answers the original question, given the exchange.**

```
system: Answer the question as concisely as possible: a short phrase, no explanation.
user:   Question: What is it called when you mix up the letters of a word?
        You asked: "Are you referring to a specific type of wordplay or linguistic
        phenomenon, or just a general mistake in spelling?"
        Reply: "I meant a specific type of wordplay or linguistic phenomenon."
        Now answer the original question.
```

> `Anagram.`

## 5. Free choice

The model decides for itself whether to ask.

```
system: You will be asked a question. You may either answer it directly, or ask
        exactly one clarifying question first if the question seems ambiguous.
        If you want to ask a question, respond with exactly:
        ASK: <your clarifying question>
        Otherwise, just answer the question directly and concisely, with no preamble.
user:   Question: What is it called when you mix up the letters of a word?
```

> `Anagram`

No `ASK:` prefix, so `asked: false`. Had it asked, the same simulated-user
exchange as 4b–4c would have run.

## Grading

Each response is checked against **reading_a's** aliases, with reading_b's
aliases as the "other" list. Normalization lowercases, strips punctuation, and
drops articles; matching is alias containment.

| outcome | rule |
|---|---|
| `hedged` | contains aliases from **both** readings — scores 0 under strict grading |
| `correct` | contains the intended reading's alias only |
| `wrong` | neither |

Hedge is checked **before** correct, so an answer that also names the other
reading is a hedge rather than a pass.

| condition | response | grade |
|---|---|---|
| answer-now | Anagram. | correct |
| oracle-clarify | An anagram. | correct |
| self-ask | Anagram. | correct |
| free-choice | Anagram | correct |

All four are correct, so **every E-B gain for this item is 0**. The model was
already answering the intended reading, so there was nothing for clarification
to add.

---

# E-C: three batches of 32 samples, temperature 1.0

E-C never asks a clarifying question and never uses a simulator. It substitutes
the rewrite directly and measures how much the answer depends on which reading
was meant. The prompt is the plain answer prompt from condition 1, sampled
repeatedly.

```
system: Answer the question as concisely as possible: a short phrase, no explanation.
user:   Question: <the ambiguous question, then each rewrite in turn>
```

Each of the 32 responses per batch is bucketed by alias matching into
`reading_a`, `reading_b`, `multiple` (matches both), or `other` (matches
neither).

| batch | prompt | resulting clusters |
|---|---|---|
| ambiguous | *What is it called when you mix up the letters of a word?* | 32/32 → `reading_a` |
| reading_a | *…formed by mixing up or rearranging the letters…* | 32/32 → `reading_a` |
| reading_b | *What is the disorder characterized by mixing up or reversing…* | 32/32 → `reading_b` |

## The variance decomposition

Treat the answer cluster as a one-hot vector. For a categorical distribution
`p`, total variance is `V(p) = 1 − Σ p(c)²` — the probability that two
independent draws disagree. With a uniform prior over which reading was
intended, the two conditional distributions are `(1, 0)` and `(0, 1)`, and
their mixture is `(0.5, 0.5)`:

```
total   = V(0.5, 0.5)                     = 0.500
within  = ½·V(1, 0) + ½·V(0, 1)           = 0.000
between = total − within                  = 0.500
```

This is the maximum possible between-reading variance: fixing the reading
determines the answer completely, and nothing is left over.

## The realized gain

Per reading, clarified accuracy minus the baseline accuracy measured on the
ambiguous batch:

| reading | baseline (ambiguous prompt) | clarified (rewritten prompt) | gain |
|---|---|---|---|
| reading_a | 1.000 | 1.000 | 0.000 |
| reading_b | 0.000 | 1.000 | +1.000 |

Averaged uniformly over both readings: **realized gain = +0.500**.

---

# Why the two disagree on this item

E-B records **zero gain**: the model guessed the intended reading unaided, so
clarification could not improve on it.

E-C records **+0.5 gain**: the answer is entirely determined by which reading is
meant, and the model never produces "dyslexia" unless the reading is fixed for
it.

Both are correct measurements of different quantities. E-B asks whether a
clarifying exchange helps *this* prompt with *this* intended reading. E-C asks
how much the answer depends on the interpretation at all, averaged over both
readings the user might have had. The second is the quantity the theory is
about; the first is what a deployed system would experience.

It also shows the estimator's limitation plainly: computing that 0.500 required
AmbigQA's two annotated rewrites. Nothing in the ambiguous batch alone reveals
it — all 32 samples said "anagram" with complete consistency, so any
uncertainty measure computed from the ambiguous prompt would report **zero**
uncertainty on an item whose answer is fully interpretation-dependent.
