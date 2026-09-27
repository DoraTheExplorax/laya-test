# Laya pipeline-confidence bench — what works and what doesn't

**Goal:** estimate whether a document-processing pipeline's final output (invoice / order after
parsing → extraction → matching → validation) is correct enough to process automatically.

**Approach:** instead of feeding the whole record to a model, ask Laya small typed questions about each
pipeline stage, turn every answer into *P(this stage's value is correct)*, and combine them into one
confidence score. Each stage has its own questions and ground truth and must pass a gate before the next
unlocks. Question wording is tuned on one half of the training cases and kept only if it also holds on
the other half (*tune* / *holdout*).

> **Core lesson: Laya reads and detects; code compares and calculates.**
> One question = one thing about one piece of text.

---

## Stage by stage

| Stage | What the questions check | Status |
|---|---|---|
| **Parser** | Does the table agree with the text? | ✅ Passes — 534/534 train, 130/130 val |
| **Extraction** | Header fields, product, unit, price basis, discount present?, tax exempt? | 🟡 Most pass; product lookup weak |
| **Matching** | Which master-data supplier / buyer / product? | 🟡 Parties pass; product weak |
| **Validation** | Situation detectors (returns, cartons, FX, VAT-inclusive prices, …) | 🟡 Detection AUROC 0.75–1.00 |
| **Final confidence** | Combination of all stages | ⚠️ ~0.60–0.80 AUROC on a 10+10 sample — too small to trust yet |

*AUROC = probability a correct case is scored above an incorrect one (0.5 = coin flip, 1.0 = perfect).*

---

## ✅ What works

### 1. One comparison per question
The parser check failed while it compared several values at once, then became perfect when split into
one pair per question.

| | Question | Evidence | Result |
|---|---|---|---|
| ❌ Before | *"For every line, does the table agree with the text on description, quantity and rate?"* | whole table + text | AUROC **0.41** |
| ✅ After | *"Do the text and the table say the same thing?"* | `The text says 8.00. The table says 12.00.` | AUROC **1.00** |

### 2. Detection instead of verification
Ask whether a situation applies; let code check the stage's value is consistent with it.

| | Question | Result |
|---|---|---|
| ❌ Before | *"What tax rate applies to this line? An export certificate overrides the standard VAT rate."* (choice 0 / 20) | caught **0/4** exemptions, AUROC 0.64 |
| ✅ After | *"Do the notes about this line make it an export item or exempt it from VAT?"* → code: exempt ⇒ tax must be 0 | caught **4/4**, AUROC **1.00** |

Same pattern for discounts: *"Do the statement or the notes for this line mention a discount, allowance
or promotion?"* → AUROC **0.92** (the 4-way "which percentage?" choice was **0.53**).

### 3. Small, focused evidence
Sending each question only what concerns it — the line's own statement plus notes that mention it —
instead of the whole document.

| Question | Whole text | Focused evidence |
|---|---|---|
| *"Which party supplies the goods?"* | 1/10 correct | **20/20** |
| Discount present? | picked 14.5% on 28 of 33 lines | AUROC 0.92 |

### 4. One note per question
Showing all of a line's notes at once made the model answer *"yes"* whenever notes existed.
Asking about each note separately (the line has the situation if any note says so):

| Situation question | All notes at once (tune / holdout) | One note at a time |
|---|---|---|
| *"…quantity is given in packs or cartons?"* | 0.31 / 0.27 | **0.91 / 0.86** |
| *"…billed by weight?"* | 0.43 / 0.36 | **0.84 / 0.82** |
| *"…price is per a quantity other than one unit?"* | 0.46 / 0.15 | **0.80 / 0.75** |
| *"…a number in a locale format (comma decimal)?"* | 0.35 / 0.45 | **0.89 / 0.92** |
| *"…a price in another currency?"* | 0.90 / 0.86 | **0.98 / 0.95** |
| *"…exempt the given line from VAT?"* | 0.92 / 0.97 | **1.00 / 1.00** |

### 5. Choice questions over closed, exclusive value sets
Currency, document type, unit, payment terms, freight, tax rate and master-data ids (supplier, buyer,
ship-to, product) are asked as *"which one?"* with options that are verified to be **exhaustive and
mutually exclusive** across every value in the data. Document type, payment terms, unit, currency,
ship-to and freight all pass (27–30 of 30).

---

## ❌ What doesn't work

### 1. Applying a rule to compute a number
The model can spot the note but cannot do the arithmetic it implies.

| Question | Needs | Result (holdout) |
|---|---|---|
| *"Does a signed note change this line's quantity from what the line states?"* | 24 + 6 = 30, 8 CTN × 12 = 96 | AUROC **0.47** |
| *"…change this line's price per unit?"* | USD 12.50 × 0.80 = 10.00 | AUROC **0.47** |

→ Numeric fields are judged by detection questions + code rules instead.

### 2. "If X applies, is the value right?" questions
*"If this line is a return or credit, is its quantity negative? If it is not a return, answer yes when
its quantity is positive."* — conditional questions answered "yes" almost always and caught **~0**
real errors; this one was even inverted (AUROC **0.03**). Split into detection + code rule, the same
situation scores **1.00**.

### 3. Role asymmetry
The model always finds the **supplier** but consistently inverts **"buyer"** in every wording tried
(AUROC 0.07–0.24). Since master data always has exactly one supplier and one buyer, the buyer is
derived as *"the party that is not the supplier"* → **20/20**.

### 4. Multi-option choices whose labels don't resemble the source
*"What total discount percent applies?"* with options `100 / 14.5 / 15`:

| Option labels | Result |
|---|---|
| `"the full price is waived"`, `"two percentage discounts applied one after the other"` … (generic) | AUROC **0.50 / 0.38** |
| Labels copying the document's wording (*"promotional allowance of 100%"*) | AUROC 0.99 — but **overfit** to this dataset's phrasing |

→ Dropped from extraction; handled by detection questions instead.

### 5. Product identity
Picking the right SKU for a line (buyer codes, variants, substitutions, repeated items) is still weak
(AUROC ~0.65 in extraction/matching).

---

## Guard rails we added

- **Tune / holdout split** of the training cases — a wording change counts only if it holds on cases it
  wasn't tuned on (several "good-looking" wordings were rejected this way).
- **Synthetic parser errors** — the data has no real parser mistakes, so 30% of cases get a tampered
  copy; otherwise a question that always says "fine" would pass.
- **Per-stage gates** — later stages unlock only after earlier ones pass on the full training split.
- **Run log** — every run and probe (question text, evidence, scores) is appended to `runs/run_log.txt`.

## Next

1. Fix product-identity questions.
2. Fit per-stage weights for the final confidence (currently a plain product of ~60 probabilities, which
   squashes scores towards 0) on train; check on val.
3. Choose the auto-processing threshold from the resulting score distribution.
