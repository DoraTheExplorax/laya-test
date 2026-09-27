"""
THE FILE YOU EDIT.  Stage-by-stage question bench for the pipeline-confidence dataset.

The pipeline runs  parser -> extraction -> matching -> validation.  Each STAGE below has its own
params (questions) that judge THAT stage's document.  You work through the stages in order:
a stage must PASS its gate (see GATE) before the bench will run the next one.

    bin/python bench/server.py                              # UI at http://localhost:8765 (run from here too)
    bin/python bench/bench.py                               # run stages in order, stop at first non-passing
    bin/python bench/bench.py --stage matching              # run up to (and including) a stage
    bin/python bench/bench.py --parser-flags                # list code-computed parser mistakes
    bin/python bench/bench.py --stage validation --force    # ignore gates
    bin/python bench/bench.py --split val                   # held-out 50 cases (gates taken from train)
    bin/python bench/bench.py --preview extraction.ship_to  # payload for one param, no API call

A param with empty `instructions` is "not written": the stage cannot pass until every param has a
question (delete params you don't want).  Answers are cached, so re-runs only pay for changed questions.

Ground truth (training split only): each stage's document is compared field-by-field with
training_annotations.expected_final_output.  Validation-stage params with a `mechanism` are instead labelled
by training_annotations.failure_mechanisms.  Annotations are never sent to the model.

Write each question so YES means "this stage's value is right" -- also when the situation it targets
doesn't occur in the case -- or set polarity "error" if YES means wrong.

Param schema
------------
type          "noul"   yes/no question; API returns P(yes).
              "choice" the model picks ONE option.  Two kinds:
                value choice   `answer_field` set: options are the possible VALUES of that field
                               (`criteria` = one of ENUMS, or `criteria_from` = per-case master data).
                               P(correct) = probability the model gave to the stage's value.
                               Don't put doc_* slices in evidence -- the model should derive the value
                               from the source, not copy it.  The same answer then scores every stage.
                verdict choice `criteria` {option: description} + `correct_choice`.
criteria_from "parties" | "locations" | "products": options = this case's master-data ids (always exhaustive).
instructions  your question.  EMPTY = not written.
polarity      "correct" (YES = right)  |  "error" (YES = wrong)
scope         "case" (asked once)  |  "line" (asked per line of the stage document; case prob = product)
evidence      slices sent as `state`, in order (list below).  Server truncates at ~1024 tokens.
fields        the stage-document fields this param vouches for.  Drives its labels and the cumulative
              confidence (a later stage's param that covers the same fields supersedes this one).
              "__line_set__" = the set/order of line ids.
mechanism     (optional) label by failure_mechanisms instead of field comparison.
label_from    "parser_flags": label by code-computed parser mistakes (parser stage).
Stage keys    document ("parser" | "extractor" | "normalization" | "matching" | "final"),
              carry_forward (default True; False = scored and gated, left out of cumulative confidence),
              synthetic_errors (parser only: fraction of cases that also get a tampered copy).
only_table_lines  (parser line params) ask only about lines that have a table row.
label_field   (with label_from parser_flags) only conflicts on this field make the label wrong.
yes_value     (noul on a two-value field, with `criteria` + `answer_field`) YES means the field should be
              this value; the stage value is scored against it.  Used for tax: "exempt?" -> 0.
no_value      (noul with `answer_field`) "is there any X?": NO means the field should be this value.
              "statement": NO means the field keeps the value from the line's own statement ("did a note change it?").
label_compare "is_no_value": label only checks none-vs-some (with no_value).
ask_unless_value  skip lines whose stage value equals this (e.g. amount only when discount != 0).
per_note      (noul) ask the question once per note about the line (state: {"line": id, "note": text}); YES if any note says so,
              NO without a call when the line has no notes.  Add "general_notes" to evidence to include notes
              that name no line.
stage_check   (noul) detection question: YES = the situation applies.  Code rule the stage value must meet
              when it applies (quantity_negative, quantity_zero, discount_100, tax_zero, schedule_split,
              price_basis_not_1, quantity_changed, unit_price_changed, delivery_date_changed).
answer_is_other  (two-option value choice) the question asks for the OTHER option; the stage value is
              right when it is not the model's pick.  Used for buyer = "the party that is not the supplier".
decision_threshold  (optional, default 0.5) P(correct) at or above this counts as "says correct" for the
              checks / caught / false alarms / gate.  Confidence scores always use the raw probabilities.

EVIDENCE SLICES
---------------
doc_header / doc_line / doc_lines / doc_line_ids / doc_totals / doc_parties
                     THIS stage's document: header / the line being judged / all lines / ids+products+qty / totals
                     (matching stage: header = selected supplier/buyer, lines = line_id + selected product_id)
final_header / final_line / final_lines / final_totals   always final_output, whatever the stage
extractor_line / normalization_line                      that stage's version of this line (compare stages)
normalization_ops    normalization_output.applied_operations
matching_line        product candidates + scores for this line
email                source email subject/body
parser_text          all parsed text blocks in reading order + tables (authoritative source)
parser_text_line     (line scope) blocks mentioning this line or no line at all + this line's table row
master_parties / master_locations / master_products / master_products_brief (no units)
master_product_line  (line scope) master record of this line's product
line_ref             (line scope) just {"line_id": ...} -- tells a value question which line, leaks no value
line_source          (line scope) this line's own statement + table row + only the notes that mention it
line_discount_notes  (line scope) only this line's notes about a discount / allowance / promotion
header_source        document header sentences + notes that mention no line
general_notes        only the notes that mention no line (e.g. "this document covers only five units")
doc_fields           just this param's own `fields` from the stage document (e.g. {"quantity": "18"})
parser_line          (parser stage, line scope) this line's text statement + table row as parsed
parser_table_check   (parser stage) every table row next to the text statement for the same line
parser_line_pairs    (parser stage, line scope) description / quantity / rate as {text, table} pairs
parser_table_pairs   (parser stage) the same pairs for every line that has a table row
parser_description_pair / parser_quantity_pair / parser_rate_pair
                     (parser stage, line scope) one value as "The text says X. The table says Y."
parser_tables / parser_warnings / parser_reading_order   raw parser_output pieces
validation_checks    validation_output.checks (the pipeline's own checks -- may be wrong)
customer_profile / commercial_policy
"""

# Closed value sets, taken from every stage document + expected output in BOTH splits.
# `bin/python bench/bench.py --check-enums` re-verifies they are exhaustive and exclusive.
# Keys are the exact field values; descriptions are yours to refine -- keep them mutually exclusive.
ENUMS = {
    "document_type": {"invoice": "invoice", "purchase_order": "purchase order", "sales_order": "sales order"},
    "currency": {"EUR": "euro", "GBP": "pound sterling"},
    "payment_terms_days": {"30": "30 days", "45": "45 days"},
    "freight": {"0.00": "no freight charge", "9.50": "freight 9.50", "19.00": "freight 19.00"},
    "unit": {"EA": "each (pieces)", "KG": "kilograms"},
    "price_basis": {"1": "price per 1 unit", "100": "price per 100 units"},
    "discount_percent": {
        "0": "no discount or allowance is stated for this line",
        "14.5": "two allowances where the second applies to the already discounted amount (10% then 5%)",
        "15": "a single 15% discount on the full amount",
        "100": "100% allowance: the goods are free but stay on the posting",
    },
    "tax_rate": {"20": "20 percent", "0": "0 percent"},
}

STAGES = {
    # =========================================================== 1. parser
    # Parser mistakes can be corrected by later stages, so this stage is scored and gated but does NOT
    # carry into the cumulative confidence.  Labels come from code-computed parser flags
    # (`bin/python bench/bench.py --parser-flags`), so they exist on both splits.
    "parser": {
        "document": "parser",
        "carry_forward": False,
        # The dataset has no parser errors, so this fraction of cases also gets a tampered copy with one
        # table cell contradicting the text (labelled by the same code flags).  Without it the gate could
        # only measure false alarms.  Tampered copies are used in this stage only.
        "synthetic_errors": 0.3,
        # One value per question: the model compares a single text/table pair reliably, but not several at
        # once (see run_log.txt, parser iterations).  Lines without a table row are not asked.
        "params": {
            "parse_description": {
                "type": "noul",
                "instructions": "Is the table description exactly the same as the text description, word for word, with no words added or missing?",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["parser_description_pair"],
                "only_table_lines": True,
                # clean pairs score >= 0.83, the hardest conflict ("legacy connector" vs "stainless legacy
                # connector") 0.51 -- ranking is perfect, 0.5 just sits at the top of the gap (run_log PROBE)
                "decision_threshold": 0.65,
                "label_from": "parser_flags",
                "label_field": "description",
                "fields": ["__parse_description__"],
            },
            "parse_quantity": {
                "type": "noul",
                "instructions": "Do the text and the table say the same thing?",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["parser_quantity_pair"],
                "only_table_lines": True,
                "label_from": "parser_flags",
                "label_field": "quantity",
                "fields": ["__parse_quantity__"],
            },
            "parse_rate": {
                "type": "noul",
                "instructions": "Do the text and the table say the same thing?",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["parser_rate_pair"],
                "only_table_lines": True,
                "label_from": "parser_flags",
                "label_field": "rate",
                "fields": ["__parse_rate__"],
            },
        },
    },
    # =========================================================== 2. extraction
    # Judges the extracted document after normalization (what matching consumes).
    # Set "document": "extractor" to judge the raw extractor output instead.
    "extraction": {
        "document": "normalization",
        "params": {
            "doc_type": {
                "type": "choice",
                "instructions": "What type of commercial document is this, as stated in the source?",
                "criteria": ENUMS["document_type"],
                "answer_field": "document_type",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["document_type"],
            },
            "doc_number_date": {
                "type": "noul",
                "instructions": "Do the document number and document date in doc_header match the source exactly, with the date written as YYYY-MM-DD?",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["parser_text", "customer_profile", "doc_header"],
                "fields": ["document_number", "document_date"],
            },
            "currency": {
                "type": "choice",
                "instructions": "In which currency must this document be settled? A signed annex or note overrides the currency printed on the form header.",
                "criteria": ENUMS["currency"],
                "answer_field": "currency",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["currency"],
            },
            "supplier": {
                "type": "choice",
                "instructions": "Which party supplies the goods (the seller)? The sender of the message is not necessarily the supplier.",
                "criteria_from": "parties",
                "answer_field": "supplier_id",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["supplier_id"],
            },
            "buyer": {
                "type": "choice",
                "instructions": "Which party supplies the goods (the seller)? The sender of the message is not necessarily the supplier.",
                # The model reliably finds the supplier but inverts "buyer" in every wording tried (run_log PROBE).
                # Master data always has exactly one supplier and one buyer, so the buyer is the other party.
                "answer_is_other": True,
                "criteria_from": "parties",
                "answer_field": "buyer_id",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["buyer_id"],
            },
            "ship_to": {
                "type": "choice",
                "instructions": "Where are the goods delivered? Billing, invoice or signature addresses are not delivery locations.",
                "criteria_from": "locations",
                "answer_field": "ship_to_id",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["ship_to_id"],
            },
            "payment_terms": {
                "type": "choice",
                "instructions": "How many days are the payment terms? A signed agreement or note overrides the standard template or footer terms.",
                "criteria": ENUMS["payment_terms_days"],
                "answer_field": "payment_terms_days",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["payment_terms_days"],
            },
            "freight": {
                "type": "choice",
                "instructions": "What is the total freight charge for this document? A freight note repeated on several pages counts only once; if no freight is stated it is zero.",
                "criteria": ENUMS["freight"],
                "answer_field": "freight",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["freight"],
            },
            "line_set": {
                "type": "noul",
                "instructions": "Does doc_line_ids list exactly the lines that should be posted, in order? Keep every source line, including cancelled lines (kept at zero quantity) and return lines; keep repeated items as separate lines; add no lines that are not in the source.",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["parser_text", "commercial_policy", "doc_line_ids"],
                "fields": ["__line_set__"],
            },
            "line_product": {
                "type": "choice",
                "instructions": "Which product is ordered on this line? Apply buyer-specific article codes, the required specification (grade, material, size, revision) and any approved substitution stated in the source.",
                "criteria_from": "products",
                "answer_field": "product_id",
                "scope": "line",
                "evidence": ["line_source"],
                "fields": ["product_id"],
            },
            "line_unit": {
                "type": "choice",
                "instructions": "In which unit is this line posted?",
                "criteria": ENUMS["unit"],
                "answer_field": "unit",
                "scope": "line",
                "evidence": ["line_source", "master_product_line"],
                "fields": ["unit"],
            },
            "line_price_basis": {
                "type": "choice",
                "instructions": "For how many units is this line's price quoted (its price unit)?",
                "criteria": ENUMS["price_basis"],
                "answer_field": "price_basis",
                "scope": "line",
                "evidence": ["line_source"],
                "fields": ["price_basis"],
            },
            "line_discount": {
                # step 1: is there any discount?  (the 4-way choice was near chance; this separates at AUROC 0.90)
                "type": "noul",
                "instructions": "Do the statement or the notes for this line mention a discount, allowance or promotion for this line?",
                "answer_field": "discount_percent",
                "no_value": "0",
                "label_compare": "is_no_value",
                "scope": "line",
                "evidence": ["line_source"],
                "fields": ["discount_percent"],
            },
            "line_tax": {
                # yes/no instead of a 2-way choice: AUROC 0.99 vs 0.64 (run_log PROBE). YES means rate 0.
                "type": "noul",
                "instructions": "Do the notes about this line make it an export item or exempt it from VAT?",
                "criteria": ENUMS["tax_rate"],
                "yes_value": "0",
                "answer_field": "tax_rate",
                "scope": "line",
                "evidence": ["line_source"],
                "fields": ["tax_rate"],
            },
        },
    },
    # =========================================================== 3. matching
    "matching": {
        "document": "matching",
        "params": {
            "supplier": {
                "type": "choice",
                "instructions": "Which party supplies the goods (the seller)? The sender of the message is not necessarily the supplier.",
                "criteria_from": "parties",
                "answer_field": "supplier_id",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["supplier_id"],
            },
            "buyer": {
                "type": "choice",
                "instructions": "Which party supplies the goods (the seller)? The sender of the message is not necessarily the supplier.",
                # The model reliably finds the supplier but inverts "buyer" in every wording tried (run_log PROBE).
                # Master data always has exactly one supplier and one buyer, so the buyer is the other party.
                "answer_is_other": True,
                "criteria_from": "parties",
                "answer_field": "buyer_id",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["buyer_id"],
            },
            "line_product": {
                "type": "choice",
                "instructions": "Which product is ordered on this line? Apply buyer-specific article codes, the required specification (grade, material, size, revision) and any approved substitution stated in the source.",
                "criteria_from": "products",
                "answer_field": "product_id",
                "scope": "line",
                "evidence": ["line_source"],
                "fields": ["product_id"],
            },
        },
    },
    # =========================================================== 4. validation
    # Judges final_output (the validated document).  `validation_checks` = the pipeline's own checks,
    # which may be wrong.
    "validation": {
        "document": "final",
        "params": {
            "doc_type": {
                "type": "choice",
                "instructions": "What type of commercial document is this, as stated in the source?",
                "criteria": ENUMS["document_type"],
                "answer_field": "document_type",
                "scope": "case",
                "evidence": ["header_source"],
                "fields": ["document_type"],
            },
            "doc_number_date": {
                "type": "noul",
                "instructions": "Do the document number and document date in doc_header match the source exactly, with the date written as YYYY-MM-DD?",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["parser_text", "customer_profile", "doc_header"],
                "fields": ["document_number", "document_date"],
            },
            "line_set": {
                "type": "noul",
                "instructions": "Does doc_line_ids list exactly the lines that should be posted, in order? Keep every source line, including cancelled lines (kept at zero quantity) and return lines; keep repeated items as separate lines; add no lines that are not in the source.",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["parser_text", "commercial_policy", "doc_line_ids"],
                "fields": ["__line_set__"],
            },
            # ------------------------------------------------------------ header
            "currency_precedence": {
                "type": "noul",
                "instructions": "Is the currency in doc_header the one the source says to settle in? A signed annex or note overrides the currency symbol on the form header. With no such note, answer yes if it matches the stated base currency.",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["header_source", "doc_header"],
                "mechanism": "currency_precedence",
                "fields": ["currency"],
            },
            "payment_terms": {
                "type": "noul",
                "instructions": "Does payment_terms_days in doc_header match the governing payment terms? A signed agreement or note overrides the template or footer terms. With no such note, answer yes if it matches the stated terms.",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["header_source", "doc_header"],
                "mechanism": "payment_terms",
                "fields": ["payment_terms_days"],
            },
            "party_roles": {
                "type": "noul",
                "instructions": "Are the parties in doc_header correct: supplier_id is the party supplying the goods and buyer_id is the party placing the order, regardless of who sent the message?",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["header_source", "master_parties", "doc_header"],
                "mechanism": "party_roles",
                "fields": ["supplier_id", "buyer_id"],
            },
            "ship_to": {
                "type": "noul",
                "instructions": "Is ship_to_id in doc_header the delivery location named in the source, not the billing or finance address?",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["header_source", "master_locations", "doc_header"],
                "mechanism": "ship_to",
                "fields": ["ship_to_id"],
            },
            "freight_scope": {
                "type": "noul",
                "instructions": "Is freight in doc_totals exactly the freight charge stated in the source, counted once even if the note is repeated, and zero if no freight is stated?",
                "polarity": "correct",
                "scope": "case",
                "evidence": ["header_source", "doc_totals"],
                "mechanism": "freight_scope",
                "fields": ["freight"],
            },
            # ------------------------------------------------------------ line quantity / unit
            "revision_quantity": {
                "type": "noul",
                "instructions": "Does this note change the given line's ordered quantity?",
                "per_note": True,
                "stage_check": "quantity_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "revision_quantity",
                "fields": ["quantity"],
            },
            "cancelled_line": {
                "type": "noul",
                "instructions": "Is this line cancelled?",
                "stage_check": "quantity_zero",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "cancelled_line",
                "fields": ["quantity"],
            },
            "credit_sign": {
                "type": "noul",
                "instructions": "Is this line a return or credit of goods supplied earlier?",
                "stage_check": "quantity_negative",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "credit_sign",
                "fields": ["quantity"],
            },
            "partial_billing": {
                "type": "noul",
                "instructions": "Does this note say only part of the quantity is covered by this document?",
                "per_note": True,
                "stage_check": "quantity_changed",
                "scope": "line",
                "evidence": ["line_source", "general_notes"],
                "mechanism": "partial_billing",
                "fields": ["quantity"],
            },
            "pack_conversion": {
                "type": "noul",
                "instructions": "Does this note say the given line's quantity is given in packs or cartons?",
                "per_note": True,
                "stage_check": "quantity_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "pack_conversion",
                "fields": ["quantity", "unit"],
            },
            "net_weight": {
                "type": "noul",
                "instructions": "Does this note say the given line is billed by weight?",
                "per_note": True,
                "stage_check": "quantity_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "net_weight",
                "fields": ["quantity", "unit"],
            },
            # ------------------------------------------------------------ line price
            "price_basis": {
                "type": "noul",
                "instructions": "Does this note say the price is per a quantity other than one unit?",
                "per_note": True,
                "stage_check": "price_basis_not_1",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "price_basis",
                "fields": ["price_basis", "unit_price"],
            },
            "gross_price": {
                "type": "noul",
                "instructions": "Does this note say the price includes VAT?",
                "per_note": True,
                "stage_check": "unit_price_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "gross_price",
                "fields": ["unit_price"],
            },
            "fx_conversion": {
                "type": "noul",
                "instructions": "Does this note give a price in another currency?",
                "per_note": True,
                "stage_check": "unit_price_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "fx_conversion",
                "fields": ["unit_price"],
            },
            "decimal_locale": {
                "type": "noul",
                "instructions": "Does this note write a number in a locale format, such as a comma as the decimal separator?",
                "per_note": True,
                "stage_check": "unit_price_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "decimal_locale",
                "fields": ["unit_price", "quantity"],
            },
            # ------------------------------------------------------------ line discount / tax
            "stacked_discount": {
                "type": "noul",
                "instructions": "If this line has several allowances where one applies on the already discounted amount, is discount_percent the compounded total (10% then 5% = 14.5, not 15)? Otherwise answer yes when the discount matches the source.",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["line_source", "doc_line"],
                "mechanism": "stacked_discount",
                "fields": ["discount_percent"],
            },
            "free_goods": {
                "type": "noul",
                "instructions": "Does this note say the given line is free of charge?",
                "per_note": True,
                "stage_check": "discount_100",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "free_goods",
                "fields": ["discount_percent", "quantity"],
            },
            "tax_exemption": {
                "type": "noul",
                "instructions": "Does this note exempt the given line from VAT?",
                "per_note": True,
                "stage_check": "tax_zero",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "tax_exemption",
                "fields": ["tax_rate"],
            },
            # ------------------------------------------------------------ line dates
            "date_locale": {
                "type": "noul",
                "instructions": "Does this note give the due date as a numeric date?",
                "per_note": True,
                "stage_check": "delivery_date_changed",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "date_locale",
                "fields": ["delivery_date"],
            },
            "split_schedule": {
                "type": "noul",
                "instructions": "Does this note split the given line's delivery across several dates?",
                "per_note": True,
                "stage_check": "schedule_split",
                "scope": "line",
                "evidence": ["line_source"],
                "mechanism": "split_schedule",
                "fields": ["schedule"],
            },
            # ------------------------------------------------------------ line product identity
            "buyer_code": {
                "type": "noul",
                "instructions": "If this line uses a customer article code, is product_id the product mapped to that code for this buyer (buyer_id in doc_parties), not the mapping for another buyer? Otherwise answer yes.",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["line_source", "master_products_brief", "doc_parties", "doc_line"],
                "mechanism": "buyer_code",
                "fields": ["product_id"],
            },
            "product_variant": {
                "type": "noul",
                "instructions": "Does product_id in doc_line match the specification the source requires for this line (grade, material, size), rather than a similar but disallowed variant?",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["line_source", "master_products_brief", "doc_line"],
                "mechanism": "product_variant",
                "fields": ["product_id"],
            },
            "substitution": {
                "type": "noul",
                "instructions": "If an approved substitution replaces this line's product (for example with a revised item), is product_id the replacement product? Otherwise answer yes.",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["line_source", "master_products_brief", "doc_line"],
                "mechanism": "substitution",
                "fields": ["product_id"],
            },
            "repeated_item": {
                "type": "noul",
                "instructions": "If this line is an additional batch of the same item as another line, are both lines kept with the same product_id? Otherwise answer yes.",
                "polarity": "correct",
                "scope": "line",
                "evidence": ["line_source", "master_products_brief", "doc_line_ids", "line_ref"],
                "mechanism": "repeated_item",
                "fields": ["product_id"],
            },
        },
    },
}

# A stage PASSES when every param has a question and each param (on the training split) has:
#   - AUROC >= min_auroc              (only checked when the sample has >= min_errors wrong values)
#   - balanced accuracy >= min_balanced_accuracy   (P(correct) >= 0.5 counts as "says correct")
#   - no truncated calls              (unless allow_truncation)
GATE = {
    "min_auroc": 0.80,
    "min_balanced_accuracy": 0.75,
    "min_errors": 3,
    "allow_truncation": False,
}

# Param probabilities -> one confidence:  "product" | "min" | "geomean"
COMBINE = "product"

# Cumulative confidence after each stage:
#   "carry_forward"  for every field, the latest stage that judged it wins (later stages can fix
#                    earlier errors; the parser stage never carries).  After `validation` this is the
#                    final-output confidence.
#   "product"        P(every stage judged clean) -- strict; earlier fixed errors still count.
# Both are always reported; this one is the headline.
CUMULATIVE = "carry_forward"

# Threshold used for coverage metrics, and the risk target used to suggest a threshold:
# suggested = lowest threshold whose "incorrect among accepted" <= TARGET_RISK (max coverage).
THRESHOLD = 0.5
TARGET_RISK = 0.05
