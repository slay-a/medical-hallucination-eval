#!/usr/bin/env python3
"""
audit_map_types.py: Fill the "what is wrong" type of P/U audit labels from the annotator's own free-text notes.

The annotator recorded the reason for every partly or not supported label as free text; this script coarsens
those reasons into the protocol's five categories with deterministic keyword rules.  The first category in
precedence order whose keyword appears in the note wins (specific beats general: a wrong dose outranks the
not-stated wording that usually accompanies it, matching the protocol's own examples).  Notes that match no
rule keep an empty type for the annotator to set in the labeling tool, and types already set in the tool are
never overwritten.  The annotator reviews the result in the tool, where each item shows the chosen category
above the note it came from.  Prints category counts only, never note or claim text.  Run with the audit tool
stopped; a timestamped backup of the labels file is written first.
"""
import json
import os
import shutil
import time
from pathlib import Path

PRIVATE = Path(__file__).resolve().parent / "results_private"
LABELS = Path(os.environ.get("AUDIT_LABELS", PRIVATE / "audit_labels.json"))
BACKUPS = PRIVATE / "audit_backups"

RULES = [
    ("med_number", ["dose", "dosage", " mg", "mcg", " ml ", "number", "wrong med", "wrong drug", "medication name",
                    "date", "frequency", "twice", "daily", "weekly", "duration", "days", "weeks", "months", "side",
                    "lateral", "unit", "amount", "quantity", "percent", "value", "wrong time"]),
    ("generic", ["generic", "advice", "follow up", "follow-up", "followup", "boilerplate", "standard", "routine"]),
    ("contradicts", ["contradict", "opposite", "conflict", "disagree", "reversed"]),
    ("other", ["certain", "possible", "likely", "rule out", "overstat", "presented as", "other"]),
    ("not_stated", ["not stated", "not in", "not mentioned", "no mention", "nowhere", "does not say", "doesn't say",
                    "doesnt say", "never", "absent", "not supported", "can't find", "cant find", "couldn",
                    "not specified", "not said", "no info", "missing", "made up", "invented", "added", "extra",
                    "assumed", "inferred", "fill", "blank", "unstated", "not discussed", "not noted",
                    "not documented", "no evidence", "not found"]),
]


def classify(note):
    low = " " + note.lower().strip() + " "
    for cat, keys in RULES:
        if any(k in low for k in keys):
            return cat
    return None


def main():
    labels = json.loads(LABELS.read_text())
    BACKUPS.mkdir(exist_ok=True)
    shutil.copy2(LABELS, BACKUPS / f"{LABELS.stem}-premap-{time.strftime('%Y%m%d-%H%M%S')}.json")
    counts, unmapped, kept = {}, 0, 0
    for rec in labels.values():
        if rec.get("label") not in ("P", "U"):
            continue
        if rec.get("type"):
            kept += 1
            continue
        cat = classify(rec.get("note") or "")
        if cat:
            rec["type"] = cat
            counts[cat] = counts.get(cat, 0) + 1
        else:
            unmapped += 1
    tmp = LABELS.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, indent=1))
    tmp.replace(LABELS)
    print(f"labels: {LABELS}")
    print(f"types kept as set in the tool: {kept}")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"mapped to {cat}: {n}")
    print(f"left for the annotator (no rule matched): {unmapped}")


if __name__ == "__main__":
    main()
