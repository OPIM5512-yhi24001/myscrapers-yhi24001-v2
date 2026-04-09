# cloud_functions/materialize-llm/main.py
# ─────────────────────────────────────────────────────────────────────────────
# Build a single, ever-growing CSV from all LLM-enriched JSONL files.
#
# Reads:  gs://<bucket>/<STRUCTURED_PREFIX>/run_id=*/jsonl_llm/*.jsonl
# Writes: gs://<bucket>/<STRUCTURED_PREFIX>/datasets/listings_master_llm_v2.csv
#
# NEW vs baseline:
#   • Reads jsonl_llm/ (LLM output) with fallback to jsonl/ (regex output)
#   • CSV schema includes: color, city, state, zip_code
#   • De-dupes by post_id – newest run wins
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# ── ENV ───────────────────────────────────────────────────────────────────────
BUCKET_NAME       = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")

storage_client = storage.Client()

RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# ── CSV schema ────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage", "transmission",
    "color", "city", "state", "zip_code",   # NEW
    "source_txt",
]


# ── GCS helpers ───────────────────────────────────────────────────────────────

def _list_run_ids(bucket: str, prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{prefix}/", delimiter="/")
    for _ in it:
        pass
    run_ids = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                run_ids.append(rid)
    return sorted(run_ids)


def _jsonl_records_for_run(bucket: str, prefix: str, run_id: str):
    """
    Yield enriched dicts for a given run.
    Prefers jsonl_llm/ (LLM-extracted); falls back to jsonl/ (regex-only).
    """
    b = storage_client.bucket(bucket)
    for sub in [f"{prefix}/run_id={run_id}/jsonl_llm/",
                f"{prefix}/run_id={run_id}/jsonl/"]:
        blobs = [bl for bl in b.list_blobs(prefix=sub) if bl.name.endswith(".jsonl")]
        if not blobs:
            continue
        for blob in blobs:
            for line in blob.download_as_text().strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec.setdefault("run_id",   run_id)
                    rec.setdefault("color",    None)
                    rec.setdefault("city",     None)
                    rec.setdefault("state",    None)
                    rec.setdefault("zip_code", None)
                    yield rec
                except Exception:
                    continue
        break  # stop after first non-empty dir


def _run_id_to_dt(rid: str) -> datetime:
    try:
        if RUN_ID_ISO_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if RUN_ID_PLAIN_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    return datetime.now(timezone.utc)


def _write_csv(records: Iterable[Dict], dest_key: str) -> int:
    blob = storage_client.bucket(BUCKET_NAME).blob(dest_key)
    n = 0
    with blob.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({c: rec.get(c) for c in CSV_COLUMNS})
            n += 1
    return n


# ── Cloud Function entry-point ────────────────────────────────────────────────

def materialize_http(request: Request):
    """
    HTTP POST trigger (no body required).
    Scans all run folders, de-dupes by post_id (newest run wins),
    writes listings_master_llm_v2.csv.
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "GCS_BUCKET env var not set"}), 500

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not run_ids:
            return jsonify({"ok": False,
                            "error": f"No run_id= folders under {STRUCTURED_PREFIX}/"}), 200

        latest_by_post: Dict[str, Dict] = {}
        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid:
                    continue
                prev = latest_by_post.get(pid)
                if prev is None or (
                    _run_id_to_dt(rec.get("run_id", rid))
                    > _run_id_to_dt(prev.get("run_id", ""))
                ):
                    latest_by_post[pid] = rec

        dest_key = f"{STRUCTURED_PREFIX}/datasets/listings_master_llm_v2.csv"
        rows = _write_csv(latest_by_post.values(), dest_key)

        return jsonify({
            "ok":              True,
            "runs_scanned":    len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written":    rows,
            "output_csv":      f"gs://{BUCKET_NAME}/{dest_key}",
        }), 200

    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
