# main.py - UPDATED MATERIALIZER 

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")

storage_client = storage.Client()

# Accept BOTH runID formats
RUN_ID_ISO_RE = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

# CSV schema (includes new fields)
CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage", "transmission",
    "color", "city", "state", "zip_code",
    "source_txt"
]

# -------------------- HELPERS --------------------

def _list_run_ids(bucket: str, structured_prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
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


def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm_updated/"

    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue

        data = blob.download_as_text()

        # FIX: proper JSONL parsing
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec.setdefault("run_id", run_id)
                yield rec
            except json.JSONDecodeError:
                continue


def _run_id_to_dt(rid: str) -> datetime:
    try:
        if RUN_ID_ISO_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if RUN_ID_PLAIN_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _open_gcs_text_writer(bucket: str, key: str):
    b = storage_client.bucket(bucket)
    blob = b.blob(key)
    return blob.open("w", encoding="utf-8")


def _write_csv(records: Iterable[Dict], dest_key: str) -> int:
    n = 0
    with _open_gcs_text_writer(BUCKET_NAME, dest_key) as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for rec in records:
            row = {c: rec.get(c, None) for c in CSV_COLUMNS}
            writer.writerow(row)
            n += 1

    return n

# -------------------- MAIN CLOUD FUNCTION --------------------

def materialize_updated_http(request: Request):
    """
    HTTP POST:
    - Reads all structured JSONL runs
    - Deduplicates by post_id (keeps newest run)
    - Writes master CSV
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        run_ids = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not run_ids:
            return jsonify({
                "ok": False,
                "error": f"no runs found under {STRUCTURED_PREFIX}/"
            }), 200

        latest_by_post: Dict[str, Dict] = {}

        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid:
                    continue

                prev = latest_by_post.get(pid)

                rec_dt = _run_id_to_dt(rec.get("run_id", rid))
                prev_dt = _run_id_to_dt(prev.get("run_id")) if prev else None

                if (prev is None) or (rec_dt > prev_dt):
                    latest_by_post[pid] = rec

        final_key = f"{STRUCTURED_PREFIX}/datasets/listings_master_llm_updated.csv"
        rows = _write_csv(latest_by_post.values(), final_key)

        return jsonify({
            "ok": True,
            "version": "materialize-llm-updated-v2-fixed",
            "runs_scanned": len(run_ids),
            "unique_listings": len(latest_by_post),
            "rows_written": rows,
            "columns": len(CSV_COLUMNS),
            "new_fields": ["color", "city", "state", "zip_code"],
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"{type(e).__name__}: {e}"
        }), 500
