# cloud_functions/llm-extractor/main.py
# ─────────────────────────────────────────────────────────────────────────────
# Reads per-listing JSONL records, fetches the original raw TXT from GCS,
# asks Gemini (Vertex AI) to extract structured fields, and writes a sibling
# "<post_id>_llm.jsonl" into the jsonl_llm/ sub-directory.
#
# EXTENDED FIELDS (new vs baseline):
#   color     – exterior colour, lowercase single word  (e.g. "silver")
#   city      – city where car is listed                (e.g. "Sacramento")
#   state     – 2-letter US state abbreviation          (e.g. "CA")
#   zip_code  – 5-digit ZIP string                      (e.g. "95814")
#
# FIXES INCLUDED:
#   1. LLM_MODEL = gemini-2.5-flash             (fixes 404/NotFound)
#   2. system_instruction merged into prompt    (fixes SDK TypeError)
#   3. "additionalProperties" removed           (fixes internal ParseError)
#   4. U+00A0 non-breaking spaces normalised
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone

from flask import Request, jsonify
from google.api_core import retry as gax_retry
from google.api_core.exceptions import (Aborted, DeadlineExceeded,
                                         InternalServerError, ResourceExhausted)
from google.cloud import storage

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

# ── ENV ───────────────────────────────────────────────────────────────────────
PROJECT_ID        = os.getenv("PROJECT_ID", "")
REGION            = os.getenv("REGION", "us-central1")
BUCKET_NAME       = os.getenv("GCS_BUCKET", "")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured")
LLM_MODEL         = os.getenv("LLM_MODEL", "gemini-2.5-flash")
OVERWRITE_DEFAULT = os.getenv("OVERWRITE", "false").lower() == "true"
MAX_FILES_DEFAULT = int(os.getenv("MAX_FILES", "0") or 0)

# ── Retry helpers ─────────────────────────────────────────────────────────────
READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0,
)

def _is_llm_retryable(exc):
    return isinstance(exc, (ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded))

# ── GCS client & model cache ──────────────────────────────────────────────────
storage_client   = storage.Client()
_CACHED_MODEL    = None

RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")


# ── Vertex AI ─────────────────────────────────────────────────────────────────
def _get_model() -> GenerativeModel:
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not PROJECT_ID:
            raise RuntimeError("PROJECT_ID env var is not set.")
        vertexai.init(project=PROJECT_ID, location=REGION)
        _CACHED_MODEL = GenerativeModel(LLM_MODEL)
        logging.info("Initialised Vertex AI model: %s in %s", LLM_MODEL, REGION)
    return _CACHED_MODEL


# JSON schema for Gemini structured output
_EXTRACT_SCHEMA = {
    "type": "object",
    # NOTE: "additionalProperties" intentionally omitted – causes ParseError in SDK
    "properties": {
        "price":        {"type": "integer",  "nullable": True},
        "year":         {"type": "integer",  "nullable": True},
        "make":         {"type": "string",   "nullable": True},
        "model":        {"type": "string",   "nullable": True},
        "mileage":      {"type": "integer",  "nullable": True},
        "transmission": {"type": "string",   "nullable": True},
        # ── NEW FIELDS ──────────────────────────────────────────────────────
        "color":        {"type": "string",   "nullable": True},
        "city":         {"type": "string",   "nullable": True},
        "state":        {"type": "string",   "nullable": True},
        "zip_code":     {"type": "string",   "nullable": True},
    },
    "required": [
        "price", "year", "make", "model", "mileage", "transmission",
        "color", "city", "state", "zip_code",
    ],
}

# System instruction is prepended to the user prompt (SDK compat)
_SYS_INSTR = (
    "Extract ONLY the following fields from the input text. "
    "Return a strict JSON object matching the provided schema. "
    "If a value is absent, use null. "
    "Rules:\n"
    "  - price:        integer USD, no symbols\n"
    "  - year:         4-digit integer model year\n"
    "  - mileage:      integer miles on the odometer\n"
    "  - transmission: 'automatic' | 'manual' | null\n"
    "  - color:        single lowercase word, exterior paint (e.g. 'black', 'silver')\n"
    "  - city:         city name where car is located (title case)\n"
    "  - state:        2-letter UPPERCASE US state abbreviation (e.g. 'CA')\n"
    "  - zip_code:     5-digit string (e.g. '90210')\n"
    "Do NOT infer values not explicitly present. Do NOT add extra keys."
)


def _extract_fields(raw_text: str) -> dict:
    """Call Gemini and return extracted field dict."""
    # Normalise non-breaking spaces (U+00A0 → U+0020)
    raw_text = raw_text.replace("\u00a0", " ")

    model  = _get_model()
    prompt = f"{_SYS_INSTR}\n\nTEXT:\n{raw_text}"

    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=40,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=_EXTRACT_SCHEMA,
    )

    resp = None
    for attempt in range(4):
        try:
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            break
        except Exception as exc:
            if not _is_llm_retryable(exc) or attempt == 3:
                logging.error("LLM call failed (attempt %d): %s", attempt + 1, exc)
                raise
            sleep = min(5.0 * (2 ** attempt), 30.0)
            logging.warning("Transient LLM error attempt %d – retrying in %.1fs: %s",
                            attempt + 1, sleep, exc)
            time.sleep(sleep)

    parsed = json.loads(resp.text)

    # ── Post-extraction normalisation ────────────────────────────────────────
    def _safe_int(v):
        try:
            return int(str(v).replace(",", "").strip()) if v not in (None, "") else None
        except (ValueError, TypeError):
            return None

    def _norm_str(v):
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    parsed["price"]    = _safe_int(parsed.get("price"))
    parsed["year"]     = _safe_int(parsed.get("year"))
    parsed["mileage"]  = _safe_int(parsed.get("mileage"))
    parsed["make"]         = _norm_str(parsed.get("make"))
    parsed["model"]        = _norm_str(parsed.get("model"))
    parsed["transmission"] = _norm_str(parsed.get("transmission"))
    parsed["color"]        = _norm_str(parsed.get("color"))
    parsed["city"]         = _norm_str(parsed.get("city"))
    parsed["state"]        = _norm_str(parsed.get("state"))
    parsed["zip_code"]     = _norm_str(parsed.get("zip_code"))

    # State should be uppercase 2-letter
    if parsed["state"] and len(parsed["state"]) == 2:
        parsed["state"] = parsed["state"].upper()

    return parsed


# ── GCS helpers ───────────────────────────────────────────────────────────────
def _list_run_ids(bucket: str, prefix: str) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{prefix}/", delimiter="/")
    for _ in it:
        pass
    runs = []
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            if RUN_ID_ISO_RE.match(rid) or RUN_ID_PLAIN_RE.match(rid):
                runs.append(rid)
    return sorted(runs)


def _normalize_run_id(run_id: str) -> str:
    try:
        if RUN_ID_ISO_RE.match(run_id):
            dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.strptime(run_id, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _list_input_jsonl(bucket: str, run_id: str) -> list[str]:
    prefix = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/"
    return [b.name for b in storage_client.bucket(bucket).list_blobs(prefix=prefix)
            if b.name.endswith(".jsonl")]


def _download(blob_name: str) -> str:
    return storage_client.bucket(BUCKET_NAME).blob(blob_name)\
                         .download_as_text(retry=READ_RETRY, timeout=120)


def _upload(blob_name: str, record: dict):
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    storage_client.bucket(BUCKET_NAME).blob(blob_name)\
                  .upload_from_string(line, content_type="application/x-ndjson")


def _exists(blob_name: str) -> bool:
    return storage_client.bucket(BUCKET_NAME).blob(blob_name).exists()


# ── HTTP entry-point ──────────────────────────────────────────────────────────
def llm_extract_http(request: Request):
    """
    HTTP POST trigger.  Optional JSON body:
        { "run_id": "...", "max_files": 50, "overwrite": false }

    For each listing in jsonl/:
      1. Reads the per-listing JSONL record.
      2. Fetches the raw .txt from GCS.
      3. Calls Gemini to extract all fields (incl. color/city/state/zip_code).
      4. Writes the enriched record to jsonl_llm/<post_id>_llm.jsonl.
    """
    logging.getLogger().setLevel(logging.INFO)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500
    if not PROJECT_ID:
        return jsonify({"ok": False, "error": "missing PROJECT_ID env"}), 500

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    run_id    = body.get("run_id")
    max_files = int(body.get("max_files") or MAX_FILES_DEFAULT or 0)
    overwrite = bool(body.get("overwrite")) if "overwrite" in body else OVERWRITE_DEFAULT

    if not run_id:
        runs = _list_run_ids(BUCKET_NAME, STRUCTURED_PREFIX)
        if not runs:
            return jsonify({"ok": False, "error": "no run_ids found"}), 200
        run_id = runs[-1]

    scraped_at_iso = _normalize_run_id(run_id)
    inputs = _list_input_jsonl(BUCKET_NAME, run_id)
    if not inputs:
        return jsonify({"ok": True, "run_id": run_id,
                        "processed": 0, "written": 0, "skipped": 0, "errors": 0}), 200
    if max_files > 0:
        inputs = inputs[:max_files]

    logging.info("run_id=%s | %d files to process", run_id, len(inputs))

    processed = written = skipped = errors = 0

    for in_key in inputs:
        processed += 1
        try:
            raw_line = _download(in_key).strip()
            if not raw_line:
                raise ValueError("empty input jsonl")
            base = json.loads(raw_line)

            post_id = base.get("post_id")
            src_txt = base.get("source_txt")
            if not post_id:
                raise ValueError("missing post_id")
            if not src_txt:
                raise ValueError("missing source_txt")

            # Output key under jsonl_llm/
            run_dir = in_key.rsplit("/jsonl/", 1)[0]
            out_key = f"{run_dir}/jsonl_llm/{post_id}_llm.jsonl"

            if not overwrite and _exists(out_key):
                skipped += 1
                continue

            raw_listing = _download(src_txt)
            fields = _extract_fields(raw_listing)

            out_record = {
                "post_id":      post_id,
                "run_id":       base.get("run_id", run_id),
                "scraped_at":   base.get("scraped_at", scraped_at_iso),
                "source_txt":   src_txt,
                # baseline fields
                "price":        fields.get("price"),
                "year":         fields.get("year"),
                "make":         fields.get("make"),
                "model":        fields.get("model"),
                "mileage":      fields.get("mileage"),
                "transmission": fields.get("transmission"),
                # ── NEW FIELDS ──────────────────────────
                "color":        fields.get("color"),
                "city":         fields.get("city"),
                "state":        fields.get("state"),
                "zip_code":     fields.get("zip_code"),
                # provenance
                "llm_provider": "vertex",
                "llm_model":    LLM_MODEL,
                "llm_ts":       datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

            _upload(out_key, out_record)
            written += 1

        except Exception as exc:
            errors += 1
            logging.error("Failed %s: %s\n%s", in_key, exc, traceback.format_exc())

    result = {
        "ok":        True,
        "run_id":    run_id,
        "processed": processed,
        "written":   written,
        "skipped":   skipped,
        "errors":    errors,
    }
    logging.info(json.dumps(result))
    return jsonify(result), 200
