# Decision Tree (baseline) + Random Forest with Optuna tuning (LLM model)
# HTTP entrypoint: train_dt_http

import os, io, json, logging, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master.csv")
DATA_KEY_LLM   = os.getenv("DATA_KEY_LLM", "structured/datasets/listings_master_llm_updated.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "structured/preds")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")
OPTUNA_TRIALS  = int(os.getenv("OPTUNA_TRIALS", "20"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _read_csv_from_gcs(client, bucket, key):
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client, bucket, key, df):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def _write_bytes_to_gcs(client, bucket, key, data, content_type):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(data, content_type=content_type)

def _clean_numeric(s):
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _prepare_df(df):
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"]  = df["scraped_at_local"].dt.date
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])
    df["vehicle_age"] = pd.Timestamp.now().year - df["year_num"]

    # Filter bad scrapes
    bad = (df["make"].str.lower().str.strip() == "contact") & \
          (df["model"].str.lower().str.strip() == "information")
    df = df[~bad].copy()
    return df


# ── OPTUNA TUNING ──────────────────────────────────────────────────────────────

def _tune_rf(X_train, y_train, n_trials=20):
    """Run Optuna to find best RandomForest hyperparameters."""

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth":         trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        preds = m.predict(X_train)
        return mean_absolute_error(y_train, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logging.info("Optuna best params: %s  MAE=%.2f",
                 study.best_params, study.best_value)
    return study.best_params


# ── PERMUTATION IMPORTANCE ─────────────────────────────────────────────────────

def _save_permutation_importance(client, pipe, X_val, y_val,
                                  feats, out_folder):
    """Compute permutation importance, save CSV + PNG to GCS."""
    result = permutation_importance(
        pipe, X_val, y_val,
        n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature":    feats,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_key = f"{OUTPUT_PREFIX}/{out_folder}/importance.csv"
    _write_csv_to_gcs(client, GCS_BUCKET, csv_key, imp_df)
    logging.info("Saved importance.csv → gs://%s/%s", GCS_BUCKET, csv_key)

    # Save plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(imp_df)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
            xerr=imp_df["std"][::-1], color=colors[::-1],
            align="center", capsize=3)
    ax.set_xlabel("Mean importance (MAE decrease)")
    ax.set_title("Permutation feature importance")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)

    png_key = f"{OUTPUT_PREFIX}/{out_folder}/importance.png"
    _write_bytes_to_gcs(client, GCS_BUCKET, png_key,
                        buf.read(), "image/png")
    logging.info("Saved importance.png → gs://%s/%s", GCS_BUCKET, png_key)

    return imp_df


# ── PDP PLOTS ──────────────────────────────────────────────────────────────────

def _save_pdp_plots(client, pipe, X_val, feats, imp_df, out_folder):
    """Generate PDP plots for top 3 features, save to GCS."""
    top3 = imp_df["feature"].head(3).tolist()
    top3 = [f for f in top3 if f in feats]

    for feat in top3:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            PartialDependenceDisplay.from_estimator(
                pipe, X_val, features=[feat],
                ax=ax, grid_resolution=50,
                line_kw={"color": "#2563eb", "linewidth": 2},
            )
            ax.set_title(f"Partial dependence — {feat}")
            ax.set_ylabel("Predicted price ($)")
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130)
            plt.close(fig)
            buf.seek(0)

            safe = feat.replace(" ", "_")
            png_key = f"{OUTPUT_PREFIX}/{out_folder}/pdp_{safe}.png"
            _write_bytes_to_gcs(client, GCS_BUCKET, png_key,
                                 buf.read(), "image/png")
            logging.info("Saved PDP for %s → gs://%s/%s",
                         feat, GCS_BUCKET, png_key)
        except Exception as e:
            logging.warning("PDP failed for %s: %s", feat, e)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def run_once(dry_run=False, max_depth=12, min_samples_leaf=10):
    client  = storage.Client(project=PROJECT_ID)
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    out_folder = now_utc.strftime('%Y%m%d%H')

    # ── ORIGINAL MODEL (preds.csv) ─────────────────────────────────────────
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)
    df = _prepare_df(df)

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates"}

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] <  today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()
    train_df   = train_df[train_df["price_num"].notna()]

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows"}

    target   = "price_num"
    cat_cols = ["make", "model"]
    num_cols = ["year_num", "mileage_num"]
    feats    = cat_cols + num_cols

    pre = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre),
                     ("model", DecisionTreeRegressor(
                         max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         random_state=42))])
    pipe.fit(train_df[feats], train_df[target])

    mae_today = None
    preds_df  = pd.DataFrame()
    if not holdout_df.empty:
        y_hat    = pipe.predict(holdout_df[feats])
        cols     = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price"]
        preds_df = holdout_df[cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]
        preds_df["pred_price"]   = np.round(y_hat, 2)
        mask = holdout_df["price_num"].notna()
        if mask.any():
            mae_today = float(mean_absolute_error(
                holdout_df["price_num"][mask], y_hat[mask]))

    out_key = f"{OUTPUT_PREFIX}/{out_folder}/preds.csv"
    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, out_key, preds_df)
        logging.info("Wrote preds.csv → gs://%s/%s", GCS_BUCKET, out_key)

    # ── LLM MODEL with OPTUNA (preds_llm_updated.csv) ─────────────────────────────
    mae_llm      = None
    preds_llm_df = pd.DataFrame()
    best_params  = {}

    try:
        df_llm = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY_LLM)
        df_llm = _prepare_df(df_llm)

        unique_dates_llm = sorted(d for d in df_llm["date_local"].dropna().unique())
        if len(unique_dates_llm) >= 2:
            today_llm   = unique_dates_llm[-1]
            train_llm   = df_llm[df_llm["date_local"] <  today_llm].copy()
            holdout_llm = df_llm[df_llm["date_local"] == today_llm].copy()
            train_llm   = train_llm[train_llm["price_num"].notna()]

            if len(train_llm) >= 40:
                cat_cols_llm = ["make", "model", "transmission", "color", "state", "city", "zip_code"]
                num_cols_llm = ["year_num", "mileage_num", "vehicle_age"]
                feats_llm    = cat_cols_llm + num_cols_llm

                # Fill missing new columns
                for c in ["transmission", "color", "state", "city", "zip_code"]:
                    if c not in train_llm.columns:
                        train_llm[c] = "unknown"
                    if c not in holdout_llm.columns:
                        holdout_llm[c] = "unknown"
                    train_llm[c]   = train_llm[c].fillna("unknown")
                    holdout_llm[c] = holdout_llm[c].fillna("unknown")

                pre_llm = ColumnTransformer(transformers=[
                    ("num", SimpleImputer(strategy="median"), num_cols_llm),
                    ("cat", Pipeline([
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh",  OneHotEncoder(handle_unknown="ignore"))
                    ]), cat_cols_llm),
                ])

                # Preprocess for Optuna
                X_tr = pre_llm.fit_transform(train_llm[feats_llm])
                y_tr = train_llm[target].values

                # Run Optuna tuning
                logging.info("Running Optuna tuning (%d trials)...", OPTUNA_TRIALS)
                best_params = _tune_rf(X_tr, y_tr, n_trials=OPTUNA_TRIALS)

                # Refit full pipeline with best params
                pre_llm2 = ColumnTransformer(transformers=[
                    ("num", SimpleImputer(strategy="median"), num_cols_llm),
                    ("cat", Pipeline([
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh",  OneHotEncoder(handle_unknown="ignore"))
                    ]), cat_cols_llm),
                ])
                pipe_llm = Pipeline([
                    ("pre", pre_llm2),
                    ("model", RandomForestRegressor(
                        **best_params,
                        random_state=42,
                        n_jobs=-1)),
                ])
                pipe_llm.fit(train_llm[feats_llm], train_llm[target])

                # Predictions
                if not holdout_llm.empty:
                    y_hat_llm    = pipe_llm.predict(holdout_llm[feats_llm])
                    cols_llm     = ["post_id", "scraped_at", "make", "model",
                                    "year", "mileage", "price",
                                    "transmission", "color", "state", "zip_code", "city"]
                    cols_llm     = [c for c in cols_llm if c in holdout_llm.columns]
                    preds_llm_df = holdout_llm[cols_llm].copy()
                    preds_llm_df["actual_price"] = holdout_llm["price_num"]
                    preds_llm_df["pred_price"]   = np.round(y_hat_llm, 2)
                    mask_llm = holdout_llm["price_num"].notna()
                    if mask_llm.any():
                        mae_llm = float(mean_absolute_error(
                            holdout_llm["price_num"][mask_llm],
                            y_hat_llm[mask_llm]))

                    out_key_llm = f"{OUTPUT_PREFIX}/{out_folder}/preds_llm.csv"
                    if not dry_run and len(preds_llm_df) > 0:
                        _write_csv_to_gcs(client, GCS_BUCKET,
                                          out_key_llm, preds_llm_df)
                        logging.info("Wrote preds_llm.csv → gs://%s/%s",
                                     GCS_BUCKET, out_key_llm)

                # Permutation importance + PDPs on holdout
               
                if not dry_run and not holdout_llm.empty:
                    pi_mask = holdout_llm[target].notna()
                    imp_df = _save_permutation_importance(
                        client, pipe_llm,
                        holdout_llm[feats_llm][pi_mask],
                        holdout_llm[target][pi_mask],
                        feats_llm, out_folder,
                    
                    )
                    _save_pdp_plots(
                        client, pipe_llm,
                        train_llm[feats_llm],
                        feats_llm, imp_df, out_folder,
                    )

    except Exception as e:
        logging.warning("LLM model failed (non-fatal): %s\n%s",
                        e, traceback.format_exc())

    return {
        "status":       "ok",
        "today_local":  str(today_local),
        "train_rows":   int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae_today":    mae_today,
        "mae_llm":      mae_llm,
        "best_params":  best_params,
        "output_key":   out_key,
        "dry_run":      dry_run,
        "timezone":     TIMEZONE,
    }


def train_dt_http(request):
    try:
        body   = request.get_json(silent=True) or {}
        result = run_once(
            dry_run          = bool(body.get("dry_run", False)),
            max_depth        = int(body.get("max_depth", 12)),
            min_samples_leaf = int(body.get("min_samples_leaf", 10)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500,
                {"Content-Type": "application/json"})
