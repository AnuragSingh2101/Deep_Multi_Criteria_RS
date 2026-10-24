
"""Two-stage multi-criteria collaborative filtering (Nassar et al., 2020).

Install: pip install numpy pandas scikit-learn torch matplotlib scikit-surprise

Run:
    python main.py                 # reuses completed fold/model results
    python main.py --force         # reruns all models
    python main.py --epochs 20     # change experimental budget

Input:
    dataset.csv columns:
        user_id,hotel_id,value,rooms,location,cleanliness,
        check_in,service,business_service,overall_rating
    No synthetic data is used. Ratings must be in [1,5].
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 2. CONSTANTS
# =============================================================================
# The seven criteria ratings the paper uses (TripAdvisor hotel schema).
CRITERIA = [
    "value", "rooms", "location", "cleanliness",
    "check_in", "service", "business_service",
]

# Full required schema: IDs + 7 criteria + 1 overall rating.
REQUIRED = ["user_id", "hotel_id"] + CRITERIA + ["overall_rating"]

# The seven model configurations compared in the paper (Table 6).
MODELS = [
    "SVD", "SVD++", "SlopeOne", "Baseline estimates",
    "KNN with baseline", "Single DNN", "Proposed Model",
]

# The six evaluation metrics (Table 6 rows).
METRICS = ["MAE", "F1", "F2", "FCP", "MAP", "MRR"]

# --- PAPER REFERENCE VALUES ONLY (Table 6) --------------------------------
# These are the numbers reported in the paper. They are NEVER written to
# "table6_results.csv" as if measured. They are written only to
# "paper_reference_table6.csv" for side-by-side comparison.
PUBLISHED = {
    "SVD":                [0.8071, 0.8370, 0.7951, 0.5024, 0.3935, 0.7272],
    "SVD++":              [0.8222, 0.8390, 0.7991, 0.5092, 0.3483, 0.7513],
    "SlopeOne":           [0.9160, 0.8666, 0.9248, 0.5096, 0.3006, 0.7502],
    "Baseline estimates": [0.8094, 0.8391, 0.7986, 0.5025, 0.3881, 0.7374],
    "KNN with baseline":  [0.8115, 0.8388, 0.7984, 0.5097, 0.3865, 0.7360],
    "Single DNN":         [0.7855, 0.8572, 0.9176, 0.4957, 0.4481, 0.7244],
    "Proposed Model":     [0.7552, 0.8769, 0.9295, 0.5126, 0.4643, 0.7664],
}


# =============================================================================
# 3. REPRODUCIBILITY
# =============================================================================
def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 4. DATASET LOADING + VALIDATION
# =============================================================================
def load_dataset(path: Path) -> pd.DataFrame:
    """Read and validate the multi-criteria dataset.

    Raises FileNotFoundError if the file is missing, ValueError if any
    required column is absent or values are out of range. Nothing is imputed.
    """
    # 4a. Fail loudly if dataset.csv is absent — never synthesise ratings.
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Extract/prepare genuine multi-criteria ratings "
            f"first. The supplied RAR contains hotel review text files that must "
            f"be parsed into dataset.csv first.\n"
            f"Required CSV header: {','.join(REQUIRED)}"
        )

    # 4b. Read the CSV.
    df = pd.read_csv(path, low_memory=False)

    # 4c. Normalise column names (strip, lowercase, unify separators).
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    ]

    # 4d. Map common alternative spellings to the canonical names.
    aliases = {
        "check_in_front_desk": "check_in",
        "check_in__front_desk": "check_in",
        "business_services": "business_service",
        "overall": "overall_rating",
        "user": "user_id",
        "item_id": "hotel_id",
    }
    df = df.rename(columns={k: v for k, v in aliases.items()
                            if k in df and v not in df})

    # 4e. Verify that all required columns are present.
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(
            f"Dataset is missing {missing}. Required: {REQUIRED}. "
            f"No missing criteria are fabricated."
        )

    # 4f. Keep only the required columns.
    df = df[REQUIRED].copy()

    # 4g. Coerce rating columns to numeric — non-numeric becomes NaN.
    for c in CRITERIA + ["overall_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4h. Reject any nulls (never impute).
    if df.isna().any().any():
        raise ValueError(
            "Null/non-numeric IDs or ratings found; clean source data before "
            "training."
        )

    # 4i. Verify all ratings are within [1, 5].
    if not df[CRITERIA + ["overall_rating"]].apply(
        lambda x: x.between(1, 5).all()
    ).all():
        raise ValueError(
            "Ratings must be within [1,5]. Filter incomplete rows "
            "(e.g., -1 means missing), never impute fake ratings."
        )

    # 4j. Force IDs to strings so the grouping key is stable.
    # NOTE: duplicate (user, hotel) rows are retained — they may be distinct
    # real reviews of the same hotel by the same user. GroupKFold keeps them
    # in the same split so they can never leak across train/test.
    df["user_id"] = df["user_id"].astype(str)
    df["hotel_id"] = df["hotel_id"].astype(str)
    return df.reset_index(drop=True)


# =============================================================================
# 5. ID ENCODING
# =============================================================================
def encode_fold(train: pd.DataFrame, test: pd.DataFrame):
    """Map string IDs to integer indices.

    Index 0 is reserved for unknown users/items (padding index).
    Vocabularies are built from `train` only, so no test IDs influence
    the embedding tables — this prevents test leakage.
    """
    # 5a. Build vocabulary: 1-based index so 0 can mean "unknown".
    u = {v: i + 1 for i, v in enumerate(train["user_id"].unique())}
    h = {v: i + 1 for i, v in enumerate(train["hotel_id"].unique())}

    # 5b. Encode a frame; unseen IDs become 0.
    def enc(d: pd.DataFrame) -> np.ndarray:
        return np.stack(
            [
                d["user_id"].map(u).fillna(0).to_numpy(dtype="int64"),
                d["hotel_id"].map(h).fillna(0).to_numpy(dtype="int64"),
            ],
            axis=1,
        )

    # 5c. Return encoded arrays + vocabulary sizes (include the +1 for unknown).
    return enc(train), enc(test), len(u) + 1, len(h) + 1


# =============================================================================
# 6. NEURAL-NETWORK BUILDERS
# =============================================================================
def make_net(n_users: int, n_hotels: int, embedding: int,
             hidden: list, out: int):
    """Build a user+hotel embedding → MLP network.

    Architecture (both paper DNNs share this shape):
        [user_emb ‖ hotel_emb] → hidden[0] → ... → hidden[-1] → out
        All layers use ReLU (the paper applies ReLU on the output layer too).
    Weight init: N(0, 0.05²) per paper §4.3.
    """
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # 6a. Embedding tables (index 0 = padding / unknown).
            self.u = nn.Embedding(n_users, embedding, padding_idx=0)
            self.i = nn.Embedding(n_hotels, embedding, padding_idx=0)

            # 6b. Stack hidden layers with ReLU activations.
            layers = []
            prev = embedding * 2          # concatenation doubles the width
            for size in hidden:
                layers += [nn.Linear(prev, size), nn.ReLU()]
                prev = size

            # 6c. Final output layer (ReLU per paper Eq. 4/9; 7 or 1 neurons).
            layers += [nn.Linear(prev, out), nn.ReLU()]
            self.layers = nn.Sequential(*layers)

            # 6d. Initialise weights per paper.
            self.apply(self.init_weights)

        @staticmethod
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            # 6e. Concatenate user + hotel embeddings, then pass through MLP.
            return self.layers(
                torch.cat((self.u(x[:, 0]), self.i(x[:, 1])), dim=-1)
            )

    return Net()


def make_aggregator():
    """Stage-2 DNN: map 7 normalised criteria ratings → overall rating."""
    import torch
    from torch import nn

    # 6f. Hidden layers [64, 32, 16, 8] with ReLU, then 1 output neuron.
    layers = []
    d = 7
    for h in [64, 32, 16, 8]:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 1), nn.ReLU()]
    net = nn.Sequential(*layers)

    # 6g. Same init scheme.
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.05)
            nn.init.zeros_(m.bias)
    return net


# =============================================================================
# 7. GENERIC TRAINING LOOP
# =============================================================================
def train_network(net, x, y, vx, vy, args, device):
    """Train a network with Adam + MAE loss and early stopping.

    Args:
        net      : torch.nn.Module
        x, y     : training arrays (IDs as int64, or floats as float32)
        vx, vy   : validation arrays (used for early stopping)
        args     : argparse namespace
        device   : 'cpu' or 'cuda'
    Returns:
        trained network loaded with the best-validation state.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # 7a. Move model to device and choose the right input dtype.
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)  # paper §4.3

    x_dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32
    train_ds = TensorDataset(
        torch.tensor(x, dtype=x_dtype),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)

    vx_t = torch.tensor(vx, dtype=x_dtype, device=device)
    vy_t = torch.tensor(vy, dtype=torch.float32, device=device)

    # 7b. Early-stopping state.
    best = float("inf")
    best_state = None
    wait = 0

    # 7c. Training loop.
    for epoch in range(args.epochs):
        net.train()
        loss_sum = 0.0
        for a, b in loader:
            a = a.to(device)
            b = b.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.abs(net(a) - b).mean()   # MAE
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(a)

        # 7d. Validation loss.
        net.eval()
        with torch.no_grad():
            vloss = float(torch.abs(net(vx_t) - vy_t).mean())

        # 7e. Save the best state; count patience.
        if vloss < best - 1e-5:
            best = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1

        # 7f. Progress logging.
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1}/{args.epochs}: "
                f"train_MAE={loss_sum/len(train_ds):.4f}, val_MAE={vloss:.4f}",
                flush=True,
            )

        # 7g. Early stop.
        if wait >= args.patience:
            break

    # 7h. Restore best weights and return.
    net.load_state_dict(best_state)
    net.to(device)
    return net


# =============================================================================
# 8. INFERENCE HELPER
# =============================================================================
def predict(net, x, device, batch_size=4096):
    """Batched inference; returns a NumPy array."""
    import torch
    net.eval()
    parts = []
    x_dtype = torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            a = torch.tensor(x[start:start + batch_size],
                             dtype=x_dtype, device=device)
            parts.append(net(a).cpu().numpy())
    return np.concatenate(parts, axis=0)


# =============================================================================
# 9. THE TWO DEEP MODELS
# =============================================================================
def train_deep(train, test, args, fold, model_name, out, device):
    """Train a deep model on one fold.

    Returns:
        (overall_predictions, criteria_predictions_or_None)
    """
    from sklearn.model_selection import train_test_split
    import torch

    # 9a. Carve out a small validation subset from the TRAINING fold
    #     (the paper uses 80% train incl. 1% validation, 20% test).
    idx = np.arange(len(train))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=max(0.01, min(0.2, args.validation_fraction)),
        random_state=args.seed + fold,
    )
    trdf = train.iloc[tr_idx]
    vadf = train.iloc[va_idx]

    # 9b. Encode IDs.
    # FIX (bug from original draft): we encode the validation set using the
    # *training* vocabulary, so validation users get real (randomly
    # initialised) embeddings instead of index 0. In the original code,
    # `_, x_val, _, _ = encode_fold(trdf, vadf)` re-built the vocabulary from
    # trdf, so every validation user fell into the padding index and validation
    # loss was meaningless.
    x_train, _, nu, nh = encode_fold(trdf, trdf)          # vocab from trdf
    _, x_test, _, _   = encode_fold(trdf, test)
    _, x_val, _, _    = encode_fold(trdf, vadf)           # uses trdf vocab
    # NOTE: `x_train` and `x_test`/`x_val` above must share one vocabulary.
    # To guarantee that, rebuild with the same mapping:
    u = {v: i + 1 for i, v in enumerate(trdf["user_id"].unique())}
    h = {v: i + 1 for i, v in enumerate(trdf["hotel_id"].unique())}

    def _enc(d):
        return np.stack(
            [
                d["user_id"].map(u).fillna(0).to_numpy(dtype="int64"),
                d["hotel_id"].map(h).fillna(0).to_numpy(dtype="int64"),
            ],
            axis=1,
        )

    x_train, x_val, x_test = _enc(trdf), _enc(vadf), _enc(test)
    nu, nh = len(u) + 1, len(h) + 1

    # 9c. ---- Single DNN path -------------------------------------------
    if model_name == "Single DNN":
        # Paper: embedding=256, hidden=[512,256,128,64,32,16,8], 1 output.
        net = make_net(nu, nh, 256, [512, 256, 128, 64, 32, 16, 8], 1)
        net = train_network(
            net, x_train, trdf[["overall_rating"]].to_numpy("float32"),
            x_val, vadf[["overall_rating"]].to_numpy("float32"),
            args, device,
        )
        result = predict(net, x_test, device).ravel()
        torch.save(net.cpu().state_dict(), out / "single_dnn.pt")
        return result, None

    # 9d. ---- Proposed two-stage model ---------------------------------
    # Stage 1: criteria-rating DNN (7 outputs).
    criteria_net = make_net(nu, nh, 64, [128, 64, 32, 16, 8], 7)
    criteria_net = train_network(
        criteria_net,
        x_train, trdf[CRITERIA].to_numpy("float32"),
        x_val,   vadf[CRITERIA].to_numpy("float32"),
        args, device,
    )

    # 9e. Predict criteria ratings on the TEST set for later aggregation.
    predicted_criteria = predict(criteria_net, x_test, device)
    torch.save(criteria_net.cpu().state_dict(), out / "criteria_dnn.pt")

    # 9f. Normalisation statistics from training rows (paper Eq. 7).
    mu  = trdf[CRITERIA].mean().to_numpy("float32")
    std = trdf[CRITERIA].std(ddof=0).to_numpy("float32")
    std[std == 0] = 1.0  # guard against zero-variance columns

    # 9g. Stage 2 trains on GROUND-TRUTH training criteria, NOT on Stage 1
    #     predictions — this is exactly what Fig. 2 of the paper shows.
    norm_train = (trdf[CRITERIA].to_numpy("float32") - mu) / std
    norm_val   = (vadf[CRITERIA].to_numpy("float32") - mu) / std

    aggregator = make_aggregator()
    aggregator = train_network(
        aggregator,
        norm_train, trdf[["overall_rating"]].to_numpy("float32"),
        norm_val,   vadf[["overall_rating"]].to_numpy("float32"),
        args, device,
    )

    # 9h. At inference: normalise Stage-1 predictions, feed to Stage 2.
    result = predict(aggregator,
                     (predicted_criteria - mu) / std,
                     device).ravel()

    # 9i. Persist Stage-2 weights + normalisation stats together.
    torch.save(
        {"state_dict": aggregator.cpu().state_dict(), "mean": mu, "std": std},
        out / "overall_dnn.pt",
    )
    return result, predicted_criteria


# =============================================================================
# 10. CLASSICAL BASELINES (scikit-surprise)
# =============================================================================
def train_surprise(train, test, name):
    """Fit one of the five classical baselines and predict on `test`."""
    try:
        from surprise import (
            Dataset, Reader, SVD, SVDpp, SlopeOne, BaselineOnly, KNNBaseline,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The five classical baselines require scikit-surprise. Install "
            "with: pip install scikit-surprise"
        ) from exc

    # 10a. Map model name → constructor.
    mapping = {
        "SVD":                lambda: SVD(random_state=42),
        "SVD++":              lambda: SVDpp(random_state=42),
        "SlopeOne":           SlopeOne,
        "Baseline estimates": BaselineOnly,
        "KNN with baseline":  lambda: KNNBaseline(
            sim_options={"name": "msd", "user_based": False}, verbose=False
        ),
    }

    # 10b. Load ratings into Surprise and fit on the full training set.
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        train[["user_id", "hotel_id", "overall_rating"]], reader
    )
    algo = mapping[name]()
    algo.fit(data.build_full_trainset())

    # 10c. Predict each test pair.
    return np.array(
        [algo.predict(u, i).est for u, i in zip(test["user_id"], test["hotel_id"])],
        dtype=float,
    )


# =============================================================================
# 11. EVALUATION METRICS
# =============================================================================
def calculate_mae(y_true, y_pred):
    """Mean Absolute Error (paper Eq. 10)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rank_metrics(test, pred, k, relevant):
    """Compute all six metrics for one (model, fold).

    Protocol (see experiment_details.json):
      - Ranking is over the user's *held-out* test items only, not the full
        unobserved catalog.
      - An item is "relevant" if its true overall_rating >= `relevant`.
      - Precision/recall/AP/MRR are macro-averaged over users.
      - FCP pools comparable pairs across users; ties are excluded.
    """
    # 11a. Attach predictions to the test frame.
    d = test[["user_id", "hotel_id", "overall_rating"]].copy()
    d["pred"] = pred

    # 11b. Accumulators.
    ap, rr, prec, rec = [], [], [], []
    con, dis = 0, 0

    # 11c. Per-user loop.
    for _, g in d.groupby("user_id", sort=False):
        y = g["overall_rating"].to_numpy()
        p = g["pred"].to_numpy()

        # Rank by descending prediction (stable → deterministic ties).
        order = np.argsort(-p, kind="stable")
        y_sorted = y[order]

        # Hits in top-k and total relevant items in the held-out set.
        hits = (y_sorted[:k] >= relevant).astype(float)
        total = int((y >= relevant).sum())
        top = len(hits)
        if not top:
            continue

        # Precision@min(k, |test items|) and recall@min(k, |test items|).
        prec.append(float(hits.sum() / top))
        rec.append(float(hits.sum() / total) if total else 0.0)

        # Average Precision (Eq. 19).
        if total:
            ap.append(
                float((hits * np.cumsum(hits) / np.arange(1, top + 1)).sum()
                      / min(k, total))
            )
        else:
            ap.append(0.0)

        # Reciprocal rank of the first relevant item (Eq. 20).
        pos = np.flatnonzero(hits)
        rr.append(1.0 / (int(pos[0]) + 1) if len(pos) else 0.0)

        # FCP pairwise counts (Eq. 14–17), ties excluded.
        for a in range(len(y)):
            delta = (y[a] - y[a + 1:]) * (p[a] - p[a + 1:])
            con += int(np.sum(delta > 0))
            dis += int(np.sum(delta < 0))

    # 11d. Macro-average precision / recall and derived F-scores.
    precision = float(np.mean(prec)) if prec else 0.0
    recall    = float(np.mean(rec))  if rec  else 0.0

    def f_beta(beta):
        denom = beta**2 * precision + recall
        return (1 + beta**2) * precision * recall / denom if denom else 0.0

    # 11e. Assemble the metric dict.
    return {
        "MAE": calculate_mae(d["overall_rating"], pred),
        "F1":  f_beta(1),
        "F2":  f_beta(2),
        "FCP": con / (con + dis) if (con + dis) else float("nan"),
        "MAP": float(np.mean(ap)) if ap else 0.0,
        "MRR": float(np.mean(rr)) if rr else 0.0,
    }


# =============================================================================
# 12. CHART
# =============================================================================
def write_chart(table, out_path):
    """2×3 bar chart, one panel per metric."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; chart skipped.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, m in zip(axes.flat, METRICS):
        ax.bar(table.index, table[m])
        ax.set_title(f"{m} ({'lower' if m == 'MAE' else 'higher'} is better)")
        ax.tick_params(axis="x", rotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 13. MAIN PIPELINE
# =============================================================================
def main():
    # ---------------------------------------------------------------------
    # 13a. Parse CLI arguments.
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--relevant", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    if args.folds < 2 or args.epochs < 1 or args.k < 1:
        parser.error("folds>=2, epochs>=1 and k>=1 required")

    # ---------------------------------------------------------------------
    # 13b. Resolve paths and load / validate the dataset.
    # ---------------------------------------------------------------------
    root = Path(__file__).resolve().parent
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = root / dataset

    try:
        df = load_dataset(dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    # ---------------------------------------------------------------------
    # 13c. Import torch late, so the error message is clear if missing.
    # ---------------------------------------------------------------------
    import torch
    from sklearn.model_selection import GroupKFold

    # ---------------------------------------------------------------------
    # 13d. Prepare results directory + fingerprint-keyed cache.
    # ---------------------------------------------------------------------
    out = root / "results"
    out.mkdir(exist_ok=True)

    signature = {
        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "validation_fraction": args.validation_fraction,
        "k": args.k,
        "relevant": args.relevant,
        "seed": args.seed,
        "folds": args.folds,
        "skip_baselines": args.skip_baselines,
        "torch": torch.__version__,
    }
    signature_id = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:16]

    cache = out / "model_checkpoints" / signature_id
    cache.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"{len(df):,} ratings; {df.user_id.nunique():,} users; "
        f"{df.hotel_id.nunique():,} hotels; device={device}; cache={signature_id}"
    )

    # ---------------------------------------------------------------------
    # 13e. Which models are we running?
    # ---------------------------------------------------------------------
    models = (
        ["Single DNN", "Proposed Model"] if args.skip_baselines else MODELS
    )

    # ---------------------------------------------------------------------
    # 13f. Build deterministic group-based 5-fold splits.
    #      Group key = (user_id, hotel_id) so no interaction is in both
    #      train and test of the same fold.
    # ---------------------------------------------------------------------
    groups = df.user_id.astype(str) + "|" + df.hotel_id.astype(str)
    rng = np.random.default_rng(args.seed)
    unique_groups = np.asarray(groups.unique(), dtype=object)
    rng.shuffle(unique_groups)
    group_ids = pd.Series(np.arange(len(unique_groups)), index=unique_groups)

    folds = GroupKFold(n_splits=args.folds)

    all_rows = []        # one dict per (fold, model)
    criteria_rows = []   # per-criterion MAE for the Proposed Model

    # ---------------------------------------------------------------------
    # 13g. Main cross-validation loop.
    # ---------------------------------------------------------------------
    for fold, (train_ix, test_ix) in enumerate(
        folds.split(df, groups=group_ids.loc[groups].to_numpy()), start=1
    ):
        train = df.iloc[train_ix].reset_index(drop=True)
        test  = df.iloc[test_ix].reset_index(drop=True)

        for model in models:
            # 13g-i. Per-(fold, model) cache folder.
            label = model.lower().replace(" ", "_").replace("+", "plus")
            folder = cache / f"fold_{fold}" / label
            folder.mkdir(parents=True, exist_ok=True)
            resultfile = folder / "metrics.json"

            # 13g-ii. Cache hit → reuse without retraining.
            if resultfile.exists() and not args.force:
                try:
                    saved = json.loads(resultfile.read_text(encoding="utf-8"))
                    all_rows.append(saved["metrics"])
                    criteria_rows += saved.get("criteria", [])
                    print(f"Fold {fold} / {model}: cached")
                    continue
                except json.JSONDecodeError:
                    print(f"Fold {fold} / {model}: corrupt cache, retraining")

            # 13g-iii. Fresh run for this (fold, model).
            print(f"Fold {fold}/{args.folds} / {model}", flush=True)
            seed_everything(args.seed + fold)

            if model in ["Single DNN", "Proposed Model"]:
                pred, cp = train_deep(train, test, args, fold, model,
                                      folder, device)
            else:
                pred = train_surprise(train, test, model)
                cp = None

            # 13g-iv. Clip predictions into the rating range before scoring
            #         (consistent across all models; does not affect the
            #          network output layer itself, only the evaluation).
            pred = np.clip(pred, 1, 5)

            # 13g-v. Compute the six metrics.
            metrics = {
                "fold": fold,
                "model": model,
                **rank_metrics(test, pred, args.k, args.relevant),
            }

            # 13g-vi. Per-criterion MAE (only the Proposed Model produces
            #          criteria predictions).
            criterion = []
            if cp is not None:
                for i, c in enumerate(CRITERIA):
                    criterion.append({
                        "fold": fold,
                        "criterion": c,
                        "MAE": calculate_mae(test[c], np.clip(cp[:, i], 1, 5)),
                    })

            # 13g-vii. Persist the per-(fold, model) result.
            resultfile.write_text(
                json.dumps({"metrics": metrics, "criteria": criterion},
                           indent=2),
                encoding="utf-8",
            )

            all_rows.append(metrics)
            criteria_rows += criterion

            print(
                "  " + ", ".join(f"{m}={metrics[m]:.4f}" for m in METRICS),
                flush=True,
            )
            # Persist incrementally so an interrupted run doesn't lose work.
            pd.DataFrame(all_rows).to_csv(out / "fold_results.csv", index=False)

    # ---------------------------------------------------------------------
    # 13h. Aggregate per-fold results into the final Table 6.
    # ---------------------------------------------------------------------
    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(out / "fold_results.csv", index=False)

    summary = fold_df.groupby("model")[METRICS].agg(["mean", "std"])
    summary.to_csv(out / "table6_results.csv")

    # Means only, ordered to match the paper's column order.
    means = fold_df.groupby("model")[METRICS].mean().reindex(models)

    # Paper reference table — clearly separated from measured results.
    reference = pd.DataFrame(PUBLISHED, index=METRICS).T.reindex(models)
    reference.to_csv(out / "paper_reference_table6.csv")

    # Criteria MAE (Table 5).
    if criteria_rows:
        pd.DataFrame(criteria_rows).to_csv(
            out / "criteria_rating_errors.csv", index=False
        )

    # Chart + human-readable report.
    write_chart(means, out / "comparison_chart.png")
    (out / "table6_results.txt").write_text(
        "MEASURED RESULTS (not the published values)\n"
        + summary.to_string()
        + "\n\nPUBLISHED PAPER REFERENCE (not measured here)\n"
        + reference.to_string(),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # 13i. Experiment metadata.
    # ---------------------------------------------------------------------
    details = {
        "paper": "Nassar, Jafar, Rahhal (2020), Knowledge-Based Systems 187, 104811",
        "configuration": signature,
        "device": device,
        "dataset_rows": len(df),
        "users": int(df.user_id.nunique()),
        "hotels": int(df.hotel_id.nunique()),
        "ranking_protocol": (
            "held-out observed interactions only; top-k per user; relevant if "
            "overall_rating >= threshold; macro precision/recall; "
            "zero-relevance users contribute zero AP/MRR"
        ),
        "fcp_protocol": "pooled comparable pairs within users, ties excluded",
        "limitations": [
            "Paper does not specify K, relevance threshold or exact baseline "
            "hyperparameters.",
            "Some test users/items may be unseen in training. Neural models "
            "use unknown-ID embedding; Surprise falls back to algorithm "
            "estimates.",
            "Metrics may differ from paper due to data source, ranking "
            "candidate universe, seed and library versions.",
            "Validation subset is drawn from training fold; stage-2 network "
            "trains on ground-truth training criteria.",
            "Final predictions are clipped to [1,5] for evaluation; the "
            "network output layer itself is unclipped ReLU (paper Eq. 4/9).",
        ],
    }
    (out / "experiment_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------------------
    # 13j. Final console summary.
    # ---------------------------------------------------------------------
    print("\nMEASURED TABLE 6:\n", means.to_string())
    print("\nResults saved in", out)


# =============================================================================
# 14. ENTRY POINT
# =============================================================================
if __name__ == "__run__":
    run()