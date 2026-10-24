"""Research-paper reimplementation: two-stage multi-criteria recommendation.

Workflow:
1. Validate the genuine TripAdvisor-style ratings in dataset.csv.
2. Split data into grouped cross-validation folds.
3. Train baseline recommenders and neural models on each training fold.
4. Predict ratings for the held-out fold and calculate evaluation metrics.
5. Average fold scores and compare them with the paper's published Table 6.

Install: pip install numpy pandas scikit-learn torch matplotlib scikit-surprise
Run:     python main.py
         python main.py --force          # retrain instead of using cached results
         python main.py --epochs 20      # change the neural-network training budget

Expected CSV columns: user_id, hotel_id, seven criteria, overall_rating.
Missing ratings must be filtered; this script does not invent or fill ratings.
The published Table 6 numbers are reference values, not locally measured results.
"""

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


# The seven hotel criteria predicted by Stage 1 of the proposed model.
CRITERIA = ['value', 'rooms', 'location', 'cleanliness', 'check_in', 'service', 'business_service']
REQUIRED = ['user_id', 'hotel_id'] + CRITERIA + ['overall_rating']
MODELS = ['SVD', 'SVD++', 'SlopeOne', 'Baseline estimates', 'KNN with baseline', 'Single DNN', 'Proposed Model']
METRICS = ['MAE', 'F1', 'F2', 'FCP', 'MAP', 'MRR']


# Reference values reported in the paper (not results produced by this script).
PUBLISHED = {'SVD': [0.8071, 0.837, 0.7951, 0.5024, 0.3935, 0.7272], 'SVD++': [0.8222, 0.839, 0.7991, 0.5092, 0.3483, 0.7513], 'SlopeOne': [0.916, 0.8666, 0.9248, 0.5096, 0.3006, 0.7502], 'Baseline estimates': [0.8094, 0.8391, 0.7986, 0.5025, 0.3881, 0.7374], 'KNN with baseline': [0.8115, 0.8388, 0.7984, 0.5097, 0.3865, 0.736], 'Single DNN': [0.7855, 0.8572, 0.9176, 0.4957, 0.4481, 0.7244], 'Proposed Model': [0.7552, 0.8769, 0.9295, 0.5126, 0.4643, 0.7664]}


# Make experiments repeatable across Python, NumPy, and PyTorch.
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Step 1: load genuine ratings and validate the expected columns and rating range.
def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Extract/prepare genuine multi-criteria ratings first. The supplied RAR contains hotel review ratings that can be extracted into dataset.csv.\nRequired CSV header: ' + ','.join(REQUIRED))
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(' ', '_').replace('/', '_').replace('-', '_') for c in df.columns]
    aliases = {'check_in_front_desk': 'check_in', 'check_in__front_desk': 'check_in', 'business_services': 'business_service', 'overall': 'overall_rating', 'user': 'user_id', 'item_id': 'hotel_id'}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df and v not in df})
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f'Dataset is missing {missing}. Required: {REQUIRED}. No missing criteria are fabricated.')
    df = df[REQUIRED].copy()
    for c in CRITERIA + ['overall_rating']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df.isna().any().any():
        raise ValueError('Null/non-numeric IDs or ratings found; clean source data before training.')
    if not df[CRITERIA + ['overall_rating']].apply(lambda x: x.between(1, 5).all()).all():
        raise ValueError('Ratings must be within [1,5]. Filter incomplete rows (e.g., -1 means missing), never impute fake ratings.')
    df['user_id'] = df.user_id.astype(str)
    df['hotel_id'] = df.hotel_id.astype(str)
    return df.reset_index(drop=True)


# Step 2: map users and hotels to integer IDs using training data only.
def encode_fold(train, test):
    u = {v: i + 1 for i, v in enumerate(train.user_id.unique())}
    h = {v: i + 1 for i, v in enumerate(train.hotel_id.unique())}

    def enc(d):
        return np.stack([d.user_id.map(u).fillna(0).to_numpy(dtype='int64'), d.hotel_id.map(h).fillna(0).to_numpy(dtype='int64')], axis=1)
    return (enc(train), enc(test), len(u) + 1, len(h) + 1)


# Build the user/item-embedding DNN used to predict ratings.
def make_net(nu, nh, embedding, hidden, out):
    import torch
    from torch import nn

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.u = nn.Embedding(nu, embedding, padding_idx=0)
            self.i = nn.Embedding(nh, embedding, padding_idx=0)
            layers = []
            prev = embedding * 2
            for size in hidden:
                layers += [nn.Linear(prev, size), nn.ReLU()]
                prev = size
            layers += [nn.Linear(prev, out), nn.ReLU()]
            self.layers = nn.Sequential(*layers)
            self.apply(self.init_weights)

        @staticmethod
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0, std=0.05)
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.layers(torch.cat((self.u(x[:, 0]), self.i(x[:, 1])), dim=-1))
    return Net()



# Stage 2: combine the seven normalized criterion ratings into one overall rating.
def make_aggregator():
    import torch
    from torch import nn
    layers = []
    d = 7
    for h in [64, 32, 16, 8]:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 1), nn.ReLU()]
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.05)
            nn.init.zeros_(m.bias)
    return net



# Train a neural network with MAE loss; validation MAE selects the best epoch.
def train_network(net, x, y, vx, vy, args, device):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    train = TensorDataset(torch.tensor(x, dtype=torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vxt = torch.tensor(vx, dtype=torch.long if np.issubdtype(vx.dtype, np.integer) else torch.float32, device=device)
    vyt = torch.tensor(vy, dtype=torch.float32, device=device)
    best = float('inf')
    best_state = None
    wait = 0
    for epoch in range(args.epochs):
        net.train()
        loss_sum = 0
        for a, b in loader:
            a = a.to(device)
            b = b.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.abs(net(a) - b).mean()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(a)
        net.eval()
        with torch.no_grad():
            vloss = float(torch.abs(net(vxt) - vyt).mean())
        if vloss < best - 1e-05:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  epoch {epoch + 1}/{args.epochs}: train_MAE={loss_sum / len(train):.4f}, val_MAE={vloss:.4f}', flush=True)
        if wait >= args.patience:
            break
    net.load_state_dict(best_state)
    net.to(device)
    return net



# Run inference in batches to reduce memory use.
def predict(net, x, device, batch_size=4096):
    import torch
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            a = torch.tensor(x[start:start + batch_size], dtype=torch.long if np.issubdtype(x.dtype, np.integer) else torch.float32, device=device)
            parts.append(net(a).cpu().numpy())
    return np.concatenate(parts, axis=0)



# Step 3: train either the single-DNN baseline or the paper-inspired two-stage model.
def train_deep(train, test, args, fold, model_name, out, device):
    from sklearn.model_selection import train_test_split
    import torch
    idx = np.arange(len(train))
    tr, va = train_test_split(idx, test_size=max(0.01, min(0.2, args.validation_fraction)), random_state=args.seed + fold)
    trdf = train.iloc[tr]
    vadf = train.iloc[va]
    x_train, x_test, nu, nh = encode_fold(trdf, test)
    _, x_val, _, _ = encode_fold(trdf, vadf)
    # Baseline: directly predict the overall rating from user and hotel IDs.
    if model_name == 'Single DNN':
        net = make_net(nu, nh, 256, [512, 256, 128, 64, 32, 16, 8], 1)
        net = train_network(net, x_train, trdf[['overall_rating']].to_numpy('float32'), x_val, vadf[['overall_rating']].to_numpy('float32'), args, device)
        result = predict(net, x_test, device).ravel()
        torch.save(net.cpu().state_dict(), out / 'single_dnn.pt')
        return (result, None)
    criteria = make_net(nu, nh, 64, [128, 64, 32, 16, 8], 7)
    criteria = train_network(criteria, x_train, trdf[CRITERIA].to_numpy('float32'), x_val, vadf[CRITERIA].to_numpy('float32'), args, device)
    predicted_criteria = predict(criteria, x_test, device)
    torch.save(criteria.cpu().state_dict(), out / 'criteria_dnn.pt')
    mu = trdf[CRITERIA].mean().to_numpy('float32')
    std = trdf[CRITERIA].std(ddof=0).to_numpy('float32')
    std[std == 0] = 1
    norm_train = (trdf[CRITERIA].to_numpy('float32') - mu) / std
    norm_val = (vadf[CRITERIA].to_numpy('float32') - mu) / std
    aggregator = make_aggregator()
    aggregator = train_network(aggregator, norm_train, trdf[['overall_rating']].to_numpy('float32'), norm_val, vadf[['overall_rating']].to_numpy('float32'), args, device)
    result = predict(aggregator, (predicted_criteria - mu) / std, device).ravel()
    torch.save({'state_dict': aggregator.cpu().state_dict(), 'mean': mu, 'std': std}, out / 'overall_dnn.pt')
    return (result, predicted_criteria)



# Step 4: fit a classical collaborative-filtering baseline and predict held-out ratings.
def train_surprise(train, test, name):
    try:
        from surprise import Dataset, Reader, SVD, SVDpp, SlopeOne, BaselineOnly, KNNBaseline
    except ImportError as exc:
        raise RuntimeError('The five classical baselines require scikit-surprise. Install it with: pip install scikit-surprise') from exc
    mapping = {'SVD': lambda: SVD(random_state=42), 'SVD++': lambda: SVDpp(random_state=42), 'SlopeOne': SlopeOne, 'Baseline estimates': BaselineOnly, 'KNN with baseline': lambda: KNNBaseline(sim_options={'name': 'msd', 'user_based': False}, verbose=False)}
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[['user_id', 'hotel_id', 'overall_rating']], reader)
    full = data.build_full_trainset()
    algo = mapping[name]()
    algo.fit(full)
    return np.array([algo.predict(u, i).est for u, i in zip(test.user_id, test.hotel_id)], dtype=float)



# Compute mean absolute error between observed and predicted ratings.
def calculate_mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))



# Compute rating and ranking metrics on held-out observed interactions.
def rank_metrics(test, pred, k, relevant):
    """Ranking is over held-out rated pairs only, not the full unobserved hotel catalog."""
    d = test[['user_id', 'hotel_id', 'overall_rating']].copy()
    d['pred'] = pred
    ap = []
    rr = []
    prec = []
    rec = []
    con = 0
    dis = 0
    for _, g in d.groupby('user_id', sort=False):
        y = g.overall_rating.to_numpy()
        p = g.pred.to_numpy()
        order = np.argsort(-p, kind='stable')
        hits = (y[order][:k] >= relevant).astype(float)
        total = int((y >= relevant).sum())
        top = len(hits)
        if not top:
            continue
        prec.append(float(hits.sum() / top))
        rec.append(float(hits.sum() / total) if total else 0.0)
        ap.append(float((hits * np.cumsum(hits) / np.arange(1, top + 1)).sum() / min(k, total)) if total else 0.0)
        pos = np.flatnonzero(hits)
        rr.append(1 / (int(pos[0]) + 1) if len(pos) else 0.0)
        for a in range(len(y)):
            delta = (y[a] - y[a + 1:]) * (p[a] - p[a + 1:])
            con += int(np.sum(delta > 0))
            dis += int(np.sum(delta < 0))
    precision = float(np.mean(prec)) if prec else 0.0
    recall = float(np.mean(rec)) if rec else 0.0

    def f(beta):
        return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if beta ** 2 * precision + recall > 0 else 0.0
    return {'MAE': calculate_mae(d.overall_rating, pred), 'F1': f(1), 'F2': f(2), 'FCP': con / (con + dis) if con + dis else float('nan'), 'MAP': float(np.mean(ap)) if ap else 0.0, 'MRR': float(np.mean(rr)) if rr else 0.0}



# Save a visual comparison of the measured models.
def write_chart(table, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        for ax, m in zip(axes.flat, METRICS):
            ax.bar(table.index, table[m])
            ax.set_title(m + ' (lower is better)' if m == 'MAE' else m + ' (higher is better)')
            ax.tick_params(axis='x', rotation=75, labelsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
    except ImportError:
        print('matplotlib not installed; chart skipped.')



# Experiment entry point: parse settings, run cross-validation, and save results.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--validation-fraction', type=float, default=0.01)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--relevant', type=float, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip-baselines', action='store_true')
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1 or args.k < 1:
        parser.error('folds>=2, epochs>=1 and k>=1 required')
    root = Path(__file__).resolve().parent
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = root / dataset
    try:
        # Validate and load the dataset before training any model.
        df = load_dataset(dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f'DATA ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
    import torch
    from sklearn.model_selection import GroupKFold
    out = root / 'results'
    out.mkdir(exist_ok=True)
    signature = {'sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'epochs': args.epochs, 'batch_size': args.batch_size, 'patience': args.patience, 'validation_fraction': args.validation_fraction, 'k': args.k, 'relevant': args.relevant, 'seed': args.seed, 'folds': args.folds, 'skip_baselines': args.skip_baselines, 'torch': torch.__version__}
    signature_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache = out / 'model_checkpoints' / signature_id
    cache.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{len(df):,} ratings; {df.user_id.nunique():,} users; {df.hotel_id.nunique():,} hotels; device={device}; cache={signature_id}')
    
    # Optionally run only the two neural models while debugging.
    models = ['Single DNN', 'Proposed Model'] if args.skip_baselines else MODELS
    groups = df.user_id.astype(str) + '|' + df.hotel_id.astype(str)
    rng = np.random.default_rng(args.seed)
    unique_groups = np.asarray(groups.unique(), dtype=object)
    rng.shuffle(unique_groups)
    group_ids = pd.Series(np.arange(len(unique_groups)), index=unique_groups)
   
    # Use grouped folds so duplicate user-hotel review pairs stay in one fold.
    folds = GroupKFold(n_splits=args.folds)
    all_rows = []
    criteria_rows = []
    
    # Step 5: repeat training and evaluation across the requested cross-validation folds.
    for fold, (train_ix, test_ix) in enumerate(folds.split(df, groups=group_ids.loc[groups].to_numpy()), start=1):
        train = df.iloc[train_ix].reset_index(drop=True)
        test = df.iloc[test_ix].reset_index(drop=True)
        for model in models:
            label = model.lower().replace(' ', '_').replace('+', 'plus')
            folder = cache / f'fold_{fold}' / label
            folder.mkdir(parents=True, exist_ok=True)
            resultfile = folder / 'metrics.json'
            if resultfile.exists() and (not args.force):
                saved = json.loads(resultfile.read_text(encoding='utf-8'))
                all_rows.append(saved['metrics'])
                criteria_rows += saved.get('criteria', [])
                print(f'Fold {fold} / {model}: cached')
                continue
            print(f'Fold {fold}/{args.folds} / {model}', flush=True)
            seed_everything(args.seed + fold)
            
            # Train the neural model or fit the selected classical baseline.
            if model in ['Single DNN', 'Proposed Model']:
                pred, cp = train_deep(train, test, args, fold, model, folder, device)
            else:
                pred = train_surprise(train, test, model)
                cp = None
            pred = np.clip(pred, 1, 5)
            metrics = {'fold': fold, 'model': model, **rank_metrics(test, pred, args.k, args.relevant)}
            criterion = []
            if cp is not None:
                for i, c in enumerate(CRITERIA):
                    criterion.append({'fold': fold, 'criterion': c, 'MAE': calculate_mae(test[c], np.clip(cp[:, i], 1, 5))})
            resultfile.write_text(json.dumps({'metrics': metrics, 'criteria': criterion}, indent=2), encoding='utf-8')
            all_rows.append(metrics)
            criteria_rows += criterion
            print('  ' + ', '.join((f'{m}={metrics[m]:.4f}' for m in METRICS)), flush=True)
            pd.DataFrame(all_rows).to_csv(out / 'fold_results.csv', index=False)
    
    # Step 6: aggregate measured fold scores and save them separately from paper values.
    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(out / 'fold_results.csv', index=False)
    summary = fold_df.groupby('model')[METRICS].agg(['mean', 'std'])
    summary.to_csv(out / 'table6_results.csv')
    means = fold_df.groupby('model')[METRICS].mean().reindex(models)
    
    # Keep the paper's published numbers in a separate reference file.
    reference = pd.DataFrame(PUBLISHED, index=METRICS).T.reindex(models)
    reference.to_csv(out / 'paper_reference_table6.csv')
    if criteria_rows:
        pd.DataFrame(criteria_rows).to_csv(out / 'criteria_rating_errors.csv', index=False)
    write_chart(means, out / 'comparison_chart.png')
    (out / 'table6_results.txt').write_text('MEASURED RESULTS (not the published values)\n' + summary.to_string() + '\n\nPUBLISHED PAPER REFERENCE (not measured here)\n' + reference.to_string(), encoding='utf-8')
    
    # Record the configuration and known evaluation limitations for reproducibility.
    details = {'paper': 'Nassar, Jafar, Rahhal (2020), Knowledge-Based Systems 187, 104811', 'configuration': signature, 'device': device, 'dataset_rows': len(df), 'users': int(df.user_id.nunique()), 'hotels': int(df.hotel_id.nunique()), 'ranking_protocol': 'held-out observed interactions only; top-k per user; relevant if overall_rating >= threshold; macro precision/recall; zero-relevance users contribute zero AP/MRR', 'fcp_protocol': 'pooled comparable pairs within users, ties excluded', 'limitations': ['Paper does not specify K, relevance threshold or exact baseline hyperparameters.', 'Some test users/items may be unseen in training. Neural models use unknown-ID embedding; Surprise falls back to algorithm estimates.', 'Metrics may differ from paper due to data source, ranking candidate universe, seed and library versions.', 'Validation subset is drawn from training fold; stage-2 network trains on ground-truth training criteria.']}
    (out / 'experiment_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print('\nMEASURED TABLE 6:\n', means.to_string())
    print('\nResults saved in', out)


if __name__ == '__main__':
    main()