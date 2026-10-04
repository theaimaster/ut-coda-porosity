# Author: Thiraj Wegala (AMICS Lab, Auburn University)
# Copyright (c) 2026 Thiraj Wegala, AMICS Lab, Auburn University.
# Released under the MIT License (see LICENSE).
"""
run_all.py - the complete pipeline in one file: model training + every paper figure + every paper number.

    python run_all.py                  full budget (v3 settings): all runs, then figures and numbers
    python run_all.py --quick          tiny budget, only to check that everything works
    python run_all.py --analysis-only  skip training, (re)build figures and numbers from finished runs
    python run_all.py --runs run1_combined_al run4_al_steel   only these training runs
    python run_all.py --figures-only   redraw the figures from the saved numbers (seconds)

Reads the anti-aliased 1,400-sample files in ../data/experimental/csv_1400/ and the simulation file
set by SIM_FILE (../data/simulation/). Signal columns s0000, s0001, ... (any length)
and the label columns density, size, location, detection are read by name.

OUTPUT  (code/outputs/, or code/outputs_quick/ with --quick)
  results/<run>/<label>/       train/test results: confusion matrices, learning curves,
                               complete_results.json (params, predictions, accuracies, timing)
  results/results_summary_all_runs.csv / .txt
  models/<run>/<label>/        full trained models + preprocessing (scaler, label encoder, params)
  figures/                     figure3-6, supp_figure1-2 (PNG 600 dpi + PDF)
  paper_numbers/               paper_numbers.txt (every quoted value) + paper_numbers.json
  logs/                        console log of each stage

TRAINING RUNS (same models, Optuna search spaces, splits and scaling as v3)
  run1_combined_al  al_small + al_large, 205 rows (25 controls)
  run2_al_small     115 rows (25 controls)
  run3_al_large     115 rows (25 controls)
  run4_al_steel     245 rows (205 aluminium, 40 steel); results for all / aluminium / steel rows
  run5_heldout      trained like run4 on all 245 rows; the 50 held-out rows (40 Al, 10 steel)
                    are used ONLY for the final score, reported for all / aluminium / steel.
  run6_simulation   the simulation study, ../data/simulation/ (518-point signals; file chosen by SIM_FILE)
  Hyperparameters are always tuned on training data only (CV for 1-NN/KNN/SVM/RF, a
  validation split for the CNN). The held-out file is never used for tuning or model choice.
  It is read only to score models that were fitted on the 245 training rows: the final
  scoring of run5, and (analysis) the held-out window ablation and Supplementary Figure 2.

ANALYSIS (was 01_make_figures.py, 02_paper_numbers.py, 04_confusion_matrices.py)
  Figures keep the original layout and style. Data come from the training files only.
  Every classifier used in the analysis is tuned with the same Optuna budget as the runs:
    - window ablation (Figure 3f, 'coda_cv' = 'heldout'): trained exactly like run5 on the 245
      training rows and scored once on the held-out aluminium rows. Full signal: the SVM tuned in
      run5 for that task (the Table 2 model). Each window alone: an SVM re-tuned from scratch on
      that window of the training part only (same Optuna search and budget).
      The grouped / ungrouped 10x5 CV versions are still printed for reference.
    - Figure 6 d-f: out-of-fold predictions over the steel traces (5-fold, all aluminium in
      every training fold), SVM re-tuned inside every fold (nested, no leakage); the CNN
      numbers use the CNN architecture tuned in run4 for that task
    - Supplementary Figure 2: the SVM held-out predictions of run5 on the aluminium rows
      (detection, location, size, density; no retraining)
    - Supplementary Figure 1: the RF test-set predictions of run6_simulation (no retraining)
  Every accuracy shown with a confusion matrix is computed from that matrix (trace / total).

Bug fixes vs v3 (kept from the previous run_all.py)
  1. CNN early stopping really restores the best epoch (v3 used a shallow state_dict copy).
  2. CNN train accuracy / overfit gap measured in eval mode on the restored best model.
  3. Test labels unseen in training no longer corrupt the LabelEncoder.
  4. Outputs go next to this script.
"""
import pandas as pd
import numpy as np
import gc, time, os, sys, json, copy, argparse, re, warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold,
                                     StratifiedGroupKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
import optuna
import joblib
from functools import partial
from collections import Counter
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)   # capped SVM trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, '..', 'data')
DATA = os.path.join(DATA_ROOT, 'experimental', 'csv_1400')     # the anti-aliased 1,400-sample files
# simulation study (518-point signals): 'sim_aligned_518.csv' (all signals) or
# 'sim_aligned_518_cleaned.csv' (336 duplicated / conflicting rows removed)
SIM_FILE = os.path.join(DATA_ROOT, 'simulation', 'sim_aligned_518_cleaned.csv')
AL_BOTH, AL_SMALL, AL_LARGE, AL_HELDOUT, STEEL = (f'{n}_antialiased_1400.csv' for n in
                                                  ('al_both', 'al_small', 'al_large', 'al_heldout', 'steel'))

# ====================== RUNS ======================
# 'train' is one file or a list of files that are stacked in that order
RUNS = [
    {'name': 'run1_combined_al', 'train': AL_BOTH},
    {'name': 'run2_al_small',    'train': AL_SMALL},
    {'name': 'run3_al_large',    'train': AL_LARGE},
    {'name': 'run4_al_steel',    'train': [AL_BOTH, STEEL], 'split_by': 'material'},
    {'name': 'run5_heldout',     'train': [AL_BOTH, STEEL], 'split_by': 'material',
     'external_tests': {'heldout': AL_HELDOUT}},
    {'name': 'run6_simulation',  'train': SIM_FILE},
]

# ====================== CONFIGURATION (as v3) ======================
DATA_CONFIG = {
    'label_names': ['density', 'size', 'location', 'detection'],
    'test_sizes': [0.2],
    'val_size': 0.15,
    'random_state': 42,
    'normalize': True,
}
MODEL_CONFIG = {'1nn': True, 'knn': True, 'svm': True, 'rf': True, 'cnn': True, 'lstm': False}
TUNE_CONFIG = {'n_trials': 50, 'n_studies': 3, 'use_pruning': True, 'n_warmup_steps': 5}
PARALLEL_CONFIG = {'cv_n_jobs': 1, 'model_n_jobs': -1}
# Iteration cap for every SVM. Some corners of the search space (polynomial kernel with large gamma
# and C) make the solver run for hours on the 2,470-signal simulation set. Normal settings converge in
# < 50,000 iterations; a capped setting stops within seconds, scores badly and is dropped by the tuner.
SVM_MAX_ITER = 1_000_000
FINAL_EPOCHS = {'cnn': 200, 'lstm': 100}
TUNE_EPOCHS = 30
GPU_CONFIG = {'use_cuda': torch.cuda.is_available(), 'num_threads': 20}
OUT_ROOT = os.path.join(HERE, 'outputs')

# ====================== NOTICE ======================
NOTICE = 'Copyright (c) 2026 Thiraj Wegala, AMICS Lab, Auburn University'
BANNER = ('=' * 74 + '\n  Written by Thiraj Wegala, AMICS Lab, Auburn University\n  ' + NOTICE +
          '\n  Released under the MIT License (see LICENSE)\n' + '=' * 74)
_SIG = ('CiAgICBfICAgIF9fICBfXyBfX18gX19fXyBfX19fICAgIF8gICAgICAgICAgXyAgICAgCiAgIC8gXCAgfCAgXC8g'
        'IHxfIF8vIF9fXy8gX19ffCAgfCB8ICAgIF9fIF98IHxfXyAgCiAgLyBfIFwgfCB8XC98IHx8IHwgfCAgIFxfX18g'
        'XCAgfCB8ICAgLyBfYCB8ICdfIFwgCiAvIF9fXyBcfCB8ICB8IHx8IHwgfF9fXyBfX18pIHwgfCB8X198IChffCB8'
        'IHxfKSB8Ci9fLyAgIFxfXF98ICB8X3xfX19cX19fX3xfX19fLyAgfF9fX19fXF9fLF98Xy5fXy8gCgogIERpc3Nl'
        'Y3RpbmcgdWx0cmFzb25pYyBlY2hvZXMgdmlhIG1hY2hpbmUgbGVhcm5pbmcgdG8gdW5jb3ZlciBzdWItd2F2ZWxl'
        'bmd0aCBkZWZlY3RzCiAgUGlwZWxpbmUgd3JpdHRlbiBieSBUaGlyYWogV2VnYWxhLCBBTUlDUyBMYWIsIEF1YnVy'
        'biBVbml2ZXJzaXR5LgogIFRoZSBmYWludCBlY2hvZXMgY2FycnkgdGhlIGFuc3dlci4gTGlzdGVuIHRvIHRoZSBj'
        'b2RhLgo=')

# ====================== ANALYSIS CONFIGURATION ======================
ANALYSIS_CONFIG = {
    # Figure 3e: difference-signal energy split, from which porosity group
    #   'small' = 40-150 um (the sizes drawn in Figure 3), 'large' = 270-780 um
    'coda_energy_group': 'small',
    # Figure 3f: accuracies from the window ablation: 'heldout' (trained like run5, scored on
    # the held-out aluminium rows), or 'grouped' / 'ungrouped' 10x5 CV on al_both
    'coda_cv': 'heldout',
    # which tuned SVM (run1, 80/20 split) is used for the window ablation
    'ablation_params_run': 'run1_combined_al',
    # held-out window ablation: reuse the SVM hyperparameters saved by the first tuning
    # (outputs/paper_numbers/heldout_ablation_params.json), so every rerun on any machine gives
    # exactly the reported numbers. Delete that file to re-tune from scratch.
    'reuse_ablation_params': True,
    # Figure 6 steel out-of-fold SVMs: reuse the per-fold hyperparameters saved by the first tuning
    # (outputs/paper_numbers/oof_params.json). Delete that file to re-tune every fold from scratch.
    'reuse_oof_params': True,
    'oof_folds': 5,          # Figure 6 (steel out-of-fold matrices)
    'oof_seed': 0,
}
# Axis labels. The aligned data are normalised to a front-wall peak of 1 (dimensionless).
YLAB_V = "Normalized amplitude"
YLAB_DV = "Amplitude difference"

torch.set_num_threads(GPU_CONFIG['num_threads'])
DEVICE = torch.device('cuda' if GPU_CONFIG['use_cuda'] else 'cpu')


def out(*parts):
    p = os.path.join(OUT_ROOT, *parts); os.makedirs(p, exist_ok=True); return p


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, np.integer) else k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


class Tee:
    """Print to console and to a log file."""
    def __init__(self, path, mode='a'): self.f = open(path, mode, encoding='utf-8'); self.o = sys.__stdout__
    def write(self, s): self.o.write(s); self.f.write(s)
    def flush(self): self.o.flush(); self.f.flush()
    def close(self): self.f.close()


# ====================== MODELS (from v3, unchanged) ======================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes, conv_layers=3, pool_layers=2,
                 dense_layers=1, dropout_rate=0.3):
        super(CNN1D, self).__init__()
        self.init_kwargs = {'input_length': input_length, 'num_classes': num_classes,
                            'conv_layers': conv_layers, 'pool_layers': pool_layers,
                            'dense_layers': dense_layers, 'dropout_rate': dropout_rate}
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        pool_layers = min(pool_layers, conv_layers)
        channels = [1]
        for i in range(conv_layers):
            channels.append(32 * (2 ** min(i, 2)))
        for i in range(conv_layers):
            self.conv_layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=3))
        for _ in range(pool_layers):
            self.pool_layers.append(nn.MaxPool1d(2))
        feature_size = input_length
        for i in range(conv_layers):
            feature_size = feature_size - 2
            if feature_size < 1:
                raise ValueError(f"CNN architecture too deep for input length {input_length}.")
            if i < pool_layers:
                feature_size = feature_size // 2
                if feature_size < 1:
                    raise ValueError(f"CNN architecture too deep for input length {input_length}.")
        first_dense_input = channels[-1] * feature_size
        if dense_layers == 1:
            dense_sizes = [first_dense_input, num_classes]
        else:
            dense_sizes = [first_dense_input]
            for i in range(dense_layers - 1):
                dense_sizes.append(max(64, 256 // (2 ** i)))
            dense_sizes.append(num_classes)
        self.dense_layers = nn.ModuleList()
        for i in range(len(dense_sizes) - 1):
            self.dense_layers.append(nn.Linear(dense_sizes[i], dense_sizes[i + 1]))
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        pool_counter = 0
        for conv in self.conv_layers:
            x = self.activation(conv(x))
            if pool_counter < len(self.pool_layers):
                x = self.pool_layers[pool_counter](x)
                pool_counter += 1
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        for dense in self.dense_layers[:-1]:
            x = self.activation(dense(x))
            x = self.dropout(x)
        return self.dense_layers[-1](x)


class LSTM_RNN(nn.Module):
    def __init__(self, input_length, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(LSTM_RNN, self).__init__()
        self.init_kwargs = {'input_length': input_length, 'hidden_size': hidden_size, 'num_layers': num_layers,
                            'num_classes': num_classes, 'dropout_rate': dropout_rate}
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out_, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out_[:, -1, :]))


# ====================== DATA ======================
def read_table(files):
    """One CSV, or several stacked in order. Relative names are looked up in DATA."""
    files = files if isinstance(files, (list, tuple)) else [files]
    return pd.concat([pd.read_csv(os.path.join(DATA, f)) for f in files], ignore_index=True)


def read_xy(path, label):
    """Signal columns (s0000, s0001, ...; any length) and one label column, both by name."""
    df = read_table(path)
    X = df[[c for c in df.columns if re.fullmatch(r's\d+', str(c))]].values.astype(float)
    y = df[label].values
    return X, y


def read_groups(path, run):
    """{'all': mask, <value>: mask, ...} from the run's 'split_by' column (e.g. material)."""
    df = read_table(path)
    groups = {'all': np.ones(len(df), bool)}
    if 'split_by' in run:
        col = df[run['split_by']].values
        for v in sorted(pd.unique(col)):
            groups[str(v)] = col == v
    return groups


def prepare(run, label, test_size):
    """Train/val/test exactly as v3. For external runs the whole training file is
    used for training + validation and the test sets come from other files.
    Every test set is scored for all rows and, if the run has 'split_by', for each group."""
    X, y = read_xy(run['train'], label)
    groups = read_groups(run['train'], run)
    idx = np.arange(len(y))
    if 'external_tests' in run:
        tr_full_idx, te_idx = idx, None
    else:
        tr_full_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=DATA_CONFIG['random_state'],
                                               shuffle=True, stratify=y)
    tr_idx, va_idx = train_test_split(tr_full_idx, test_size=DATA_CONFIG['val_size'],
                                      random_state=DATA_CONFIG['random_state'], shuffle=True, stratify=y[tr_full_idx])
    X_train, X_val, y_train, y_val = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

    scaler = None
    if DATA_CONFIG['normalize']:
        scaler = StandardScaler().fit(X_train)                 # fit on training part only
        X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
    le = LabelEncoder().fit(y[tr_full_idx])
    train_group_masks = {g: np.concatenate([m[tr_idx], m[va_idx]]) for g, m in groups.items()}

    tests = {}
    if 'external_tests' in run:
        for tname, f in run['external_tests'].items():
            p = f
            if not os.path.exists(os.path.join(DATA, p)):
                print(f'[warning] external test file {f} not found - skipped'); continue
            Xt, yt = read_xy(p, label)
            if scaler is not None: Xt = scaler.transform(Xt)
            for g, m in read_groups(p, run).items():
                if m.any(): tests[f'{tname}_{g}'] = {'X': Xt, 'y': yt, 'mask': m, 'group': g}
    else:
        Xt = X[te_idx]
        if scaler is not None: Xt = scaler.transform(Xt)
        for g, m in groups.items():
            if m[te_idx].any(): tests[f'test_{g}'] = {'X': Xt, 'y': y[te_idx], 'mask': m[te_idx], 'group': g}

    return dict(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                y_train_enc=le.transform(y_train), y_val_enc=le.transform(y_val),
                train_group_masks=train_group_masks, tests=tests, scaler=scaler, le=le)


# ====================== TRAINING ======================
def nn_accuracy(model, X, y_enc, batch_size=64):
    model.eval(); correct = 0
    with torch.no_grad():
        for bx, by in DataLoader(TimeSeriesDataset(X, y_enc), batch_size=batch_size):
            correct += (model(bx.to(DEVICE)).argmax(1).cpu() == by).sum().item()
    return 100 * correct / len(y_enc)


def train_neural_network(model, X_train, y_train_enc, X_val, y_val_enc, config, epochs, name, patience=15):
    np.random.seed(DATA_CONFIG['random_state']); torch.manual_seed(DATA_CONFIG['random_state'])
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train_enc), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val_enc), batch_size=config['batch_size'])
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    tr_acc, va_acc, tr_loss, va_loss = [], [], [], []
    best_acc, best_state, best_epoch, wait = 0, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train(); loss_sum = correct = total = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(); o = model(bx); loss = criterion(o, by); loss.backward(); optimizer.step()
            loss_sum += loss.item(); correct += (o.argmax(1) == by).sum().item(); total += by.size(0)
        tr_acc.append(100 * correct / total); tr_loss.append(loss_sum / len(train_loader))
        model.eval(); vloss = vcorrect = vtotal = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE); o = model(bx)
                vloss += criterion(o, by).item(); vcorrect += (o.argmax(1) == by).sum().item(); vtotal += by.size(0)
        va_acc.append(100 * vcorrect / vtotal); va_loss.append(vloss / len(val_loader))
        if va_acc[-1] > best_acc:
            best_acc, best_epoch, wait = va_acc[-1], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}   # fix #1
        else:
            wait += 1
        if epoch % 10 == 0:
            print(f'{name} Epoch {epoch}: Train Loss = {tr_loss[-1]:.4f}, Train Acc (dropout on) = {tr_acc[-1]:.2f}%, Val Acc = {va_acc[-1]:.2f}%')
        if wait >= patience:
            print(f'{name} Early stopping at epoch {epoch}'); break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.training_time = time.perf_counter() - t0
    model.final_train_accuracy = nn_accuracy(model, X_train, y_train_enc)                   # fix #2
    model.best_val_accuracy, model.best_epoch = best_acc, best_epoch
    model.overfit_gap = model.final_train_accuracy - best_acc
    model.train_accuracies, model.val_accuracies, model.train_losses, model.val_losses = tr_acc, va_acc, tr_loss, va_loss
    print(f'{name} Best epoch {best_epoch} | train acc (eval mode) {model.final_train_accuracy:.2f}% | '
          f'best val {best_acc:.2f}% | gap {model.overfit_gap:.2f}% | {model.training_time:.1f}s')
    return model


# ====================== OPTUNA OBJECTIVES (v3 search spaces) ======================
def _knn_params(trial, k):
    p = {'n_neighbors': k, 'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
         'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])}
    if p['metric'] == 'minkowski': p['p'] = trial.suggest_int('p', 1, 4)
    return p

def objective_1nn(trial, base_data):
    X, y, cv = (*base_data, 3)[:3]      # cv: 3 folds (default) or a list of (train, test) folds
    m = KNeighborsClassifier(**_knn_params(trial, 1), n_jobs=PARALLEL_CONFIG['model_n_jobs'])
    return cross_val_score(m, X, y, cv=cv, n_jobs=PARALLEL_CONFIG['cv_n_jobs']).mean()

def objective_knn(trial, base_data):
    X, y, cv = (*base_data, 3)[:3]      # cv: 3 folds (default) or a list of (train, test) folds
    m = KNeighborsClassifier(**_knn_params(trial, trial.suggest_int('n_neighbors', 3, 15)), n_jobs=PARALLEL_CONFIG['model_n_jobs'])
    return cross_val_score(m, X, y, cv=cv, n_jobs=PARALLEL_CONFIG['cv_n_jobs']).mean()

def objective_svm(trial, base_data):
    X, y, cv = (*base_data, 3)[:3]      # cv: 3 folds (default) or a list of (train, test) folds
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    p = {'C': trial.suggest_float('C', 1, 1e5, log=True), 'kernel': kernel,
         'gamma': trial.suggest_float('gamma', 1e-5, 1, log=True)}
    if kernel == 'poly': p['degree'] = trial.suggest_int('degree', 1, 5)
    return cross_val_score(SVC(**p, random_state=DATA_CONFIG['random_state'], max_iter=SVM_MAX_ITER), X, y, cv=cv,
                           n_jobs=PARALLEL_CONFIG['cv_n_jobs']).mean()

def objective_rf(trial, base_data):
    X, y, cv = (*base_data, 3)[:3]      # cv: 3 folds (default) or a list of (train, test) folds
    p = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 10, 100),
         'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])}
    m = RandomForestClassifier(**p, random_state=DATA_CONFIG['random_state'], n_jobs=PARALLEL_CONFIG['model_n_jobs'])
    return cross_val_score(m, X, y, cv=cv, n_jobs=PARALLEL_CONFIG['cv_n_jobs']).mean()

def _nn_trial(trial, data_tuple, build, params):
    X_train, y_train_enc, X_val, y_val_enc = data_tuple
    np.random.seed(DATA_CONFIG['random_state'] + trial.number); torch.manual_seed(DATA_CONFIG['random_state'] + trial.number)
    tl = DataLoader(TimeSeriesDataset(X_train, y_train_enc), batch_size=params['batch_size'], shuffle=True)
    vl = DataLoader(TimeSeriesDataset(X_val, y_val_enc), batch_size=params['batch_size'])
    try:
        model = build().to(DEVICE)
        crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=params['learning_rate'])
        acc = 0.0
        for epoch in range(TUNE_EPOCHS):
            model.train()
            for bx, by in tl:
                bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad(); crit(model(bx), by).backward(); opt.step()
            model.eval(); c = t_ = 0
            with torch.no_grad():
                for bx, by in vl:
                    c += (model(bx.to(DEVICE)).argmax(1).cpu() == by).sum().item(); t_ += by.size(0)
            acc = c / t_; trial.report(acc, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
        return acc
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f'Error in trial: {e}'); return 0.0
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def objective_cnn(trial, data_tuple):
    p = {'conv_layers': trial.suggest_int('conv_layers', 1, 4), 'pool_layers': trial.suggest_int('pool_layers', 1, 3),
         'dense_layers': trial.suggest_int('dense_layers', 1, 3), 'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
         'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)}
    p['pool_layers'] = min(p['pool_layers'], p['conv_layers'])
    X_train, y_train_enc = data_tuple[0], data_tuple[1]
    build = lambda: CNN1D(X_train.shape[1], len(np.unique(y_train_enc)), p['conv_layers'], p['pool_layers'],
                          p['dense_layers'], p['dropout_rate'])
    return _nn_trial(trial, data_tuple, build, p)

def objective_lstm(trial, data_tuple):
    p = {'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]), 'num_layers': trial.suggest_int('num_layers', 1, 3),
         'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5), 'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)}
    X_train, y_train_enc = data_tuple[0], data_tuple[1]
    build = lambda: LSTM_RNN(X_train.shape[1], p['hidden_size'], p['num_layers'], len(np.unique(y_train_enc)), p['dropout_rate'])
    return _nn_trial(trial, data_tuple, build, p)


def robust_optimization(objective_fn, data, model_name, verbose=True):
    n_trials, n_studies = TUNE_CONFIG['n_trials'], TUNE_CONFIG['n_studies']
    if model_name == '1nn':
        n_trials, n_studies = min(n_trials, 20), min(n_studies, 2)
    is_nn = model_name in ('cnn', 'lstm')
    studies = []
    for i in range(n_studies):
        if verbose: print(f'  Study {i + 1}/{n_studies}...', end=' ', flush=True)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=TUNE_CONFIG['n_warmup_steps']) \
            if (TUNE_CONFIG['use_pruning'] and is_nn) else optuna.pruners.NopPruner()
        study = optuna.create_study(direction='maximize', pruner=pruner,
                                    sampler=optuna.samplers.TPESampler(seed=DATA_CONFIG['random_state'] + i))
        study.optimize(partial(objective_fn, data_tuple=data) if is_nn else partial(objective_fn, base_data=data),
                       n_trials=n_trials, show_progress_bar=False)
        studies.append(study)
        if verbose: print(f'Best: {study.best_value:.4f}')
    vals = [s.best_value for s in studies]
    if verbose: print(f'  Stats: Best={max(vals):.4f}, Mean={np.mean(vals):.4f}, Std={np.std(vals):.4f}')
    return max(studies, key=lambda s: s.best_value)


# ====================== EVALUATION ======================
def predict_labels(name, model, X, le):
    """Original-label predictions and inference time."""
    t0 = time.perf_counter()
    if name in ('CNN', 'LSTM'):
        model.eval()
        with torch.no_grad():
            enc = model(torch.FloatTensor(X).unsqueeze(1).to(DEVICE)).argmax(1).cpu().numpy()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        pred = le.inverse_transform(enc)
    else:
        pred = model.predict(X)
    return pred, time.perf_counter() - t0


def evaluate(models, test, le, label_name, split_tag, test_name, out_dir):
    """Accuracy + confusion matrix on test rows selected by test['mask']."""
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    m = test['mask']; y_true = test['y'][m]
    for name, model in models.items():
        pred, t_inf = predict_labels(name, model, test['X'], le)
        y_pred = pred[m]
        acc = 100 * accuracy_score(y_true, y_pred)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        title = f'{name} - {label_name} | {split_tag} | {test_name}\nAccuracy: {acc:.2f}% (n={len(y_true)})'
        print(f'\n{title}')
        print(pd.DataFrame(cm, index=[f'true {l}' for l in labels], columns=[f'pred {l}' for l in labels]).to_string())
        plt.figure(figsize=(8, 6.5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cm_{name}_{label_name}_{split_tag}_{test_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        res[name] = {'accuracy': acc, 'n_test': int(len(y_true)), 'labels': labels, 'confusion_matrix': cm.tolist(),
                     'y_true': y_true.tolist(), 'y_pred': list(y_pred),
                     'inference_time_total_s': t_inf, 'inference_time_per_sample_ms': 1000 * t_inf / len(test['y'])}
    return res


def plot_learning_curves(model, name, label_name, split_tag, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5)); ep = range(1, len(model.train_accuracies) + 1)
    ax[0].plot(ep, model.train_accuracies, 'b-', label='Train Acc (dropout on)'); ax[0].plot(ep, model.val_accuracies, 'r-', label='Val Acc')
    ax[1].plot(ep, model.train_losses, 'b-', label='Train Loss'); ax[1].plot(ep, model.val_losses, 'r-', label='Val Loss')
    for a in ax:
        a.axvline(model.best_epoch + 1, color='g', ls='--', label=f'Best epoch ({model.best_epoch})'); a.legend(); a.grid(alpha=.3)
    ax[0].set_title(f'{name} - Accuracy | gap {model.overfit_gap:.2f}%'); ax[1].set_title(f'{name} - Loss')
    plt.suptitle(f'{label_name} - {split_tag}'); plt.tight_layout()
    d = os.path.join(out_dir, 'learning_curves'); os.makedirs(d, exist_ok=True)
    plt.savefig(os.path.join(d, f'learning_curve_{name}_{label_name}_{split_tag}.png'), dpi=300, bbox_inches='tight'); plt.close()


def save_models(models, model_dir, results_dir, label_name, split_tag, scaler, le, input_length, best_params):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({'scaler': scaler, 'label_encoder': le, 'input_length': input_length, 'best_params': best_params,
                 'label_name': label_name, 'split': split_tag}, os.path.join(model_dir, f'preprocessing_{label_name}_{split_tag}.joblib'))
    for name, model in models.items():
        base = os.path.join(model_dir, f'{name}_{label_name}_{split_tag}')
        if name in ('CNN', 'LSTM'):
            model_cpu = copy.deepcopy(model).to('cpu')
            torch.save(model_cpu, base + '_full.pt')          # full model; load with run_all.py importable
            torch.save({'model_state_dict': model_cpu.state_dict(), 'model_class': type(model).__name__,
                        'init_kwargs': model.init_kwargs, 'classes_': le.classes_.tolist(), 'input_length': input_length,
                        'best_val_accuracy': model.best_val_accuracy, 'best_epoch': model.best_epoch,
                        'final_train_accuracy': model.final_train_accuracy, 'overfit_gap': model.overfit_gap,
                        'train_accuracies': model.train_accuracies, 'val_accuracies': model.val_accuracies,
                        'train_losses': model.train_losses, 'val_losses': model.val_losses}, base + '.pt')
            plot_learning_curves(model, name, label_name, split_tag, results_dir)
        else:
            joblib.dump({'model': model, 'scaler': scaler, 'label_encoder': le, 'input_length': input_length,
                         'label_name': label_name, 'split': split_tag}, base + '.joblib')
    print(f'  Models saved to: {model_dir}')


# ====================== ONE (run, label, split) ======================
def run_one(run, label_name, test_size, results_dir, model_dir):
    external = 'external_tests' in run
    split_tag = 'trainall' if external else f'split{int(round((1 - test_size) * 100))}'
    D = prepare(run, label_name, test_size)
    print(f'\nData: train {D["X_train"].shape}, val {D["X_val"].shape}, ' +
          ', '.join(f'{k} {v["X"].shape} (reported rows: {int(v["mask"].sum())})' for k, v in D['tests'].items()))
    print(f'Classes: {list(D["le"].classes_)} | train distribution: {dict(Counter(D["y_train"]))}')

    # --- tuning: training + validation parts only
    trad = (D['X_train'], D['y_train']); nnd = (D['X_train'], D['y_train_enc'], D['X_val'], D['y_val_enc'])
    objectives = {'1nn': (objective_1nn, trad), 'knn': (objective_knn, trad), 'rf': (objective_rf, trad),
                  'svm': (objective_svm, trad), 'cnn': (objective_cnn, nnd), 'lstm': (objective_lstm, nnd)}
    best_params = {}
    for k, (fn, data) in objectives.items():
        if MODEL_CONFIG[k]:
            print(f'\nOptimizing {k.upper()}...'); best_params[k] = robust_optimization(fn, data, k).best_params

    # --- final training on train + val (NN: 10% of that for early stopping)
    X_full = np.vstack([D['X_train'], D['X_val']]); y_full = np.concatenate([D['y_train'], D['y_val']])
    y_full_enc = np.concatenate([D['y_train_enc'], D['y_val_enc']])
    X_tr_f, X_va_f, y_tr_f, y_va_f = train_test_split(X_full, y_full_enc, test_size=0.1,
                                                      random_state=DATA_CONFIG['random_state'], stratify=y_full_enc)
    n_cls = len(D['le'].classes_); models = {}
    if MODEL_CONFIG['cnn']:
        c = best_params['cnn'].copy(); c['pool_layers'] = min(c['pool_layers'], c['conv_layers'])
        print('\nTraining CNN...')
        models['CNN'] = train_neural_network(CNN1D(X_full.shape[1], n_cls, c['conv_layers'], c['pool_layers'], c['dense_layers'],
                                                   c['dropout_rate']), X_tr_f, y_tr_f, X_va_f, y_va_f, c, FINAL_EPOCHS['cnn'], 'CNN')
    if MODEL_CONFIG['lstm']:
        c = best_params['lstm'].copy(); print('\nTraining LSTM...')
        models['LSTM'] = train_neural_network(LSTM_RNN(X_full.shape[1], c['hidden_size'], c['num_layers'], n_cls, c['dropout_rate']),
                                              X_tr_f, y_tr_f, X_va_f, y_va_f, c, FINAL_EPOCHS['lstm'], 'LSTM')
    sk = {'1nn': ('1-NN', lambda p: KNeighborsClassifier(**p, n_jobs=PARALLEL_CONFIG['model_n_jobs'])),
          'knn': ('KNN', lambda p: KNeighborsClassifier(**p, n_jobs=PARALLEL_CONFIG['model_n_jobs'])),
          'svm': ('SVM', lambda p: SVC(**p, random_state=DATA_CONFIG['random_state'], max_iter=SVM_MAX_ITER)),
          'rf':  ('RF', lambda p: RandomForestClassifier(**p, random_state=DATA_CONFIG['random_state'], n_jobs=PARALLEL_CONFIG['model_n_jobs']))}
    for k, (name, make) in sk.items():
        if MODEL_CONFIG[k]:
            p = best_params[k].copy()
            if k == '1nn': p['n_neighbors'] = 1
            t0 = time.perf_counter(); models[name] = make(p).fit(X_full, y_full); models[name].training_time = time.perf_counter() - t0
            print(f'Trained {name} in {models[name].training_time:.2f}s')

    # --- train accuracy per group (all / aluminium / steel) for the overfitting table
    train_acc = {}
    for name, model in models.items():
        pred, _ = predict_labels(name, model, X_full, D['le'])
        train_acc[name] = {g: 100 * accuracy_score(y_full[m], pred[m])
                           for g, m in D['train_group_masks'].items() if m.any()}

    # --- final scoring: the ONLY place test / held-out data is used
    print(f'\n{"-" * 80}\nEVALUATION\n{"-" * 80}')
    evals = {t: evaluate(models, test, D['le'], label_name, split_tag, t, results_dir) for t, test in D['tests'].items()}
    save_models(models, model_dir, results_dir, label_name, split_tag, D['scaler'], D['le'], X_full.shape[1], best_params)

    over = {}
    for name, model in models.items():
        o = {f'train_accuracy_{g}': a for g, a in train_acc[name].items()}
        if name in ('CNN', 'LSTM'):
            o.update(val_accuracy=model.best_val_accuracy, best_epoch=model.best_epoch)
        for t, r in evals.items():
            g = D['tests'][t]['group']
            o[f'test_accuracy_{t}'] = r[name]['accuracy']
            o[f'gap_train_test_{t}'] = train_acc[name].get(g, np.nan) - r[name]['accuracy']
        o['training_time_s'] = model.training_time
        over[name] = o
    print(f'\nSummary {label_name} {split_tag}:')
    print(pd.DataFrame(over).T.round(2).to_string())
    return {'best_params': best_params, 'eval_results': evals, 'overfitting_timing': over,
            'data_info': {'train_size': len(X_full), 'val_size': len(D['X_val']),
                          'tests': {k: int(v['mask'].sum()) for k, v in D['tests'].items()},
                          'test_groups': {k: v['group'] for k, v in D['tests'].items()},
                          'train_group_sizes': {g: int(m.sum()) for g, m in D['train_group_masks'].items()},
                          'class_distribution_train': dict(Counter(y_full))}}


def run_training(selected=None):
    for run in RUNS:
        if selected and run['name'] not in selected: continue
        tee = Tee(os.path.join(out('logs'), f'{run["name"]}.log')); sys.stdout = tee
        try:
            sizes = [None] if 'external_tests' in run else DATA_CONFIG['test_sizes']
            for label in DATA_CONFIG['label_names']:
                results_dir = out('results', run['name'], label); model_dir = out('models', run['name'], label)
                done = os.path.join(results_dir, 'complete_results.json')
                if os.path.exists(done):
                    print(f'\n[skip] {run["name"]} / {label} already complete'); continue
                results = {}
                for ts in sizes:
                    tag = 'held-out test' if ts is None else f'{int(round((1 - ts) * 100))}/{int(round(ts * 100))} split'
                    print(f'\n{"=" * 80}\n{run["name"]} | {label} | {tag}\n{"=" * 80}')
                    results[str(ts) if ts is not None else 'trainall'] = run_one(run, label, ts, results_dir, model_dir)
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                with open(done, 'w') as f:
                    json.dump(make_json_serializable(results), f, indent=2)
        finally:
            sys.stdout = sys.__stdout__; tee.close()
    write_summary()


def write_summary():
    rows = []
    for run in RUNS:
        for label in DATA_CONFIG['label_names']:
            p = os.path.join(OUT_ROOT, 'results', run['name'], label, 'complete_results.json')
            if not os.path.exists(p): continue
            for split, r in json.load(open(p)).items():
                for test, ev in r['eval_results'].items():
                    g = r['data_info']['test_groups'][test]
                    for model, m in ev.items():
                        rows.append({'run': run['name'], 'label': label,
                                     'split': {'0.2': '80/20', '0.3': '70/30', 'trainall': 'train on all 245'}.get(split, split), 'test_set': test,
                                     'rows': g, 'model': model,
                                     'accuracy': round(m['accuracy'], 2), 'n_test': m['n_test'],
                                     'train_accuracy': round(r['overfitting_timing'][model].get(f'train_accuracy_{g}', np.nan), 2)})
    if not rows: return
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out('results'), 'results_summary_all_runs.csv'), index=False)
    with open(os.path.join(out('results'), 'results_summary_all_runs.txt'), 'w') as f:
        for (run, split, test), g in df.groupby(['run', 'split', 'test_set'], sort=False):
            t = g.pivot(index='model', columns='label', values='accuracy')[[l for l in DATA_CONFIG['label_names'] if l in set(g.label)]]
            t['Mean'] = t.mean(axis=1).round(2)
            nmin, nmax = g.n_test.min(), g.n_test.max()
            block = f'\n{run} | split {split} | {test} | {g.rows.iloc[0]} rows (n={nmin if nmin == nmax else f"{nmin}-{nmax}"})\n{t.to_string()}\n'
            print(block); f.write(block)
    print(f'\nSummary written to {os.path.join(OUT_ROOT, "results", "results_summary_all_runs.csv")}')


# =====================================================================================
#                ANALYSIS: paper figures and paper numbers (training files only)
# =====================================================================================
T, N = 52.0, 1400
t = np.linspace(0, T, N)
DT = T / N
C = {40: "#0072BD", 75: "#D95319", 150: "#12836B",
     270: "#EDB120", 335: "#6BB7E8", 780: "#D95FA0"}
C_AL, C_ST = "#0072BD", "#D95319"
C_PRIM, C_CODA, C_FULL = "#C0504D", "#1F4E79", "#7F7F7F"


def setup_style():
    """Paper style. Liberation Serif if present, otherwise the metric-identical Times New Roman."""
    fam = "Liberation Serif"
    p = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
    elif fam not in {f.name for f in font_manager.fontManager.ttflist}:
        fam = "Times New Roman" if "Times New Roman" in {f.name for f in font_manager.fontManager.ttflist} else "serif"
    plt.rcParams.update({"font.family": fam, "font.size": 10,
                         "axes.grid": True, "grid.color": "#c8c8c8",
                         "grid.linewidth": 0.5, "axes.axisbelow": True,
                         "axes.edgecolor": "k", "axes.linewidth": 0.8})


def load_dataset(path):
    """X (n, samples), Y (n, 4) as density, size, location, detection, and the material column."""
    df = read_table(path)
    sig = [c for c in df.columns if re.fullmatch(r's\d+', str(c))]
    Y = df[['density', 'size', 'location', 'detection']].to_numpy(int)
    mat = df['material'].to_numpy(str) if 'material' in df.columns else np.array(['aluminium'] * len(df))
    return df[sig].to_numpy(float), Y, mat


def cx(v):
    """Complexity. RMS of the second derivative, un-normalized."""
    d2 = (v[2:] - 2 * v[1:-1] + v[:-2]) / DT ** 2
    return np.sqrt((d2 ** 2).mean())


def center_ylim(ax, m, s):
    """Vertically centre the data (means +/- sd) in the panel, with equal margins above and below."""
    lo, hi = min(np.subtract(m, s)), max(np.add(m, s)); pad = 0.5 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

def fin(ax, xl, yl, ti, ts=10):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ti, fontsize=ts)


def lower_until_clear(ax, leg, curves, pad=0.04):
    """Extend the y-axis downward (as in panels 3d and 5a) until the legend box sits below
    every curve inside its time span."""
    fig = ax.figure
    for _ in range(60):
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transData.inverted())
        m = (t >= bb.x0) & (t <= bb.x1)
        lo_curve = min(c[m].min() for c in curves)
        lo, hi = ax.get_ylim()
        if bb.y1 + pad * (hi - lo) <= lo_curve:
            return
        ax.set_ylim(lo - 0.05 * (hi - lo), hi)


def save(fig, stem):
    d = out('figures')
    fig.savefig(os.path.join(d, stem + ".png"), dpi=600, facecolor="w",
                metadata={'Author': 'Thiraj Wegala', 'Copyright': NOTICE})
    fig.savefig(os.path.join(d, stem + ".pdf"), facecolor="w",
                metadata={'Author': 'Thiraj Wegala', 'Subject': NOTICE})
    plt.close(fig)


_F = np.array([0.000,0.007,0.013,0.086,0.143,0.155,0.164,0.198,0.205,0.216,
               0.233,0.272,0.333,0.353,0.440,0.482,0.664,0.682,0.714,0.833,
               0.857,0.914,0.980,1.000])
_RGB = np.array([[248,248,251],[238,241,247],[231,236,243],[196,210,225],
                 [177,194,216],[173,192,214],[171,189,213],[161,182,209],
                 [160,181,208],[156,178,206],[152,175,204],[143,168,200],
                 [124,154,190],[117,149,187],[ 91,131,174],[ 79,122,167],
                 [ 42, 91,143],[ 40, 89,141],[ 36, 85,138],[ 24, 70,124],
                 [ 21, 67,121],[ 16, 60,115],[  9, 52,108],[  7, 51,107]], float)
CM_BLUE = ListedColormap(np.column_stack(
    [np.interp(np.linspace(0, 1, 256), _F, _RGB[:, k]) for k in range(3)]) / 255)


def confusion(ax, M, title, ticks=None):
    """ticks: the class codes of the rows/columns (default 1..n)."""
    n = M.shape[0]; vmax = M.max()
    ticks = list(range(1, n + 1)) if ticks is None else list(ticks)
    ax.imshow(M, cmap=CM_BLUE, vmin=0, vmax=vmax)
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                continue
            r, g, b = np.array(CM_BLUE(M[i, j] / vmax)[:3]) * 255
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    color="white" if lum < 155 else "black", fontsize=10)
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks)
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="major", visible=False)
    ax.grid(which="minor", color="w", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_linewidth(1.8); s.set_color("k")
    ax.tick_params(axis="y", pad=1.5)
    ax.set_xlabel("Predicted", labelpad=2)
    ax.set_ylabel("True", labelpad=0)
    ax.set_title(title, fontsize=10)


class Report:
    """Collects every quoted number: printed, written to paper_numbers.txt and .json."""
    def __init__(self): self.lines, self.values = [], {}
    def p(self, s=''): print(s); self.lines.append(s)
    def v(self, key, val): self.values[key] = make_json_serializable(val)
    def save(self):
        d = out('paper_numbers')
        with open(os.path.join(d, 'paper_numbers.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines) + '\n')
        with open(os.path.join(d, 'paper_numbers.json'), 'w') as f:
            json.dump(self.values, f, indent=1)


def tuned_params(run, label, model):
    """Best hyperparameters found for (run, label) during training (first split)."""
    p = os.path.join(OUT_ROOT, 'results', run, label, 'complete_results.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} missing: run the training for {run} first')
    r = json.load(open(p))
    return r[sorted(r)[0]]['best_params'][model]


def tune_svm(Xs, y):
    """SVM tuned with the same Optuna search space and budget as the runs (3-fold CV on Xs only)."""
    return robust_optimization(objective_svm, (Xs, y), 'svm', verbose=False).best_params


def tune_rf(Xs, y):
    """Random forest tuned with the same Optuna search space and budget as the runs."""
    return robust_optimization(objective_rf, (Xs, y), 'rf', verbose=False).best_params


def oof(X_fixed, y_fixed, X_cv, y_cv, cnn_params=None, tag='', model='SVM'):
    """Out-of-fold predictions for every row of X_cv. X_fixed/y_fixed (may be empty) are in every
    training fold. The scaler and the classifier (SVM or RF) are re-fitted and re-tuned in every fold."""
    k = ANALYSIS_CONFIG['oof_folds']
    pfile = os.path.join(out('paper_numbers'), 'oof_params.json')
    saved = json.load(open(pfile)) if (ANALYSIS_CONFIG['reuse_oof_params'] and os.path.exists(pfile)) else {}
    new_params = {}
    preds = {model: np.zeros(len(y_cv), int)}
    if cnn_params is not None: preds['CNN'] = np.zeros(len(y_cv), int)
    for f, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=ANALYSIS_CONFIG['oof_seed']).split(X_cv, y_cv)):
        Xtr = np.vstack([X_fixed, X_cv[tr]]); ytr = np.concatenate([y_fixed, y_cv[tr]])
        sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(X_cv[te])
        if model == 'SVM':
            p = saved.get(tag, {}).get(str(f + 1)) or tune_svm(Xtr_s, ytr)
            new_params[str(f + 1)] = p
            clf = SVC(**p, random_state=DATA_CONFIG['random_state'], max_iter=SVM_MAX_ITER)
        else:
            p = tune_rf(Xtr_s, ytr)
            clf = RandomForestClassifier(**p, random_state=DATA_CONFIG['random_state'], n_jobs=PARALLEL_CONFIG['model_n_jobs'])
        preds[model][te] = clf.fit(Xtr_s, ytr).predict(Xte_s)
        src = 'saved' if (model == 'SVM' and str(f + 1) in saved.get(tag, {})) else 'tuned'
        print(f'  {tag} fold {f + 1}/{k}: {src} {model} {p}')
        if cnn_params is not None:
            c = dict(cnn_params); c['pool_layers'] = min(c['pool_layers'], c['conv_layers'])
            le = LabelEncoder().fit(ytr); ye = le.transform(ytr)
            a, b_, ya, yb = train_test_split(Xtr_s, ye, test_size=0.1, random_state=DATA_CONFIG['random_state'], stratify=ye)
            net = train_neural_network(CNN1D(Xtr_s.shape[1], len(le.classes_), c['conv_layers'], c['pool_layers'],
                                             c['dense_layers'], c['dropout_rate']), a, ya, b_, yb, c,
                                       FINAL_EPOCHS['cnn'], f'CNN {tag} fold {f + 1}')
            preds['CNN'][te] = predict_labels('CNN', net, Xte_s, le)[0]
    if model == 'SVM' and tag not in saved:
        allp = json.load(open(pfile)) if os.path.exists(pfile) else {}
        allp[tag] = new_params
        with open(pfile, 'w') as fp_:
            json.dump(allp, fp_, indent=1)
    return preds


# ------------------------------------------------------------------ paper numbers (was 02)
def paper_numbers(R):
    X, _Y, _ = load_dataset(AL_BOTH)
    n = len(X)
    den, siz, loc, det = _Y[:, 0], _Y[:, 1], _Y[:, 2], _Y[:, 3]
    ctrl = det == 1
    small = (siz <= 4)                 # al_small block: 40-150 um porosity + the controls
    large = (siz >= 5) | ctrl          # al_large block: 270-780 um porosity + controls
    y = {'detection': det, 'location': loc, 'size': siz, 'density': den}

    # configurations: (size, density, location) triples, controls in blocks of 5
    grp = np.zeros(n, int); k = 0; seen = {}; ci = 0
    for i in range(n):
        if det[i] == 1:
            key = ('c', ci // 5); ci += 1
        else:
            key = (siz[i], den[i], loc[i])
        if key not in seen:
            seen[key] = k; k += 1
        grp[i] = seen[key]

    mS = X[small & ctrl].mean(0); sS = mS.std()
    mL = X[large & ctrl].mean(0); sL = mL.std()

    R.p('=' * 74)
    R.p('SUPPLEMENTARY TABLE 5   SNR = 20 log10(A_MAX / sigma_CONTROL)')
    R.p('  A_MAX          max |porosity - mean control| per repetition, averaged')
    R.p('  sigma_CONTROL  temporal std of the block-matched mean control waveform')
    R.p('  window         the full 52 us acquisition window, both terms')
    R.p('=' * 74)
    R.p(f'sigma_CONTROL: al_small {sS:.5f}, al_large {sL:.5f}\n')
    R.v('sigma_control', {'al_small': sS, 'al_large': sL})

    ROWS = [(40, 'Rayleigh (k.a = 0.20)', 2, True), (75, 'Rayleigh (k.a = 0.37)', 3, True),
            (150, 'Rayleigh (k.a = 0.75)', 4, True), (270, 'Transition (k.a = 1.35)', 5, False),
            (335, 'Transition (k.a = 1.67)', 6, False), (780, 'Geometric (k.a = 3.89)', 7, False)]
    R.p(f"{'Size':>7} | {'Regime':23} | {'Density':17} | {'Left':>7} {'Center':>7} {'Right':>7} | {'Range':>6}")
    allv = []; snr = {}
    for um, reg, s, is_small in ROWS:
        blk, ref, sig = (small, mS, sS) if is_small else (large, mL, sL)
        dens = [(3, 'Low (100 pores)'), (4, 'High (200 pores)')] if is_small \
            else [(2, 'Low (50 pores)'), (3, 'High (100 pores)')]
        for d, dlab in dens:
            v = []
            for l in (2, 3, 4):
                Rr = X[blk & (den == d) & (siz == s) & (loc == l)]
                v.append(20 * np.log10(np.abs(Rr - ref).max(1).mean() / sig))
            allv += v; snr[f'{um}um_{dlab}'] = v
            R.p(f'{um:5} um | {reg:23} | {dlab:17} | ' + ' '.join(f'{x:7.2f}' for x in v) + f' | {max(v)-min(v):6.2f}')
    allv = np.array(allv)
    R.p(f'\nrange {allv.min():.2f} to {allv.max():.2f} dB, {(allv < 4).sum()} of {len(allv)} configurations below 4 dB')
    R.v('snr_table_dB', snr)
    with open(os.path.join(out('paper_numbers'), 'snr_table.json'), 'w') as f_:
        json.dump(make_json_serializable(snr), f_)

    R.p('\n' + '=' * 74); R.p('CONVENTIONAL DESCRIPTORS'); R.p('=' * 74)
    for lbl, blk, sel in [('40 to 150 um', small, np.isin(siz, [2, 3, 4])),
                          ('270 to 780 um', large, np.isin(siz, [5, 6, 7]))]:
        Cc = X[blk & ctrl]; cmn = Cc.mean(0)
        cc = np.array([np.abs(Cc[i] - np.delete(Cc, i, 0).mean(0)).max() for i in range(len(Cc))])
        pc = np.abs(X[blk & sel] - cmn).max(1)
        pv = stats.ttest_ind(pc, cc, equal_var=False).pvalue
        R.p(f'  {lbl}: porosity {pc.mean():.3f}, control scatter {cc.mean():.3f}, '
            f'ratio {pc.mean()/cc.mean():.2f}x, Welch p = {pv:.2f}')
        R.v(f'peak_vs_control_{lbl}', {'porosity': pc.mean(), 'control': cc.mean(), 'ratio': pc.mean() / cc.mean(), 'p': pv})

    tu = np.arange(N) / 26.9e6 * 1e6
    pr, cd = tu < 10, tu >= 10
    F = np.abs(np.fft.rfft(X, axis=1)); f = np.fft.rfftfreq(N, 1 / 26.9e6) / 1e6
    NAMES = ['peak amplitude', 'peak-to-peak', 'full-signal RMS', 'coda RMS',
             'baseline-subtracted peak', 'baseline-subtracted RMS',
             'coda-to-primary energy ratio', 'time-of-flight', 'spectral centroid']

    def desc(idx, cmn):
        A = X[idx]; Dd = A - cmn
        return np.column_stack([
            np.abs(A[:, pr]).max(1), A[:, pr].max(1) - A[:, pr].min(1),
            np.sqrt((A ** 2).mean(1)), np.sqrt((A[:, cd] ** 2).mean(1)),
            np.abs(Dd).max(1), np.sqrt((Dd ** 2).mean(1)),
            (A[:, cd] ** 2).sum(1) / (A[:, pr] ** 2).sum(1),
            tu[pr][np.abs(A[:, pr]).argmax(1)], (F[idx] @ f) / F[idx].sum(1)])

    def best_threshold(v, yy):
        b = (0, v.min(), 1)
        for c in np.unique(np.quantile(v, np.linspace(0, 1, 201))):
            for s in (1, -1):
                a = np.mean(((v - c) * s > 0).astype(int) == (yy == yy.max()).astype(int))
                if a > b[0]:
                    b = (a, c, s)
        return b

    res = {t2: [] for t2 in y}; chosen = {t2: np.zeros(9) for t2 in y}
    for rep in range(10):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for task, yy in y.items():
            for tr, te in cv.split(X, yy, grp):
                cmn = X[tr][det[tr] == 1].mean(0)
                Dtr, Dte = desc(tr, cmn), desc(te, cmn)
                acc = []
                for j in range(9):
                    if task == 'detection':
                        _, c, s = best_threshold(Dtr[:, j], yy[tr])
                        pp = np.where((Dte[:, j] - c) * s > 0, 2, 1)
                    else:
                        cls = np.unique(yy[tr])
                        ct = np.array([Dtr[yy[tr] == q, j].mean() for q in cls])
                        pp = cls[np.abs(Dte[:, j, None] - ct[None, :]).argmin(1)]
                    acc.append(np.mean(pp == yy[te]))
                chosen[task] += np.array(acc); res[task].append(max(acc))

    R.p('\n  best conventional descriptor, 10x5 grouped CV')
    for task in y:
        v = 100 * np.array(res[task])
        vals, c = np.unique(y[task], return_counts=True)
        R.p(f'    {task:10s} {v.mean():5.1f} +/- {v.std():4.1f}   '
            f'(best: {NAMES[int(chosen[task].argmax())]}; majority class {100*c.max()/n:.1f}%)')
        R.v(f'conventional_{task}', {'mean': v.mean(), 'std': v.std(), 'best': NAMES[int(chosen[task].argmax())],
                                     'majority': 100 * c.max() / n})

    # same descriptors, fitted on all 205 training aluminium rows, scored once on the held-out
    # aluminium rows (the protocol of Table 2). Descriptor chosen by training accuracy; the best
    # held-out descriptor (chosen with hindsight, optimistic) is printed as well.
    Xh, Yh, mh = load_dataset(AL_HELDOUT); alh = mh == 'aluminium'; Xh, Yh = Xh[alh], Yh[alh]
    cmn = X[ctrl].mean(0)
    def desc_rows(A):
        Fa = np.abs(np.fft.rfft(A, axis=1)); Dd = A - cmn
        return np.column_stack([
            np.abs(A[:, pr]).max(1), A[:, pr].max(1) - A[:, pr].min(1),
            np.sqrt((A ** 2).mean(1)), np.sqrt((A[:, cd] ** 2).mean(1)),
            np.abs(Dd).max(1), np.sqrt((Dd ** 2).mean(1)),
            (A[:, cd] ** 2).sum(1) / (A[:, pr] ** 2).sum(1),
            tu[pr][np.abs(A[:, pr]).argmax(1)], (Fa @ f) / Fa.sum(1)])
    Dtr_all, Dho = desc_rows(X), desc_rows(Xh)
    R.p('\n  conventional descriptors on the held-out aluminium rows (fitted on the 205 training rows)')
    conv_ho = {}
    for task, c_ in [('detection', 3), ('location', 2), ('size', 1), ('density', 0)]:
        yy, yh = _Y[:, c_], Yh[:, c_]; tra, tea = [], []
        for j in range(9):
            if task == 'detection':
                _, c, s = best_threshold(Dtr_all[:, j], yy)
                ptr = np.where((Dtr_all[:, j] - c) * s > 0, 2, 1); pte = np.where((Dho[:, j] - c) * s > 0, 2, 1)
            else:
                cls = np.unique(yy); ct = np.array([Dtr_all[yy == q, j].mean() for q in cls])
                ptr = cls[np.abs(Dtr_all[:, j, None] - ct).argmin(1)]; pte = cls[np.abs(Dho[:, j, None] - ct).argmin(1)]
            tra.append(np.mean(ptr == yy)); tea.append(np.mean(pte == yh))
        js, jb = int(np.argmax(tra)), int(np.argmax(tea)); maj = 100 * np.bincount(yh).max() / len(yh)
        conv_ho[task] = {'chosen': NAMES[js], 'heldout': 100 * tea[js], 'best_hindsight': NAMES[jb],
                         'best_hindsight_acc': 100 * tea[jb], 'majority': maj}
        R.p(f'    {task:10s} {NAMES[js]:28s} {100*tea[js]:5.1f}%   (hindsight best: {NAMES[jb]}, '
            f'{100*tea[jb]:.1f}%; majority class {maj:.1f}%)')
    R.v('conventional_heldout', conv_ho)

    R.p('\n' + '=' * 74); R.p('WINDOW ABLATION'); R.p('=' * 74)
    tt = np.linspace(0, 52, N)
    cut = int(np.searchsorted(tt, 10))
    R.p(f'10 us boundary at sample {cut} of {N}\n')
    R.p('  difference-signal energy')
    energy = {}
    for key, lbl, m, ref in [('small', '40 to 150 um', small & (det == 2), mS),
                             ('large', '270 to 780 um', large & (det == 2), mL)]:
        Dd = X[m] - ref; e = (Dd ** 2).sum()
        energy[key] = (100 * (Dd[:, :cut] ** 2).sum() / e, 100 * (Dd[:, cut:] ** 2).sum() / e)
        R.p(f'    {lbl}: primary pulse {energy[key][0]:.1f}%, coda {energy[key][1]:.1f}%')
    R.v('energy_split_percent', energy)
    # energy of the received signal itself (not the difference), same traces and windows
    R.p('  received-signal energy')
    energy_total = {}
    for key, lbl, m in [('small', '40 to 150 um porosity', small & (det == 2)),
                        ('large', '270 to 780 um porosity', large & (det == 2)),
                        ('control', 'controls', ctrl), ('all', 'all training aluminium', np.ones(n, bool))]:
        A = X[m]; e = (A ** 2).sum()
        energy_total[key] = (100 * (A[:, :cut] ** 2).sum() / e, 100 * (A[:, cut:] ** 2).sum() / e)
        R.p(f'    {lbl}: primary pulse {energy_total[key][0]:.1f}%, coda {energy_total[key][1]:.1f}%')
    R.v('energy_total_split_percent', energy_total)

    WIN = [('full signal', slice(0, N)), ('primary pulse (0-10 us)', slice(0, cut)),
           ('coda (10-52 us)', slice(cut, N))]
    svm_p = {task: tuned_params(ANALYSIS_CONFIG['ablation_params_run'], task, 'svm') for task in ('detection', 'size')}
    R.p(f'  SVM hyperparameters (tuned in {ANALYSIS_CONFIG["ablation_params_run"]}): {svm_p}')

    def ev(cols, task, grouped, reps=10):
        yy = y[task]; o = []
        for r in range(reps):
            cv = (StratifiedGroupKFold(5, shuffle=True, random_state=r) if grouped
                  else StratifiedKFold(5, shuffle=True, random_state=r))
            it = cv.split(X, yy, grp) if grouped else cv.split(X, yy)
            for tr, te in it:
                sc = StandardScaler().fit(X[tr][:, cols])
                m = SVC(**svm_p[task], random_state=DATA_CONFIG['random_state'], max_iter=SVM_MAX_ITER).fit(sc.transform(X[tr][:, cols]), yy[tr])
                o.append(m.score(sc.transform(X[te][:, cols]), yy[te]))
        return 100 * np.mean(o)

    ablation = {}
    for grouped, lbl in [(True, 'grouped'), (False, 'ungrouped')]:
        R.p(f'\n  {lbl} 10x5 CV')
        b = {}
        for nm, cols in WIN:
            b[nm] = (ev(cols, 'detection', grouped), ev(cols, 'size', grouped))
            R.p(f'    {nm:26s} detection {b[nm][0]:5.1f}   size {b[nm][1]:5.1f}')
        full = b['full signal']
        for nm in ('primary pulse (0-10 us)', 'coda (10-52 us)'):
            R.p(f'      cost of using only {nm:24s} '
                f'detection {full[0]-b[nm][0]:+5.1f} pp   size {full[1]-b[nm][1]:+5.1f} pp')
        ablation[lbl] = b
    R.p(f"\n  majority-class baseline, detection: {100*np.bincount(det)[1:].max()/n:.1f}%")
    ablation['heldout'] = heldout_ablation(R, WIN)
    R.v('window_ablation', ablation)

    # values drawn in Figure 3e/f (was coda_numbers.json)
    e = energy[ANALYSIS_CONFIG['coda_energy_group']]; b = ablation[ANALYSIS_CONFIG['coda_cv']]
    et = energy_total[ANALYSIS_CONFIG['coda_energy_group']]
    coda = {'energy_primary': e[0], 'energy_coda': e[1], 'total_primary': et[0], 'total_coda': et[1]}
    if ANALYSIS_CONFIG['coda_cv'] == 'heldout':
        coda['size_majority_heldout'] = b['_majority_aluminium']['size']
        for t_ in ('detection', 'location', 'size', 'density'):
            for nm_, key_ in [("Full signal", 'full signal'), ("Coda only\n(10-52 $\\mu$s)", 'coda (10-52 us)'),
                              ("Primary only\n(0-10 $\\mu$s)", 'primary pulse (0-10 us)')]:
                coda[f'{nm_}|{t_.capitalize()}'] = b['_detail'][t_][key_]['aluminium']
            coda[f'majority|{t_.capitalize()}'] = b['_majority_aluminium'][t_]
    for nm, key in [("Full signal", 'full signal'), ("Coda only\n(10-52 $\\mu$s)", 'coda (10-52 us)'),
                    ("Primary only\n(0-10 $\\mu$s)", 'primary pulse (0-10 us)')]:
        coda[f'{nm}|Detection'], coda[f'{nm}|Size'] = b[key]
    with open(os.path.join(out('paper_numbers'), 'coda_numbers.json'), 'w') as fjs:
        json.dump(coda, fjs, indent=1)
    R.p(f"\n  Figure 3e/f use: energy of the {ANALYSIS_CONFIG['coda_energy_group']} group, "
        f"{ANALYSIS_CONFIG['coda_cv']} window ablation (saved as coda_numbers.json)")
    return coda


def heldout_ablation(R, WIN, tasks=('detection', 'size', 'location', 'density')):
    """Window ablation on the held-out protocol. Training exactly as run5 (all 245 training rows,
    scaler fitted on the 85% tuning part, SVM tuned by 3-fold CV on that part, final fit on all 245);
    each model is scored once on the held-out file. Full signal = the SVM tuned in run5 (the
    Table 2 model); each window alone = an SVM re-tuned from scratch on that window."""
    X, Y, mat = load_dataset([AL_BOTH, STEEL])
    Xh, Yh, math = load_dataset(AL_HELDOUT)
    al = math == 'aluminium'
    col = {'density': 0, 'size': 1, 'location': 2, 'detection': 3}
    R.p('\n  held-out window ablation (trained on the 245 training rows as run5; '
        f'scored on {al.sum()} held-out aluminium rows, {(~al).sum()} steel rows)')
    pfile = os.path.join(out('paper_numbers'), 'heldout_ablation_params.json')
    saved = json.load(open(pfile)) if (ANALYSIS_CONFIG['reuse_ablation_params'] and os.path.exists(pfile)) else {}
    R.p(f'    window-model hyperparameters: {"reused from " + os.path.basename(pfile) if saved else "tuned now"}')
    res = {}
    for task in tasks:
        y = Y[:, col[task]]; yh = Yh[:, col[task]]
        tr, _va = train_test_split(np.arange(len(y)), test_size=DATA_CONFIG['val_size'],
                                   random_state=DATA_CONFIG['random_state'], shuffle=True, stratify=y)
        res[task] = {}
        for nm, cols in WIN:
            sc = StandardScaler().fit(X[tr][:, cols])
            if nm == 'full signal':
                p = tuned_params('run5_heldout', task, 'svm')      # the Table 2 model
            elif task in saved and nm in saved[task]:
                p = saved[task][nm]
            else:
                p = tune_svm(sc.transform(X[tr][:, cols]), y[tr])
            m = SVC(**p, random_state=DATA_CONFIG['random_state'], max_iter=SVM_MAX_ITER)
            pr = m.fit(sc.transform(X[:, cols]), y).predict(sc.transform(Xh[:, cols]))
            ok = pr == yh
            res[task][nm] = {'aluminium': 100 * ok[al].mean(), 'steel': 100 * ok[~al].mean(),
                             'all': 100 * ok.mean(), 'params': p}
            R.p(f'    {task:10s} {nm:26s} aluminium {100*ok[al].mean():5.1f}   '
                f'steel {100*ok[~al].mean():5.1f}   all {100*ok.mean():5.1f}')
    if not saved:
        with open(pfile, 'w') as fp_:
            json.dump({t_: {nm: res[t_][nm]['params'] for nm, _ in WIN if nm != 'full signal'} for t_ in tasks},
                      fp_, indent=1)
    maj = {t_: 100 * np.bincount(Yh[al, col[t_]]).max() / al.sum() for t_ in tasks}
    R.p('    majority-class baseline, held-out aluminium: ' +
        ', '.join(f'{t_} {v:.1f}%' for t_, v in maj.items()))
    # Figure 3f draws all four tasks on the held-out aluminium rows
    return {nm: (res['detection'][nm]['aluminium'], res['size'][nm]['aluminium']) for nm, _ in WIN} | \
           {'_detail': res, '_majority_aluminium': maj}


# ------------------------------------------------------------------ steel matrices (was 04)
def steel_matrices(R):
    X, Y, mat = load_dataset([AL_BOTH, STEEL])
    Xal, Yal = X[mat == 'aluminium'], Y[mat == 'aluminium']
    Xst, Yst = X[mat == 'steel'], Y[mat == 'steel']
    R.p('\n' + '=' * 74)
    R.p(f'FIGURE 6 d-f  steel out-of-fold predictions ({len(Xst)} steel traces, '
        f'{ANALYSIS_CONFIG["oof_folds"]}-fold, all {len(Xal)} aluminium traces in every training fold)')
    R.p('  SVM re-tuned inside every fold; CNN uses the architecture tuned in run4 for that task')
    R.p('=' * 74)
    results = {}
    for tname, col in [('detection', 3), ('density', 0), ('location', 2), ('size', 1)]:
        preds = oof(Xal, Yal[:, col], Xst, Yst[:, col], tuned_params('run4_al_steel', tname, 'cnn'), f'steel {tname}')
        yst = Yst[:, col]
        labels = sorted(set(yst.tolist()) | set(preds['SVM'].tolist()) | set(preds['CNN'].tolist()))
        for m in preds:
            cm = confusion_matrix(yst, preds[m], labels=labels)
            results[f'{m}_{tname}'] = {'cm': cm.tolist(), 'labels': labels,
                                       'acc': check_matrix(cm, 100 * np.mean(preds[m] == yst), f'steel {m} {tname}')}
            R.p(f'  {m} {tname:10} acc {results[f"{m}_{tname}"]["acc"]:5.1f}%  n={cm.sum()}')
    R.v('fig6_confusion_matrices', results)
    # aluminium-only SVM (run1 parameters, fitted on all aluminium rows) applied to steel without retraining
    R.p('  aluminium-only SVM applied to the steel traces without retraining')
    tr_res = {}
    for tname, col in [('detection', 3), ('location', 2), ('size', 1)]:
        y = Yal[:, col]; tr_, _ = train_test_split(np.arange(len(y)), test_size=DATA_CONFIG['val_size'],
                                                   random_state=DATA_CONFIG['random_state'], shuffle=True, stratify=y)
        sc = StandardScaler().fit(Xal[tr_])
        m = SVC(**tuned_params('run1_combined_al', tname, 'svm'), random_state=DATA_CONFIG['random_state'],
                max_iter=SVM_MAX_ITER).fit(sc.transform(Xal), y)
        ys = Yst[:, col]; a_ = 100 * np.mean(m.predict(sc.transform(Xst)) == ys)
        maj = 100 * np.bincount(ys).max() / len(ys); tr_res[tname] = {'acc': a_, 'majority': maj}
        R.p(f'    {tname:10s} {a_:5.1f}%   (majority class {maj:.1f}%)')
    R.v('al_to_steel_transfer', tr_res)
    with open(os.path.join(out('paper_numbers'), 'fig6_confusion_matrices.json'), 'w') as f:
        json.dump(make_json_serializable(results), f, indent=1)
    return results


# ------------------------------------------------------------------ Supplementary Figure 2 counts
def experimental_matrices(R):
    X, Y, _ = load_dataset(AL_BOTH)
    R.p('\n' + '=' * 74)
    R.p(f'SUPPLEMENTARY FIGURE 2  aluminium out-of-fold SVM predictions ({len(X)} traces, '
        f'{ANALYSIS_CONFIG["oof_folds"]}-fold, SVM re-tuned inside every fold)')
    R.p('=' * 74)
    res = {}
    for tname, col in [('detection', 3), ('size', 1), ('density', 0)]:
        yy = Y[:, col]
        p = oof(np.empty((0, X.shape[1])), np.empty(0, int), X, yy, None, f'al {tname}')['SVM']
        labels = sorted(set(yy.tolist()) | set(p.tolist()))
        cm = confusion_matrix(yy, p, labels=labels)
        res[tname] = {'cm': cm.tolist(), 'labels': labels, 'acc': check_matrix(cm, 100 * np.mean(p == yy), f'al {tname} SVM')}
        R.p(f'  SVM {tname:10} acc {res[tname]["acc"]:5.1f}%  n={cm.sum()}')
    R.v('supp2_confusion_matrices', res)
    return res


def heldout_matrices(R, test='heldout_aluminium', model='SVM'):
    """Supplementary Figure 2: the run5 held-out predictions (no retraining) for every task."""
    R.p('\n' + '=' * 74)
    R.p(f'SUPPLEMENTARY FIGURE 2  run5 held-out predictions, {model}, rows: {test}')
    R.p('=' * 74)
    res = {}
    for tname in ('detection', 'location', 'size', 'density'):
        p = os.path.join(OUT_ROOT, 'results', 'run5_heldout', tname, 'complete_results.json')
        if not os.path.exists(p):
            raise FileNotFoundError(f'{p} missing: run the training for run5_heldout first')
        r = json.load(open(p)); ev = r[sorted(r)[0]]['eval_results'][test][model]
        yt, yp = np.array(ev['y_true']), np.array(ev['y_pred'])
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
        cm = confusion_matrix(yt, yp, labels=labels)
        res[tname] = {'cm': cm.tolist(), 'labels': labels,
                      'acc': check_matrix(cm, ev['accuracy'], f'run5 {test} {tname} {model}')}
        R.p(f'  {model} {tname:10} acc {res[tname]["acc"]:5.1f}%  n={cm.sum()}  labels {labels}')
    R.v('supp2_confusion_matrices', res)
    return res


# ------------------------------------------------------------------ numbers quoted in the text (from the saved runs)
def derived_numbers(R, S5=None):
    """Every remaining number quoted in the paper, computed from the saved training runs and the data files."""
    T = ('detection', 'location', 'size', 'density'); M = ('SVM', '1-NN', 'KNN', 'CNN', 'RF')
    def res(run, task):
        r = json.load(open(os.path.join(OUT_ROOT, 'results', run, task, 'complete_results.json')))
        return r[sorted(r)[0]]
    def maj(y): v, c = np.unique(y, return_counts=True); return 100 * c.max() / len(y)
    D = {}
    R.p('\n' + '=' * 74); R.p('NUMBERS QUOTED IN THE TEXT'); R.p('=' * 74)
    # Table 2
    for run, test, lbl in [('run1_combined_al', 'test_all', 'random split (run1, aluminium)'),
                           ('run5_heldout', 'heldout_aluminium', 'held-out aluminium (run5)')]:
        R.p(f'\nTABLE 2  {lbl}   (detection / location / size / density)')
        rr = {t: res(run, t)['eval_results'][test] for t in T}
        for m in M:
            R.p(f'   {m:5s} ' + '  '.join(f'{rr[t][m]["accuracy"]:5.1f}' for t in T) +
                f'   mean {np.mean([rr[t][m]["accuracy"] for t in T]):5.1f}')
        R.p('   n = ' + str(rr['size']['SVM']['n_test']) + '   majority class: ' +
            '  '.join(f'{maj(rr[t]["SVM"]["y_true"]):5.1f}' for t in T))
        D[f'table2_{run}'] = {t: {m: rr[t][m]['accuracy'] for m in M} for t in T}
    # held-out steel (text) and train-test gaps / inference (Table 2 note, Supp. Table 4, Methods)
    st = {t: res('run5_heldout', t)['eval_results']['heldout_steel']['SVM'] for t in T}
    R.p('\nHELD-OUT STEEL (SVM, n = %d): ' % st['size']['n_test'] + '  '.join(f'{t} {st[t]["accuracy"]:.1f}' for t in T))
    R.p('\nTRAIN-TEST GAP (training accuracy minus held-out aluminium accuracy, mean of 4 tasks) and INFERENCE (sum of 4 tasks)')
    inf = {}
    for m in M:
        g = np.mean([res('run5_heldout', t)['overfitting_timing'][m]['gap_train_test_heldout_aluminium'] for t in T])
        inf[m] = sum(res('run5_heldout', t)['eval_results']['heldout_aluminium'][m]['inference_time_per_sample_ms'] for t in T)
        R.p(f'   {m:5s} gap {g:5.1f} pp   inference {1000*inf[m]:8.0f} us   total cycle (+1 ms acquisition) {inf[m]+1:6.2f} ms')
    cyc = inf['SVM'] + 1
    R.p(f'   SVM: cycle {cyc:.2f} ms = {100*cyc/1e4:.4f}% of a 10 s layer;'
        f' 2,000-layer build (5.6 h): inference {2*inf["SVM"]:.2f} s, full inspection {2*cyc:.2f} s')
    # per-regime random splits
    R.p('\nPER-REGIME RANDOM SPLITS (SVM; detection / location / size / density)')
    for run in ('run2_al_small', 'run3_al_large'):
        rr = {t: res(run, t)['eval_results']['test_all']['SVM'] for t in T}
        R.p(f'   {run:14s} ' + '  '.join(f'{rr[t]["accuracy"]:5.1f}' for t in T) + '   n = ' + str(rr['size']['n_test']) +
            '   majority: ' + '  '.join(f'{maj(rr[t]["y_true"]):4.1f}' for t in T))
    # simulation
    R.p('\nSIMULATION (run6, test set)')
    for t in T:
        e = res('run6_simulation', t)['eval_results']['test_all']
        best = max(M, key=lambda m: e[m]['accuracy'])
        R.p(f'   {t:10s} best {best} {e[best]["accuracy"]:5.1f}%   (all: ' + ', '.join(f'{m} {e[m]["accuracy"]:.1f}' for m in M) +
            f')   majority {maj(e["SVM"]["y_true"]):.1f}%   n = {e["SVM"]["n_test"]}')
    d6 = res('run6_simulation', 'size')['data_info']
    R.p(f'   samples: train {d6["train_size"]} + test {list(d6["tests"].values())[0]}')
    # error structure (held-out SVM) and held-out composition
    ho = {t: res('run5_heldout', t)['eval_results']['heldout_aluminium']['SVM'] for t in T}
    yl, pl = np.array(ho['location']['y_true']), np.array(ho['location']['y_pred'])
    err = yl != pl; lr = ((yl == 2) & (pl == 4)) | ((yl == 4) & (pl == 2))
    yd, pd_ = np.array(ho['detection']['y_true']), np.array(ho['detection']['y_pred'])
    yn, pn = np.array(ho['density']['y_true']), np.array(ho['density']['y_pred'])
    dn_err = yn != pn; dn_34 = dn_err & np.isin(yn, [3, 4]) & np.isin(pn, [3, 4])
    ys_, ps_ = np.array(ho['size']['y_true']), np.array(ho['size']['y_pred'])
    R.p('\nHELD-OUT ERROR STRUCTURE (SVM, aluminium)')
    R.p(f'   location errors {err.sum()}, left<->right {lr.sum()}')
    R.p(f'   size errors {(ys_ != ps_).sum()} of {len(ys_)}')
    R.p(f'   porous detected {((yd == 2) & (pd_ == 2)).sum()} of {(yd == 2).sum()}; controls correct {((yd == 1) & (pd_ == 1)).sum()} of {(yd == 1).sum()}')
    R.p(f'   density errors {dn_err.sum()}, between 100 and 200 pores {dn_34.sum()}')
    m40 = ys_ == 2
    R.p(f'   40 um specimens only: size {100*np.mean(ys_[m40] == ps_[m40]):.1f}%, density {100*np.mean(yn[m40] == pn[m40]):.1f}%')
    # random split: 100-pore class contains small-low and large-high porosities
    X, Y, _ = load_dataset(AL_BOTH); den, siz = Y[:, 0], Y[:, 1]
    _, te = train_test_split(np.arange(len(den)), test_size=DATA_CONFIG['test_sizes'][0],
                             random_state=DATA_CONFIG['random_state'], shuffle=True, stratify=den)
    rd = res('run1_combined_al', 'density')['eval_results']['test_all']['SVM']
    yt, yp = np.array(rd['y_true']), np.array(rd['y_pred']); assert np.array_equal(yt, den[te])
    c3 = yt == 3; sm = c3 & (siz[te] <= 4); lg = c3 & (siz[te] >= 5)
    R.p(f'\nRANDOM SPLIT 100-PORE CLASS (SVM): {(yp[c3] == 3).sum()} of {c3.sum()} '
        f'({100*np.mean(yp[c3] == 3):.0f}%); small {(yp[sm] == 3).sum()}/{sm.sum()}, large {(yp[lg] == 3).sum()}/{lg.sum()}')
    # steel cross-validation location errors (Figure 6e)
    if S5 is not None:
        L = np.array(S5['SVM_location']['cm']); lab = S5['SVM_location']['labels']
        i2, i4 = lab.index(2), lab.index(4)
        R.p(f'STEEL CV LOCATION (SVM): errors {L.sum() - np.trace(L)}, left<->right {L[i2, i4] + L[i4, i2]}')
    # SNR summaries (Supplementary Table 5)
    snr = json.load(open(os.path.join(out('paper_numbers'), 'snr_table.json')))
    by_size = {}
    for k, v in snr.items(): by_size.setdefault(k.split('um')[0], []).extend(v)
    means = {k: np.mean(v) for k, v in by_size.items()}
    ray = [k for k in means if int(k) <= 150]
    R.p('\nSNR SUMMARY')
    R.p(f'   Figure 3a configurations (200 pores, center): ' + ', '.join(f'{k} um {snr[f"{k}um_High (200 pores)"][1]:.2f} dB' for k in ('40', '75', '150')))
    R.p(f'   size-averaged spread {max(means.values())-min(means.values()):.2f} dB; Rayleigh-only spread '
        f'{max(means[k] for k in ray)-min(means[k] for k in ray):.2f} dB; largest spread between positions '
        f'{max(max(v)-min(v) for v in snr.values()):.2f} dB')
    allv = [x for v in snr.values() for x in v]
    rayv = [x for k, v in snr.items() if int(k.split('um')[0]) <= 150 for x in v]
    R.p(f'   configurations below 4 dB: {sum(x < 4 for x in allv)} of {len(allv)} '
        f'(Rayleigh regime {sum(x < 4 for x in rayv)} of {len(rayv)})')
    R.v('derived_numbers', D)


# ------------------------------------------------------------------ Supplementary Figure 1 counts
def check_matrix(cm, stored_acc, what):
    """Accuracy shown with a confusion matrix must be exactly trace / total of that matrix."""
    acc = 100 * np.trace(cm) / cm.sum()
    if abs(acc - stored_acc) > 1e-9:
        raise ValueError(f'{what}: matrix gives {acc:.6f}% but the stored accuracy is {stored_acc:.6f}%')
    return acc


def simulation_matrices(R):
    """Supplementary Figure 1 from the saved run6_simulation results (RF, 80/20 test set), no retraining."""
    R.p('\n' + '=' * 74)
    R.p('SUPPLEMENTARY FIGURE 1  simulation, RF, test set of run6_simulation (80/20 split, tuned on the '
        'training part only; the same predictions as the run6 results table)')
    R.p('=' * 74)
    res = {}
    for tname in ('density', 'location', 'size'):
        p = os.path.join(OUT_ROOT, 'results', 'run6_simulation', tname, 'complete_results.json')
        if not os.path.exists(p):
            raise FileNotFoundError(f'{p} missing: run the training for run6_simulation first')
        r = json.load(open(p)); ev = r[sorted(r)[0]]['eval_results']['test_all']['RF']
        yt, yp = np.array(ev['y_true']), np.array(ev['y_pred'])
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
        cm = confusion_matrix(yt, yp, labels=labels)
        acc = check_matrix(cm, ev['accuracy'], f'run6_simulation {tname} RF')
        res[tname] = {'cm': cm.tolist(), 'labels': labels, 'acc': acc}
        R.p(f'  RF {tname:10} acc {acc:5.1f}%  n={cm.sum()}   (run6 results table: {ev["accuracy"]:.2f}%)')
    R.v('supp1_confusion_matrices', res)
    return res


def draw_matrix_row(mats, titles, stem):
    """Three confusion matrices side by side (Supplementary Figures 1-2 layout)."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.72))
    for ax, M, ti in zip(axes, mats, titles):
        confusion(ax, M, ti)
    fig.tight_layout(pad=0.4, w_pad=3.4); save(fig, stem)


def supp_matrix_side():
    """Side (inches) of one confusion matrix in the Supplementary Figure 1-2 layout, measured by
    building that layout, so Figure 6 matches it exactly with the fonts of this machine."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.72))
    for ax in axes:
        confusion(ax, np.eye(3, dtype=int) + 1, "a) SVM \u2013 Detection\nAcc: 100%")
    fig.tight_layout(pad=0.4, w_pad=3.4); fig.canvas.draw()
    bb = axes[1].get_images()[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.close(fig)
    return min(bb.width, bb.height)


def draw_figure6(ST, stem="figure6"):
    """Figure 6: control waveforms, difference signals and the steel confusion matrices in ST."""
    setup_style()
    Xc, _Yc, matc = load_dataset([AL_BOTH, STEEL])
    denc, sizc, locc, detc = _Yc[:, 0], _Yc[:, 1], _Yc[:, 2], _Yc[:, 3]
    alu_s = (matc == 'aluminium') & (sizc <= 4)      # al_small block (with the controls)
    stl = matc == 'steel'
    mA6 = Xc[alu_s & (detc == 1)].mean(0); aA6 = np.abs(mA6).max()
    mS6 = Xc[stl & (detc == 1)].mean(0); aS6 = np.abs(mS6).max()
    pA6 = Xc[alu_s & (denc == 4) & (sizc == 2) & (locc == 3)].mean(0)
    pS6 = Xc[stl & (sizc == 3) & (locc == 3)].mean(0)      # steel size class 3 = keyhole (2 = lack-of-fusion)
    dA6, dS6 = pA6 - mA6, pS6 - mS6

    # Rows a-c keep the paper layout exactly (in inches). Row d-f is made tall enough for confusion
    # matrices of the same size as in the Supplementary Figures (same gridspec method as Figure 3).
    S = supp_matrix_side()
    H0, top0, bot0, s0, r0 = 6.25, 0.945, 0.075, 0.72, np.array([1.26, 1.0, 1.20])
    cell = (top0 - bot0) * H0 / (3 + 2 * s0); unit = cell * 3 / r0.sum()
    r = r0.copy(); r[2] = S / unit
    s1 = s0 * r0.sum() / r.sum(); cell1 = cell * r.sum() / r0.sum()
    H = H0 + (3 * cell1 + 2 * s0 * cell) - (top0 - bot0) * H0
    fig = plt.figure(figsize=(6.5, H))
    gs = fig.add_gridspec(3, 6, hspace=s1, wspace=1.55, left=0.115, right=0.985,
                          top=1 - (1 - top0) * H0 / H, bottom=bot0 * H0 / H, height_ratios=list(r))
    for kk, (lo, hi, lbl) in enumerate([(0, 10, "a) Control waveforms, 0-10 $\\mu$s"),
                                        (25, 40, "b) Control waveforms, 25-40 $\\mu$s")]):
        az = fig.add_subplot(gs[0, 3*kk:3*kk+3]); m = (t >= lo) & (t <= hi)
        az.plot(t[m], mA6[m], color=C_AL, lw=0.6, label="Aluminum")
        az.plot(t[m], mS6[m], color=C_ST, lw=0.6, label="Steel 316L")
        az.set_xlim(lo, hi)
        if lo == 25:
            az.set_xticks([25, 30, 35, 40])
        mx = max(np.abs(mA6[m]).max(), np.abs(mS6[m]).max())
        az.set_ylim(-1.50 * mx, 1.08 * mx)
        az.legend(loc="lower right", fontsize=8, ncol=2, framealpha=1,
                  borderpad=0.3, labelspacing=0.25, columnspacing=0.9,
                  handlelength=1.2)
        fin(az, "Time ($\\mu$s)", YLAB_V, lbl)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, dA6, color=C_AL, lw=0.6, label="Aluminum, 40 $\\mu$m artificial")
    ax.plot(t, dS6, color=C_ST, lw=0.6, label="Steel 316L, keyhole porosity")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlim(0, T)
    leg = ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=1)
    fin(ax, "Time ($\\mu$s)", YLAB_DV,
        "c) Difference signals in both materials")
    lower_until_clear(ax, leg, [dA6, dS6])

    ST_DET, ST_LOC, ST_SIZE = (np.array(ST[f'SVM_{k}']['cm']) for k in ('detection', 'location', 'size'))
    acc = lambda M: 100 * np.trace(M) / M.sum()
    for kk, (M, ti) in enumerate([(ST_DET, f"d) SVM – Detection\nAcc: {acc(ST_DET):.1f}%"),
                                  (ST_LOC, f"e) SVM – Location\nAcc: {acc(ST_LOC):.1f}%"),
                                  (ST_SIZE, f"f) SVM – Size\nAcc: {acc(ST_SIZE):.1f}%")]):
        slot = fig.add_subplot(gs[2, 2*kk:2*kk+2]).get_position(); fig.delaxes(fig.axes[-1])
        # S x S inches; the row spans exactly the width of panel c (left 0.115 to right 0.985)
        gap = (0.985 - 0.115 - 3 * S / 6.5) / 2
        confusion(fig.add_axes([0.115 + kk * (S / 6.5 + gap), slot.y0, S / 6.5, slot.height]), M, ti)
    save(fig, stem)
    return aA6, aS6, mA6, mS6, ST_DET, ST_LOC, ST_SIZE


# ------------------------------------------------------------------ figures (was 01)
def make_figures(R, CODA, ST, SUPP2, SUPP1):
    setup_style()
    Xs_, Ys_, _ = load_dataset(AL_SMALL)
    Xl_, Yl_, _ = load_dataset(AL_LARGE)

    def cfg(group, d=None, s=None, l=None):
        X_, Y_ = (Xs_, Ys_) if group == "al_small" else (Xl_, Yl_)
        den_, siz_, loc_, det_ = Y_[:, 0], Y_[:, 1], Y_[:, 2], Y_[:, 3]
        m = (det_ == 1) if d is None else ((den_ == d) & (siz_ == s) & (loc_ == l))
        return X_[m]

    mS = cfg("al_small").mean(0); aS = np.abs(mS).max()
    mL = cfg("al_large").mean(0); aL = np.abs(mL).max()
    SIZES = [(40,"al_small",4,2,3,mS,aS), (75,"al_small",4,3,3,mS,aS),
             (150,"al_small",4,4,3,mS,aS), (270,"al_large",3,5,3,mL,aL),
             (335,"al_large",3,6,3,mL,aL), (780,"al_large",3,7,3,mL,aL)]

    # ------------------------------------------------------------------ Figure 3
    # rows a-d, gaps and margins identical to the paper layout (in inches); only row e/f is 60%
    # taller, so the legend in f clears the bar labels
    fig = plt.figure(figsize=(6.5, 7.3924))
    gs = fig.add_gridspec(4, 2, hspace=0.53628, wspace=0.34, left=0.09, right=0.985,
                          top=0.95891, bottom=0.05661, height_ratios=[0.88, 0.90, 1.12, 1.632])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, mS, color="k", lw=0.6, label="Control")
    for d, g, dd, ss, ll, _, _ in SIZES[:3]:
        ax.plot(t, cfg(g, dd, ss, ll).mean(0), color=C[d], lw=0.6,
                label=f"{d}$\\mu$m, 200 pores, Center")
    for x in (10, 25, 40):
        ax.axvline(x, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(0, T); ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=1)
    fin(ax, "Time ($\\mu$s)", YLAB_V,
        "a) Overlaid raw waveforms for different porosity sizes")
    for k, (lo, hi, lbl) in enumerate([(0, 10, "b) Zoom: 0-10 $\\mu$s"),
                                       (25, 40, "c) Zoom: 25-40 $\\mu$s")]):
        az = fig.add_subplot(gs[1, k]); m = (t >= lo) & (t <= hi)
        az.plot(t[m], mS[m], color="k", lw=0.6)
        for d, g, dd, ss, ll, _, _ in SIZES[:3]:
            az.plot(t[m], cfg(g, dd, ss, ll).mean(0)[m], color=C[d], lw=0.6)
        az.set_xlim(lo, hi); fin(az, "Time ($\\mu$s)", YLAB_V, lbl)
    ax = fig.add_subplot(gs[2, :])
    for d, g, dd, ss, ll, ref, _ in SIZES[:3]:
        ax.plot(t, cfg(g, dd, ss, ll).mean(0) - ref, color=C[d], lw=0.6,
                label=f"{d}$\\mu$m, 200 pores, Center - Control")
    ax.axvspan(0, 10, color=C_PRIM, alpha=0.10, lw=0)
    ax.axvspan(10, T, color=C_CODA, alpha=0.10, lw=0)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlim(0, T)
    ax.set_ylim(ax.get_ylim()[0] * 2.7, ax.get_ylim()[1])
    ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=1,
              borderpad=0.3, labelspacing=0.25, columnspacing=0.9,
              handlelength=1.2)
    fin(ax, "Time ($\\mu$s)", YLAB_DV,
        "d) Difference signals reveal size-dependent features")

    # e) share of energy in each window: the received signal itself vs the difference signal
    axe = fig.add_subplot(gs[3, 0])
    we, xe = 0.34, np.arange(2)
    for k, (nm, col, keys) in enumerate([("Primary pulse (0-10 $\\mu$s)", C_PRIM, ("total_primary", "energy_primary")),
                                         ("Coda (10-52 $\\mu$s)", C_CODA, ("total_coda", "energy_coda"))]):
        v = [CODA[kk] for kk in keys]
        axe.bar(xe + (k - 0.5) * we, v, width=we, color=col, edgecolor="k", linewidth=0.6, label=nm)
        for x, yv in zip(xe + (k - 0.5) * we, v):
            axe.text(x, yv + 2.0, f"{yv:.0f}%", ha="center", fontsize=8)
    axe.set_xticks(xe)
    axe.set_xticklabels(["Received signal", "Porosity-induced\nchange"])
    axe.set_ylim(0, 128); axe.set_yticks([0, 20, 40, 60, 80, 100])
    axe.legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=1,
               handlelength=1.0, borderpad=0.3, handletextpad=0.4)
    fin(axe, "", "Energy (%)",
        "e) Share of energy in each window")

    # f) held-out accuracy for each window alone, all four tasks, with each task's majority-class baseline
    axf = fig.add_subplot(gs[3, 1])
    tasks = ["Detection", "Location", "Size", "Density"]
    wins = [("Full signal", "Full", C_FULL), ("Coda only\n(10-52 $\\mu$s)", "Coda only", C_CODA),
            ("Primary only\n(0-10 $\\mu$s)", "Primary only", C_PRIM)]
    w, xs = 0.26, np.arange(len(tasks))
    for k, (key, lab, col) in enumerate(wins):
        v = [CODA[f"{key}|{tk}"] for tk in tasks]
        axf.bar(xs + (k - 1) * w, v, width=w, color=col, edgecolor="k", linewidth=0.6, label=lab)
    for x, tk in zip(xs, tasks):
        b_ = CODA.get(f"majority|{tk}")
        if b_ is not None:
            axf.plot([x - 1.6 * w, x + 1.6 * w], [b_, b_], color="k", ls="--", lw=0.8)
    axf.set_xticks(xs); axf.set_xticklabels(tasks, fontsize=8.5)
    axf.set_ylim(0, 128); axf.set_yticks([0, 25, 50, 75, 100])
    axf.legend(fontsize=7.5, loc="upper center", ncol=3, framealpha=1, handlelength=1.0,
               borderpad=0.3, columnspacing=0.9, handletextpad=0.4)
    fin(axf, "", "Held-out accuracy (%)", "f) Accuracy trained on each window alone")
    save(fig, "figure3")

    # ------------------------------------------------------------------ Figure 4
    DENS = [(0, None, "k", "Control"),
            (100, (3, 3, 3), C[40], "75$\\mu$m, 100 pores, Center"),
            (200, (4, 3, 3), C[75], "75$\\mu$m, 200 pores, Center")]
    fig = plt.figure(figsize=(6.5, 4.01))
    gs = fig.add_gridspec(2, 2, hspace=0.60, wspace=0.30, left=0.09, right=0.985,
                          top=0.925, bottom=0.11)
    ax = fig.add_subplot(gs[0, :])
    for n_, c_, col, lab in DENS:
        Rr = cfg("al_small") if c_ is None else cfg("al_small", *c_)
        ax.plot(t, Rr.mean(0), color=col, lw=0.6, label=lab)
    ax.set_xlim(0, T); ax.legend(loc="lower right", fontsize=8, framealpha=1)
    fin(ax, "Time ($\\mu$s)", YLAB_V, "a) Effect of porosity density")
    az = fig.add_subplot(gs[1, 0]); m = t <= 10
    for n_, c_, col, lab in DENS:
        Rr = cfg("al_small") if c_ is None else cfg("al_small", *c_)
        az.plot(t[m], Rr.mean(0)[m], color=col, lw=0.6)
    az.set_xlim(0, 10)
    fin(az, "Time ($\\mu$s)", YLAB_V, "b) Zoom: main pulse region")
    axc = fig.add_subplot(gs[1, 1]); xs, ms, ss_, f4 = [], [], [], {}
    for n_, c_, col, lab in DENS:
        Rr = cfg("al_small") if c_ is None else cfg("al_small", *c_)
        v = np.array([cx(r) / np.abs(r).max() for r in Rr])
        f4[n_] = v; xs.append(n_); ms.append(v.mean()); ss_.append(v.std(ddof=1))
    axc.errorbar(xs, ms, yerr=ss_, marker="o", ms=7, lw=1.8, color="#0072BD", capsize=4)
    axc.set_xticks(xs)
    fin(axc, "Porosity density (number of pores)",
        "Normalized complexity ($\\mu$s$^{-2}$)", "c) Normalized complexity vs density")
    save(fig, "figure4")

    # ------------------------------------------------------------------ Figure 5
    fig = plt.figure(figsize=(6.5, 4.70))
    gs = fig.add_gridspec(2, 2, hspace=0.60, wspace=0.34, left=0.095, right=0.985,
                          top=0.935, bottom=0.105, height_ratios=[1.30, 1.0])
    ax = fig.add_subplot(gs[0, :])
    for d, g, dd, ss, ll, ref, amp in SIZES:
        ax.plot(t, 100 * (cfg(g, dd, ss, ll).mean(0) - ref) / amp, color=C[d], lw=0.6,
                label=f"{d} $\\mu$m")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlim(0, T)
    ax.set_ylim(ax.get_ylim()[0] * 1.75, ax.get_ylim()[1])
    ax.legend(loc="lower right", fontsize=8, ncol=3, framealpha=1,
              columnspacing=0.9, handlelength=1.2, borderpad=0.3,
              labelspacing=0.25)
    fin(ax, "Time ($\\mu$s)", "Normalized difference (%)",
        "a) Normalized difference signals across full size range")
    cm_, cs_, pm, ps, lb, f5 = [], [], [], [], [], {}
    for d, g, dd, ss, ll, ref, amp in SIZES:
        Rr = cfg(g, dd, ss, ll)
        cv = np.array([cx(r - ref) / amp for r in Rr])
        pv = np.array([100 * np.abs(r - ref).max() / amp for r in Rr])
        f5[d] = (cv, pv)
        cm_.append(cv.mean()); cs_.append(cv.std(ddof=1))
        pm.append(pv.mean()); ps.append(pv.std(ddof=1)); lb.append(str(d))
    xi = np.arange(6)
    axb = fig.add_subplot(gs[1, 0])
    axb.errorbar(xi, cm_, yerr=cs_, marker="o", ms=7, lw=1.8, color="#0072BD", capsize=4)
    axb.set_xticks(xi); axb.set_xticklabels(lb); center_ylim(axb, cm_, cs_)
    fin(axb, "Porosity size ($\\mu$m)",
        "Normalized difference complexity ($\\mu$s$^{-2}$)",
        "b) Difference-signal complexity vs size")
    axc = fig.add_subplot(gs[1, 1])
    axc.errorbar(xi, pm, yerr=ps, marker="s", ms=7, lw=1.8, color="#D95319", capsize=4)
    axc.set_xticks(xi); axc.set_xticklabels(lb); center_ylim(axc, pm, ps)
    fin(axc, "Porosity size ($\\mu$m)", "Normalized peak difference (%)",
        "c) Peak difference vs porosity size")
    save(fig, "figure5")

    aA6, aS6, mA6, mS6, ST_DET, ST_LOC, ST_SIZE = draw_figure6(ST)
    acc = lambda M: 100 * np.trace(M) / M.sum()

    # ------------------------------------------------- Supplementary Figures 1-2
    # Supp 1: simulation study
    D_rf, L_rf, S_rf = (np.array(SUPP1[k]['cm']) for k in ('density', 'location', 'size'))
    draw_matrix_row([D_rf, L_rf, S_rf],
                    [f"a) RF – Density\nAcc: {acc(D_rf):.0f}%", f"b) RF – Location\nAcc: {acc(L_rf):.0f}%",
                     f"c) RF – Size\nAcc: {acc(S_rf):.0f}%"], "supp_figure1")
    # Supp 2: held-out aluminium, SVM, all four tasks, ticks are the class codes
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.2))   # one row, same width as Supplementary Figure 1
    for ax, (k, lab) in zip(axes.flat, [('detection', 'a) SVM – Detection'), ('location', 'b) SVM – Location'),
                                         ('size', 'c) SVM – Size'), ('density', 'd) SVM – Density')]):
        M = np.array(SUPP2[k]['cm'])
        confusion(ax, M, f"{lab}\nAcc: {acc(M):.1f}%", ticks=SUPP2[k]['labels'])
    fig.tight_layout(pad=0.4, w_pad=1.6); save(fig, "supp_figure2")

    # ------------------------------------------------------------------ numbers
    R.p('\n' + '=' * 74); R.p('FIGURE NUMBERS'); R.p('=' * 74)
    R.p("FIGURE 4c  normalized complexity (us^-2)")
    for n_ in (0, 100, 200):
        R.p(f"   {n_:3d} pores  {f4[n_].mean():6.2f} +/- {f4[n_].std(ddof=1):.2f}")
    p4 = stats.ttest_ind(f4[0], f4[200], equal_var=False)[1]
    R.p(f"   control vs 200 pores  p = {p4:.3f}"
        f"   change {100*(f4[0].mean()-f4[200].mean())/f4[0].mean():.1f}%")
    R.v('fig4c', {str(k): {'mean': v.mean(), 'sd': v.std(ddof=1)} for k, v in f4.items()} | {'p_control_vs_200': p4})
    R.p("\nFIGURE 5b/5c")
    for d in (40, 75, 150, 270, 335, 780):
        cv, pv = f5[d]
        R.p(f"   {d:4d} um  complexity {cv.mean():6.2f} +/- {cv.std(ddof=1):4.2f}"
            f"   peak {pv.mean():5.1f}% +/- {pv.std(ddof=1):4.1f}")
    R.v('fig5', {str(d): {'complexity': [f5[d][0].mean(), f5[d][0].std(ddof=1)],
                          'peak_pct': [f5[d][1].mean(), f5[d][1].std(ddof=1)]} for d in f5})
    for nm, k in (("complexity", 0), ("peak difference", 1)):
        a = np.concatenate([f5[d][k] for d in (40, 75, 150)])
        b = np.concatenate([f5[d][k] for d in (270, 335, 780)])
        R.p(f"   {nm:16s} Rayleigh {a.mean():6.2f} -> geometric {b.mean():6.2f}"
            f"   factor {b.mean()/a.mean():.2f}   p = {stats.ttest_ind(a,b,equal_var=False)[1]:.1e}")
    R.p("\nFIGURE 6")
    R.p(f"   control peak amplitude   aluminium {aA6:.4f}   steel {aS6:.4f}")
    R.p(f"   primary pulse arrival    aluminium {t[np.abs(mA6).argmax()]:.2f} us"
        f"   steel {t[np.abs(mS6).argmax()]:.2f} us")
    for nm, M in [("SVM detection", ST_DET), ("SVM location", ST_LOC), ("SVM size", ST_SIZE)]:
        R.p(f"   {nm:14s} N={M.sum()}  accuracy {acc(M):5.1f}%")
    R.p("\nSUPPLEMENTARY FIGURE 1")
    for nm, M in [("RF density", D_rf), ("RF location", L_rf), ("RF size", S_rf)]:
        R.p(f"   {nm:14s} N={M.sum()}  accuracy {acc(M):5.1f}%")
    R.p("\nSUPPLEMENTARY FIGURE 2")
    for k in ('detection', 'location', 'size', 'density'):
        M = np.array(SUPP2[k]['cm'])
        R.p(f"   SVM {k:10s} N={M.sum()}  accuracy {acc(M):5.1f}%")


def redraw_figures():
    """Redraw all figures from the numbers saved by the last analysis (no training, no tuning)."""
    d = os.path.join(OUT_ROOT, 'paper_numbers')
    CODA = json.load(open(os.path.join(d, 'coda_numbers.json')))
    ST = json.load(open(os.path.join(d, 'fig6_confusion_matrices.json')))
    pn = json.load(open(os.path.join(d, 'paper_numbers.json')))
    make_figures(Report(), CODA, ST, pn['supp2_confusion_matrices'], pn['supp1_confusion_matrices'])
    print(f'Figures redrawn in {out("figures")}')


def run_analysis():
    tee = Tee(os.path.join(out('logs'), 'analysis.log'), 'w'); sys.stdout = tee
    try:
        R = Report()
        R.p(BANNER)
        R.p(f'Paper numbers - generated {time.strftime("%Y-%m-%d %H:%M")} - budget '
            f'{TUNE_CONFIG["n_studies"]} studies x {TUNE_CONFIG["n_trials"]} trials')
        import platform, sklearn
        R.p(f'Environment: Python {platform.python_version()}, {platform.system()} {platform.machine()}, '
            f'numpy {np.__version__}, scikit-learn {sklearn.__version__}, optuna {optuna.__version__}, '
            f'torch {torch.__version__}')
        R.p('Data: training files for all analyses; the held-out file is used only to score models '
            'fitted on the training rows (window ablation, conventional descriptors, Supplementary Figure 2).\n')
        CODA = paper_numbers(R)
        ST = steel_matrices(R)
        SUPP2 = heldout_matrices(R)
        SUPP1 = simulation_matrices(R)
        derived_numbers(R, ST)
        make_figures(R, CODA, ST, SUPP2, SUPP1)
        R.save()
        print(f'\nFigures: {out("figures")}\nPaper numbers: {out("paper_numbers")}')
    finally:
        sys.stdout = sys.__stdout__; tee.close()


# ====================== MAIN ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='tiny budget smoke test (outputs_quick/)')
    ap.add_argument('--runs', nargs='*', help='only these training runs')
    ap.add_argument('--analysis-only', action='store_true', help='skip training, only figures + numbers')
    ap.add_argument('--no-analysis', action='store_true', help='training only')
    ap.add_argument('--figures-only', action='store_true',
                    help='only redraw the figures from the saved paper numbers (after a finished run)')
    ap.add_argument('--amics', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.amics:
        import base64; print(base64.b64decode(''.join(_SIG)).decode()); return
    print(BANNER)
    global OUT_ROOT, TUNE_EPOCHS
    if a.quick:
        TUNE_CONFIG.update(n_trials=2, n_studies=1); FINAL_EPOCHS.update(cnn=3, lstm=3); TUNE_EPOCHS = 2
        OUT_ROOT = os.path.join(HERE, 'outputs_quick')
    os.makedirs(OUT_ROOT, exist_ok=True)
    if a.figures_only:
        redraw_figures(); return
    print(f'CUDA: {GPU_CONFIG["use_cuda"]}' + (f' ({torch.cuda.get_device_name(0)})' if GPU_CONFIG['use_cuda'] else ''))
    print(f'Budget: {TUNE_CONFIG["n_studies"]} studies x {TUNE_CONFIG["n_trials"]} trials, CNN {FINAL_EPOCHS["cnn"]} epochs')
    if not a.analysis_only:
        run_training(a.runs)
    if not a.no_analysis:
        run_analysis()


if __name__ == '__main__':
    # saved full models pickle their class as run_all.CNN1D, so they load wherever run_all.py is importable
    sys.modules['run_all'] = sys.modules['__main__']
    for _c in (CNN1D, LSTM_RNN, TimeSeriesDataset):
        _c.__module__ = 'run_all'
    main()