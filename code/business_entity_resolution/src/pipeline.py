"""
Business Entity Resolution Pipeline — Scalable Implementation
==============================================================
Handles 1.7M S1 × 10M S2/S3 scale with efficient blocking and batch processing.

Usage:
    python pipeline.py --mode full --sample 0.05   # quick dev run on 5% sample
    python pipeline.py --mode full                   # full training + test inference
    python pipeline.py --mode test                   # test inference only (needs trained model)
"""
import os
import sys
import re
import time
import argparse
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import lightgbm as lgb
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

###############################################################################
# PATHS
###############################################################################
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR  = os.path.join(BASE_DIR, 'code', 'business_entity_resolution', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

###############################################################################
# TEXT NORMALIZATION  (vectorized via .apply — avoid row-iteration)
###############################################################################
_SUFFIX_RE = re.compile(
    r'\b(llc|inc|corp|corporation|ltd|limited|co|company|pllc|llp|lp|plc|'
    r'sa|sarl|sas|eurl|srl|gmbh|ag|pvt|private|enterprises?)\b', re.I)
_ADDR_ABBREV = {
    'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'dr': 'drive',
    'rd': 'road', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'cir': 'circle', 'hwy': 'highway', 'pkwy': 'parkway', 'ter': 'terrace',
    'apt': 'apartment', 'ste': 'suite', 'fl': 'floor',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
}
_PUNCT_RE = re.compile(r'[^\w\s]')
_MULTI_WS = re.compile(r'\s+')

def _norm(text):
    if not isinstance(text, str) or not text:
        return ''
    t = text.lower()
    t = t.replace('&', ' and ')
    t = _PUNCT_RE.sub(' ', t)
    t = _MULTI_WS.sub(' ', t).strip()
    return t

def norm_name(text):
    t = _norm(text)
    t = _SUFFIX_RE.sub('', t)
    t = _MULTI_WS.sub(' ', t).strip()
    return t

def norm_addr(text):
    t = _norm(text)
    tokens = t.split()
    return ' '.join(_ADDR_ABBREV.get(tk, tk) for tk in tokens)

_ZIP_US = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
_PIN_IN = re.compile(r'\b(\d{6})\b')

def extract_postal(addr, country=''):
    if not isinstance(addr, str):
        return ''
    country_l = country.lower() if isinstance(country, str) else ''
    if country_l == 'india':
        m = _PIN_IN.search(addr)
        return m.group(1) if m else ''
    m = _ZIP_US.search(addr)
    if m:
        return m.group(1)
    m = _PIN_IN.search(addr)
    return m.group(1) if m else ''

def soundex(name):
    if not name:
        return ''
    name = name.upper()
    coded = name[0]
    mapping = {'B':'1','F':'1','P':'1','V':'1',
               'C':'2','G':'2','J':'2','K':'2','Q':'2','S':'2','X':'2','Z':'2',
               'D':'3','T':'3','L':'4','M':'5','N':'5','R':'6'}
    prev = mapping.get(name[0], '0')
    for c in name[1:]:
        code = mapping.get(c, '0')
        if code != '0' and code != prev:
            coded += code
        prev = code
        if len(coded) >= 4:
            break
    return (coded + '0000')[:4]

# ---- Stop words for blocking tokens ----
_STOP = frozenset({'the','a','an','of','and','in','for','to','on','at','by',
                    'de','la','le','les','du','des','et','en',
                    'private','limited','pvt','ltd','llc','inc','corp',
                    'company','co','group','services','enterprise','enterprises'})

def name_tokens(name_norm):
    """Significant tokens for blocking."""
    return [t for t in name_norm.split() if t not in _STOP and len(t) > 1]


###############################################################################
# PREPROCESSING — add all derived columns
###############################################################################
def preprocess(df):
    t0 = time.time()
    n = len(df)
    print(f"  Preprocessing {n:,} records ... ", end='', flush=True)
    
    df['name_norm'] = df['business_name'].fillna('').map(norm_name)
    df['addr_norm'] = df['business_address'].fillna('').map(norm_addr)
    df['postal'] = df.apply(lambda r: extract_postal(r['business_address'], r['country']), axis=1)
    
    # Blocking keys
    nts = df['name_norm'].map(name_tokens)
    df['name_first'] = nts.map(lambda t: t[0] if t else '')
    df['name_pre3'] = df['name_first'].str[:3]
    df['name_sx'] = df['name_first'].map(soundex)
    
    
    print(f"done in {time.time()-t0:.0f}s")
    return df


###############################################################################
# BLOCKING
###############################################################################
class Blocker:
    """Multi-key inverted-index blocker + optional TF-IDF ANN per country."""

    def __init__(self, max_block=500, tfidf_k=10):
        self.max_block = max_block
        self.tfidf_k = tfidf_k

    # ---- build phase: index S2/S3 ----
    def fit(self, s2s3):
        t0 = time.time()
        print("  Building inverted index on S2/S3 ...", flush=True)
        self.idx = defaultdict(list)
        for eid, country, postal, pre3, sx in zip(
                s2s3['entity_id'], s2s3['country'],
                s2s3['postal'], s2s3['name_pre3'], s2s3['name_sx']):
            if postal:
                self.idx[f"cp:{country}:{postal}"].append(eid)
            if pre3:
                self.idx[f"cn:{country}:{pre3}"].append(eid)
            if sx:
                self.idx[f"cs:{country}:{sx}"].append(eid)
        # prune mega-blocks  (they'd dominate runtime without adding recall)
        pruned = 0
        for k in list(self.idx):
            if len(self.idx[k]) > self.max_block:
                pruned += 1
                del self.idx[k]
        print(f"    keys: {len(self.idx):,}, pruned {pruned:,} blocks > {self.max_block}")

        # ---- TF-IDF per country ----
        print("  Building TF-IDF indices ...", flush=True)
        self.tfidf = {}
        for ctry in s2s3['country'].unique():
            mask = s2s3['country'] == ctry
            sub = s2s3.loc[mask]
            if len(sub) == 0:
                continue
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4),
                                  max_features=80_000, sublinear_tf=True, dtype=np.float32)
            mat = vec.fit_transform(sub['name_norm'].values)
            self.tfidf[ctry] = (vec, mat, sub['entity_id'].values)
            print(f"    {ctry}: {len(sub):,} records, TF-IDF shape {mat.shape}")

        print(f"  Blocker fit done in {time.time()-t0:.0f}s")

    # ---- candidate generation for all of S1 ----
    def transform(self, s1):
        t0 = time.time()
        print("  Generating candidates ...", flush=True)
        n = len(s1)
        cands = {}

        # Step 1 — inverted index
        for eid, country, postal, pre3, sx in zip(
                s1['entity_id'], s1['country'],
                s1['postal'], s1['name_pre3'], s1['name_sx']):
            hits = set()
            for key in (f"cp:{country}:{postal}" if postal else None,
                        f"cn:{country}:{pre3}" if pre3 else None,
                        f"cs:{country}:{sx}" if sx else None):
                if key and key in self.idx:
                    hits.update(self.idx[key])
            cands[eid] = hits

        # Step 2 — TF-IDF top-k per country batch
        for ctry in s1['country'].unique():
            if ctry not in self.tfidf:
                continue
            vec, mat, ids23 = self.tfidf[ctry]
            mask = s1['country'] == ctry
            sub = s1.loc[mask]
            eids = sub['entity_id'].values
            names = sub['name_norm'].values

            batch = 5000
            for start in range(0, len(sub), batch):
                end = min(start+batch, len(sub))
                q = vec.transform(names[start:end])
                sim = q @ mat.T
                for j in range(end - start):
                    row_data = sim.data[sim.indptr[j]:sim.indptr[j+1]]
                    row_indices = sim.indices[sim.indptr[j]:sim.indptr[j+1]]
                    if len(row_data) > self.tfidf_k:
                        top_k_idx = np.argpartition(row_data, -self.tfidf_k)[-self.tfidf_k:]
                        top_ids_indices = row_indices[top_k_idx]
                    else:
                        top_ids_indices = row_indices
                    for idx in top_ids_indices:
                        cands.setdefault(eids[start+j], set()).add(ids23[idx])
                if start % (batch*20) == 0 and start:
                    print(f"    TF-IDF {ctry} {start:,}/{len(sub):,}", flush=True)

        # Ensure every S1 entity has an entry
        for eid in s1['entity_id'].values:
            cands.setdefault(eid, set())

        total = sum(len(v) for v in cands.values())
        nonempty = sum(1 for v in cands.values() if v)
        elapsed = time.time() - t0
        print(f"  Candidates: {total:,} pairs, {nonempty:,}/{n:,} S1 with >=1 cand, {elapsed:.0f}s")
        return cands


###############################################################################
# FEATURE COMPUTATION  (vectorized with rapidfuzz)
###############################################################################
FEATURE_NAMES = [
    'name_exact', 'name_lev', 'name_jaro', 'name_token_sort',
    'name_jaccard', 'name_containment', 'name_3gram_jaccard',
    'name_prefix_ratio', 'name_soundex_match', 'name_len_ratio',
    'addr_lev', 'addr_jaccard', 'addr_containment',
    'addr_num_jaccard', 'postal_exact', 'postal_both_present',
    'addr_len_ratio', 'addr2_empty',
    'country_match',
    'name_addr_prod', 'name_lev_addr_lev_prod',
]

def _jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    inter = len(s1 & s2)
    return inter / (len(s1) + len(s2) - inter)

def compute_features_vec(s1_rows, s23_rows):
    """
    s1_rows, s23_rows: aligned lists of record-dicts (same length).
    Returns np.ndarray of shape (n, num_features).
    """
    n = len(s1_rows)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i in range(n):
        r1 = s1_rows[i]
        r2 = s23_rows[i]

        nn1 = r1['name_norm']
        nn2 = r2['name_norm']
        an1 = r1['addr_norm']
        an2 = r2['addr_norm']

        # Name features
        X[i, 0] = 1.0 if nn1 == nn2 else 0.0                         # name_exact
        X[i, 1] = fuzz.ratio(nn1, nn2) / 100.0                       # name_lev (normalised)
        X[i, 2] = fuzz.partial_ratio(nn1, nn2) / 100.0               # name_jaro (partial)
        X[i, 3] = fuzz.token_sort_ratio(nn1, nn2) / 100.0            # name_token_sort

        toks1 = set(nn1.split())
        toks2 = set(nn2.split())
        X[i, 4] = _jaccard(toks1, toks2)                             # name_jaccard
        X[i, 5] = len(toks1 & toks2) / max(len(toks1), 1)           # name_containment
        ng1 = frozenset(nn1[k:k+3] for k in range(max(0, len(nn1)-2)))
        ng2 = frozenset(nn2[k:k+3] for k in range(max(0, len(nn2)-2)))
        X[i, 6] = _jaccard(ng1, ng2)                                # name_3gram_jaccard

        ml = max(len(nn1), len(nn2), 1)
        pfx = 0
        for c1, c2 in zip(nn1, nn2):
            if c1 != c2: break
            pfx += 1
        X[i, 7] = pfx / ml                                           # name_prefix_ratio
        X[i, 8] = 1.0 if r1['name_sx'] and r1['name_sx'] == r2['name_sx'] else 0.0  # soundex
        X[i, 9] = min(len(nn1), len(nn2)) / max(len(nn1), len(nn2), 1)  # name_len_ratio

        # Address features
        X[i, 10] = fuzz.ratio(an1, an2) / 100.0                       # addr_lev
        atoks1 = set(an1.split())
        atoks2 = set(an2.split())
        X[i, 11] = _jaccard(atoks1, atoks2)                           # addr_jaccard
        X[i, 12] = len(atoks1 & atoks2) / max(len(atoks1), 1)        # addr_containment

        nums1 = set(re.findall(r'\d+', an1))
        nums2 = set(re.findall(r'\d+', an2))
        X[i, 13] = _jaccard(nums1, nums2)                             # addr_num_jaccard

        p1, p2 = r1['postal'], r2['postal']
        X[i, 14] = 1.0 if (p1 and p2 and p1 == p2) else 0.0          # postal_exact
        X[i, 15] = 1.0 if (p1 and p2) else 0.0                       # postal_both_present
        X[i, 16] = min(len(an1), len(an2)) / max(len(an1), len(an2), 1)  # addr_len_ratio
        X[i, 17] = 1.0 if not an2 else 0.0                            # addr2_empty

        # Country match
        X[i, 18] = 1.0 if r1['country'] == r2['country'] else 0.0

        # Interaction features
        X[i, 19] = X[i, 4] * X[i, 11]                                 # name_addr_prod
        X[i, 20] = X[i, 1] * X[i, 10]                                 # name_lev_addr_lev_prod

    return X


###############################################################################
# MACRO F_0.5 EVALUATION
###############################################################################
def macro_f05(pred_dict, gt_dict, all_s1_ids):
    scores = []
    for s1_id in all_s1_ids:
        true = gt_dict.get(s1_id, set())
        pred = pred_dict.get(s1_id, set())
        if not true and not pred:
            scores.append(1.0)
        elif not true or not pred:
            scores.append(0.0)
        else:
            tp = len(true & pred)
            p = tp / len(pred)
            r = tp / len(true)
            if p + r > 0:
                scores.append(1.25 * p * r / (0.25 * p + r))
            else:
                scores.append(0.0)
    return float(np.mean(scores))


###############################################################################
# MAIN PIPELINE
###############################################################################
def load_sources(data_dir, prefix):
    print(f"Loading {prefix} sources ...", flush=True)
    kw = dict(sep='\t', dtype=str, na_filter=False)
    s1 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source1.tsv'), **kw)
    s2 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source2.tsv'), **kw)
    s3 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source3.tsv'), **kw)
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3


def load_gt(path):
    gt = pd.read_csv(path, sep='\t', dtype=str, na_filter=False)
    d = {}
    for sid, mids in zip(gt['source1_entity_id'], gt['matched_entity_ids']):
        d[sid] = set(mids.split(',')) if mids.strip() else set()
    return d


def build_lookup(df):
    """entity_id → row-dict lookup."""
    cols = ['entity_id','name_norm','addr_norm','postal','name_sx','country',
            'business_name','business_address']
    recs = {}
    for vals in zip(*(df[c] for c in cols)):
        d = dict(zip(cols, vals))
        recs[d['entity_id']] = d
    return recs


def make_pairs_and_labels(s1_ids, cands, gt_dict, s23_lookup, neg_ratio=3, rng=None):
    """Build (s1_id, s23_id, label) triples from candidates + ground truth."""
    if rng is None:
        rng = np.random.RandomState(42)
    pairs, labels = [], []
    for s1_id in s1_ids:
        true = gt_dict.get(s1_id, set())
        c = cands.get(s1_id, set())

        # positives: true matches that exist in s23_lookup
        pos = [m for m in true if m in s23_lookup]
        for m in pos:
            pairs.append((s1_id, m))
            labels.append(1)

        # negatives: from candidates minus true
        negs = [m for m in (c - true) if m in s23_lookup]
        max_neg = max(len(pos) * neg_ratio, 3)
        if len(negs) > max_neg:
            negs = list(rng.choice(negs, max_neg, replace=False))
        for m in negs:
            pairs.append((s1_id, m))
            labels.append(0)

    return pairs, np.array(labels, dtype=np.int8)


def featurize(pairs, s1_lookup, s23_lookup, batch_size=200_000):
    """Compute feature matrix for list of (s1_id, s23_id) pairs."""
    n = len(pairs)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s1_rows = []
        s23_rows = []
        for i in range(start, end):
            s1_id, s23_id = pairs[i]
            r1 = s1_lookup.get(s1_id)
            r2 = s23_lookup.get(s23_id)
            if r1 is None or r2 is None:
                valid_mask[i] = False
                s1_rows.append({'name_norm':'','addr_norm':'','postal':'','name_sx':'',
                                'country':'','business_name':'',
                                'business_address':''})
                s23_rows.append(s1_rows[-1])
            else:
                s1_rows.append(r1)
                s23_rows.append(r2)
        X[start:end] = compute_features_vec(s1_rows, s23_rows)

        if start and start % (batch_size * 5) == 0:
            print(f"    featurize {start:,}/{n:,}", flush=True)

    return X, valid_mask


def run_train(sample_frac=1.0):
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    s1, s2, s3 = load_sources(TRAIN_DIR, 'train')
    gt_dict = load_gt(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'))

    # Optional sampling
    if sample_frac < 1.0:
        rng = np.random.RandomState(42)
        n = int(len(s1) * sample_frac)
        keep = set(rng.choice(s1['entity_id'].values, n, replace=False))
        s1 = s1[s1['entity_id'].isin(keep)].reset_index(drop=True)
        # keep S2/S3 that are ground-truth matches, plus random extras
        gt_ids = set()
        for sid in keep:
            gt_ids |= gt_dict.get(sid, set())
        extra2 = set(rng.choice(s2['entity_id'].values, min(len(s2), n*3), replace=False))
        extra3 = set(rng.choice(s3['entity_id'].values, min(len(s3), n*3), replace=False))
        s2 = s2[s2['entity_id'].isin(gt_ids | extra2)].reset_index(drop=True)
        s3 = s3[s3['entity_id'].isin(gt_ids | extra3)].reset_index(drop=True)
        gt_dict = {k: v for k, v in gt_dict.items() if k in keep}
        print(f"  Sampled: S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    # Preprocess
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    s3 = preprocess(s3)
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s23):,}")

    # Build lookups
    s1_lookup = build_lookup(s1)
    s23_lookup = build_lookup(s23)

    # Train/val split (85/15 on S1 entity IDs)
    rng = np.random.RandomState(42)
    all_s1 = s1['entity_id'].values.copy()
    rng.shuffle(all_s1)
    split = int(len(all_s1) * 0.85)
    train_ids = set(all_s1[:split])
    val_ids   = set(all_s1[split:])
    print(f"  Train S1: {len(train_ids):,}  Val S1: {len(val_ids):,}")

    # Blocking (on all S1 for recall measurement)
    blocker = Blocker(max_block=500, tfidf_k=10)
    blocker.fit(s23)
    cands = blocker.transform(s1)

    # Blocking recall on validation
    hits = total = 0
    for sid in val_ids:
        for m in gt_dict.get(sid, set()):
            total += 1
            if m in cands.get(sid, set()):
                hits += 1
    blocking_recall = hits / max(total, 1)
    total_cand_pairs = sum(len(v) for v in cands.values())
    max_possible = len(s1) * len(s23)
    reduction = 1.0 - total_cand_pairs / max_possible
    print(f"\n  Blocking recall (val): {blocking_recall:.4f}  ({hits:,}/{total:,})")
    print(f"  Reduction ratio: {reduction:.8f}")
    print(f"  Candidate pairs: {total_cand_pairs:,}")

    # Build training + validation pairs
    print("\nBuilding train/val pairs ...")
    tr_pairs, y_tr = make_pairs_and_labels(list(train_ids), cands, gt_dict, s23_lookup, neg_ratio=3)
    vl_pairs, y_vl = make_pairs_and_labels(list(val_ids),   cands, gt_dict, s23_lookup, neg_ratio=3)
    print(f"  Train: {len(tr_pairs):,} pairs  (pos={y_tr.sum():,}, neg={len(y_tr)-y_tr.sum():,})")
    print(f"  Val:   {len(vl_pairs):,} pairs  (pos={y_vl.sum():,}, neg={len(y_vl)-y_vl.sum():,})")

    # Featurize
    print("\nFeaturizing training pairs ...")
    X_tr, m_tr = featurize(tr_pairs, s1_lookup, s23_lookup)
    X_tr = X_tr[m_tr]; y_tr = y_tr[m_tr]; tr_pairs = [p for p, ok in zip(tr_pairs, m_tr) if ok]

    print("Featurizing validation pairs ...")
    X_vl, m_vl = featurize(vl_pairs, s1_lookup, s23_lookup)
    X_vl = X_vl[m_vl]; y_vl = y_vl[m_vl]; vl_pairs = [p for p, ok in zip(vl_pairs, m_vl) if ok]

    # Train LightGBM
    print("\nTraining LightGBM ...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES, free_raw_data=False)
    dval   = lgb.Dataset(X_vl, label=y_vl, feature_name=FEATURE_NAMES, free_raw_data=False)

    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 100, 'verbosity': -1, 'n_jobs': -1,
        'scale_pos_weight': float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1),
    }
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])

    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    order = np.argsort(imp)[::-1]
    print("\nFeature importance (gain):")
    for idx in order[:15]:
        print(f"  {FEATURE_NAMES[idx]:30s}  {imp[idx]:.0f}")

    # ---- Threshold tuning on ALL validation candidates ----
    print("\nScoring all validation candidates for threshold tuning ...")
    val_all_pairs = []
    for sid in val_ids:
        for c in cands.get(sid, set()):
            if c in s23_lookup:
                val_all_pairs.append((sid, c))
    print(f"  Val candidate pairs: {len(val_all_pairs):,}")

    X_vc, m_vc = featurize(val_all_pairs, s1_lookup, s23_lookup)
    X_vc = X_vc[m_vc]
    val_all_pairs = [p for p, ok in zip(val_all_pairs, m_vc) if ok]
    probs = model.predict(X_vc)

    best_f05, best_thr = 0, 0.5
    for thr in np.arange(0.10, 0.96, 0.02):
        pred_d = defaultdict(set)
        for sid in val_ids:
            pred_d[sid] = set()
        for i, (sid, cid) in enumerate(val_all_pairs):
            if probs[i] >= thr:
                pred_d[sid].add(cid)
        f = macro_f05(pred_d, gt_dict, list(val_ids))
        if f > best_f05:
            best_f05, best_thr = f, thr

    print(f"\n  Best threshold: {best_thr:.2f}")
    print(f"  Validation F_0.5:  {best_f05:.4f}")

    # Save
    artefact = {'model': model, 'threshold': best_thr, 'features': FEATURE_NAMES,
                'blocker_cfg': {'max_block': blocker.max_block, 'tfidf_k': blocker.tfidf_k}}
    path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {path}")

    return dict(blocking_recall=blocking_recall, reduction_ratio=reduction,
                val_f05=best_f05, threshold=best_thr)


def run_test(sample_frac=1.0):
    print("=" * 80)
    print("TEST INFERENCE")
    print("=" * 80)

    s1, s2, s3 = load_sources(TEST_DIR, 'test')

    if sample_frac < 1.0:
        rng = np.random.RandomState(42)
        n = int(len(s1) * sample_frac)
        keep = set(rng.choice(s1['entity_id'].values, n, replace=False))
        s1 = s1[s1['entity_id'].isin(keep)].reset_index(drop=True)
        # No ground truth on test -- just take a matching random slice of S2/S3
        # so blocking/TF-IDF build time shrinks too, not just S1.
        n2 = min(len(s2), max(n * 5, 500))
        n3 = min(len(s3), max(n * 5, 500))
        s2 = s2[s2['entity_id'].isin(rng.choice(s2['entity_id'].values, n2, replace=False))].reset_index(drop=True)
        s3 = s3[s3['entity_id'].isin(rng.choice(s3['entity_id'].values, n3, replace=False))].reset_index(drop=True)
        print(f"  [TEST SAMPLE MODE] S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
        print("  NOTE: output from this mode is NOT a valid submission "
              "(missing S1 entities) -- smoke-test only.")

    s1 = preprocess(s1)
    s2 = preprocess(s2)
    s3 = preprocess(s3)
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s23):,}")

    # Load model
    path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(path, 'rb') as f:
        artefact = pickle.load(f)
    model = artefact['model']
    thr   = artefact['threshold']
    print(f"  Model loaded, threshold={thr:.2f}")

    # Lookups
    s1_lookup  = build_lookup(s1)
    s23_lookup = build_lookup(s23)

    # Blocking
    cfg = artefact['blocker_cfg']
    blocker = Blocker(**cfg)
    blocker.fit(s23)
    cands = blocker.transform(s1)

    # Score
    print("\nScoring candidate pairs ...")
    all_pairs = []
    for sid in s1['entity_id'].values:
        for c in cands.get(sid, set()):
            all_pairs.append((sid, c))
    print(f"  Total pairs: {len(all_pairs):,}")

    matches = defaultdict(set)
    batch = 500_000
    for start in range(0, len(all_pairs), batch):
        end = min(start + batch, len(all_pairs))
        chunk = all_pairs[start:end]
        X, mask = featurize(chunk, s1_lookup, s23_lookup)
        X = X[mask]
        valid_chunk = [p for p, ok in zip(chunk, mask) if ok]
        if len(X):
            p = model.predict(X)
            for i, (sid, cid) in enumerate(valid_chunk):
                if p[i] >= thr:
                    matches[sid].add(cid)
        if start % (batch * 3) == 0 and start:
            print(f"    {start:,}/{len(all_pairs):,}", flush=True)

    # Write outputs
    print("\nWriting outputs ...")
    mp = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    cp = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')

    with open(mp, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for sid in s1['entity_id'].values:
            m = ','.join(sorted(matches.get(sid, set())))
            f.write(f'{sid}\t{m}\n')

    with open(cp, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for sid in s1['entity_id'].values:
            c = ','.join(sorted(cands.get(sid, set())))
            f.write(f'{sid}\t{c}\n')

    n_matched = sum(1 for v in matches.values() if v)
    n_links   = sum(len(v) for v in matches.values())
    print(f"  Matched S1: {n_matched:,}  Links: {n_links:,}  Singletons: {len(s1)-n_matched:,}")
    print(f"  Written: {mp}")
    print(f"  Written: {cp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train','test','full'], default='full')
    ap.add_argument('--sample', type=float, default=1.0,
                    help='Fraction of training data (for fast dev)')
    ap.add_argument('--test-sample', type=float, default=None,
                    help='Fraction of test data (for fast dev). '
                         'Defaults to --sample if not given. '
                         'NOTE: output is NOT a valid submission when < 1.0.')
    args = ap.parse_args()
    test_sample = args.test_sample if args.test_sample is not None else args.sample

    if args.mode in ('train', 'full'):
        res = run_train(args.sample)
        print(f"\n{'='*80}\nTRAINING RESULTS\n{'='*80}")
        for k, v in res.items():
            print(f"  {k}: {v}")

    if args.mode in ('test', 'full'):
        run_test(test_sample)


if __name__ == '__main__':
    main()
