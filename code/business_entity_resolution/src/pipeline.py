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
_ZIP_FR = re.compile(r'\b(\d{5})\b')

def extract_postal(addr, country=''):
    if not isinstance(addr, str):
        return ''
    country_l = country.lower() if isinstance(country, str) else ''
    if country_l == 'india':
        m = _PIN_IN.search(addr)
        return m.group(1) if m else ''
    if country_l == 'france':
        m = _ZIP_FR.search(addr)
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
                    'company','co','group','services','enterprise','enterprises',
                    'near','opposite','behind','next','nagar','road','street',
                    'avenue','floor','apartment','suite','building','tower',
                    'block','sector','phase','plot','no'})

def name_tokens(name_norm):
    """Significant tokens for blocking."""
    return [t for t in name_norm.split() if t not in _STOP and len(t) > 1]


def extract_addr_numbers(addr_norm):
    """Extract numeric tokens from normalized address."""
    return set(re.findall(r'\d+', addr_norm))

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
    
    # Blocking keys — ALL significant tokens, not just the first
    nts = df['name_norm'].map(name_tokens)
    df['name_toks'] = nts  # list of significant tokens
    df['name_first'] = nts.map(lambda t: t[0] if t else '')
    df['name_pre3'] = df['name_first'].str[:3]
    df['name_sx'] = df['name_first'].map(soundex)
    
    # All-token soundex codes for multi-token phonetic blocking
    df['all_sx'] = nts.map(lambda toks: list(set(soundex(t) for t in toks if t)))
    
    # Address numbers for address-based blocking
    df['addr_nums'] = df['addr_norm'].map(extract_addr_numbers)
    
    # Country normalization (handle potential inconsistencies)
    df['country_norm'] = df['country'].fillna('').str.strip().str.lower()
    
    print(f"done in {time.time()-t0:.0f}s")
    return df


###############################################################################
# BLOCKING
###############################################################################
class Blocker:
    """Multi-key inverted-index blocker + TF-IDF ANN.
    
    Key improvements over original:
    - Multi-token blocking (all significant tokens, not just first)
    - Country-agnostic keys alongside country-scoped ones
    - Phonetic codes on all tokens
    - Address number blocking
    - Higher max_block tolerance
    - Higher TF-IDF k
    """

    MAX_CANDS_PER_ENTITY = 500  # cap candidates per S1 entity for runtime

    def __init__(self, max_block=2000, tfidf_k=20):
        self.max_block = max_block
        self.tfidf_k = tfidf_k

    # ---- build phase: index S2/S3 ----
    def fit(self, s2s3):
        t0 = time.time()
        print("  Building inverted index on S2/S3 ...", flush=True)
        self.idx = defaultdict(list)
        # Store name lookup for candidate cap pre-filtering
        self._s23_names = dict(zip(s2s3['entity_id'], s2s3['name_norm']))
        
        for eid, country, postal, toks, all_sx, addr_nums in zip(
                s2s3['entity_id'], s2s3['country_norm'],
                s2s3['postal'], s2s3['name_toks'],
                s2s3['all_sx'], s2s3['addr_nums']):
            
            # 1. Postal code blocking (country-scoped AND country-agnostic)
            if postal:
                self.idx[f"p:{postal}"].append(eid)
            
            # 2. Name token blocking — every significant token as a key
            #    (country-scoped to keep block sizes manageable for common tokens)
            for tok in toks:
                if len(tok) >= 3:
                    self.idx[f"nt:{country}:{tok}"].append(eid)
                # Country-agnostic prefix (4-char to limit block sizes at scale)
                if len(tok) >= 4:
                    self.idx[f"np:{tok[:4]}"].append(eid)
            
            # 3. Soundex blocking on all tokens (country-scoped)
            for sx in all_sx:
                if sx:
                    self.idx[f"sx:{country}:{sx}"].append(eid)
            
            # 4. Address number blocking — specific street/building numbers
            #    (only for "distinctive" numbers, skip very short ones like "1", "2")
            for num in addr_nums:
                if len(num) >= 3:  # 3+ digit numbers are more distinctive
                    self.idx[f"an:{country}:{num}"].append(eid)
        
        # Prune mega-blocks (they dominate runtime without adding recall)
        pruned = 0
        for k in list(self.idx):
            if len(self.idx[k]) > self.max_block:
                pruned += 1
                del self.idx[k]
        print(f"    keys: {len(self.idx):,}, pruned {pruned:,} blocks > {self.max_block}")

        # ---- TF-IDF per country (for name similarity) ----
        print("  Building TF-IDF indices ...", flush=True)
        self.tfidf = {}
        for ctry in s2s3['country_norm'].unique():
            mask = s2s3['country_norm'] == ctry
            sub = s2s3.loc[mask]
            if len(sub) == 0:
                continue
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4),
                                  max_features=100_000, sublinear_tf=True, dtype=np.float32)
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

        # Step 1 — inverted index lookups
        for eid, country, postal, toks, all_sx, addr_nums in zip(
                s1['entity_id'], s1['country_norm'],
                s1['postal'], s1['name_toks'],
                s1['all_sx'], s1['addr_nums']):
            hits = set()
            
            # Postal code lookup (country-agnostic)
            if postal:
                key = f"p:{postal}"
                if key in self.idx:
                    hits.update(self.idx[key])
            
            # Name token lookup — all significant tokens
            for tok in toks:
                if len(tok) >= 3:
                    key = f"nt:{country}:{tok}"
                    if key in self.idx:
                        hits.update(self.idx[key])
                # Country-agnostic prefix fallback (4-char)
                if len(tok) >= 4:
                    key = f"np:{tok[:4]}"
                    if key in self.idx:
                        hits.update(self.idx[key])
            
            # Soundex lookup on all tokens
            for sx in all_sx:
                if sx:
                    key = f"sx:{country}:{sx}"
                    if key in self.idx:
                        hits.update(self.idx[key])
            
            # Address number lookup
            for num in addr_nums:
                if len(num) >= 3:
                    key = f"an:{country}:{num}"
                    if key in self.idx:
                        hits.update(self.idx[key])
            
            cands[eid] = hits

        # Step 2 — TF-IDF top-k per country batch
        for ctry in s1['country_norm'].unique():
            if ctry not in self.tfidf:
                continue
            vec, mat, ids23 = self.tfidf[ctry]
            mask = s1['country_norm'] == ctry
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

        # Per-entity candidate cap: if too many candidates, keep the most
        # promising ones using a cheap character-overlap pre-filter (much
        # faster than fuzz.ratio for thousands of candidates).
        s1_names = dict(zip(s1['entity_id'], s1['name_norm']))
        capped = 0
        for eid in list(cands):
            if len(cands[eid]) > self.MAX_CANDS_PER_ENTITY:
                capped += 1
                nn1 = s1_names.get(eid, '')
                nn1_set = set(nn1.split())
                scored = []
                for cid in cands[eid]:
                    nn2 = self._s23_names.get(cid, '')
                    nn2_set = set(nn2.split())
                    # Token overlap score (cheap approximation of Jaccard)
                    inter = len(nn1_set & nn2_set)
                    union = len(nn1_set | nn2_set)
                    score = inter / max(union, 1)
                    scored.append((score, cid))
                scored.sort(reverse=True)
                cands[eid] = set(cid for _, cid in scored[:self.MAX_CANDS_PER_ENTITY])
        if capped:
            print(f"    Capped {capped:,} entities to {self.MAX_CANDS_PER_ENTITY} candidates")

        # Ensure every S1 entity has an entry
        for eid in s1['entity_id'].values:
            cands.setdefault(eid, set())

        total = sum(len(v) for v in cands.values())
        nonempty = sum(1 for v in cands.values() if v)
        elapsed = time.time() - t0
        avg_cands = total / max(n, 1)
        print(f"  Candidates: {total:,} pairs, {nonempty:,}/{n:,} S1 with >=1 cand, avg={avg_cands:.1f}/entity, {elapsed:.0f}s")
        return cands


###############################################################################
# FEATURE COMPUTATION
###############################################################################
FEATURE_NAMES = [
    # Name similarity features (0-13)
    'name_exact',             # 0:  exact match after normalization
    'name_lev',               # 1:  fuzz.ratio (normalized levenshtein)
    'name_partial',           # 2:  fuzz.partial_ratio
    'name_token_sort',        # 3:  fuzz.token_sort_ratio
    'name_token_set',         # 4:  fuzz.token_set_ratio
    'name_jaccard',           # 5:  token jaccard
    'name_containment_s1',    # 6:  |intersection|/|s1_tokens| — s1 tokens covered
    'name_containment_s23',   # 7:  |intersection|/|s23_tokens| — s23 tokens covered
    'name_3gram_jaccard',     # 8:  character 3-gram jaccard
    'name_prefix_ratio',      # 9:  common prefix length / max length
    'name_soundex_match',     # 10: soundex of first token matches
    'name_len_ratio',         # 11: min(len)/max(len)
    'name_first_exact',       # 12: first significant token exact match
    'name_len_diff',          # 13: abs length difference (raw)
    
    # Address similarity features (14-24)
    'addr_lev',               # 14: fuzz.ratio on address
    'addr_token_sort',        # 15: fuzz.token_sort_ratio on address
    'addr_jaccard',           # 16: token jaccard on address
    'addr_containment',       # 17: token containment on address
    'addr_num_jaccard',       # 18: numeric token jaccard
    'addr_num_overlap',       # 19: count of overlapping numeric tokens
    'postal_exact',           # 20: postal code exact match
    'postal_both_present',    # 21: both have postal codes
    'postal_prefix3',         # 22: first 3 digits of postal match
    'addr_len_ratio',         # 23: address length ratio
    'addr2_empty',            # 24: s23 address is empty
    
    # Cross-field features (25-29)
    'country_match',          # 25: country exact match
    'name_addr_prod',         # 26: name_jaccard * addr_jaccard
    'name_lev_addr_lev_prod', # 27: name_lev * addr_lev
    'name_token_set_addr',    # 28: name_token_set * addr_lev
    'max_name_sim',           # 29: max of name_lev, name_token_sort, name_token_set
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
    nf = len(FEATURE_NAMES)
    X = np.zeros((n, nf), dtype=np.float32)

    for i in range(n):
        r1 = s1_rows[i]
        r2 = s23_rows[i]

        nn1 = r1['name_norm']
        nn2 = r2['name_norm']
        an1 = r1['addr_norm']
        an2 = r2['addr_norm']

        # ----- Name features -----
        X[i, 0] = 1.0 if nn1 and nn2 and nn1 == nn2 else 0.0            # name_exact
        X[i, 1] = fuzz.ratio(nn1, nn2) / 100.0                           # name_lev
        X[i, 2] = fuzz.partial_ratio(nn1, nn2) / 100.0                   # name_partial
        X[i, 3] = fuzz.token_sort_ratio(nn1, nn2) / 100.0                # name_token_sort
        X[i, 4] = fuzz.token_set_ratio(nn1, nn2) / 100.0                 # name_token_set

        toks1 = set(nn1.split()) if nn1 else set()
        toks2 = set(nn2.split()) if nn2 else set()
        X[i, 5] = _jaccard(toks1, toks2)                                 # name_jaccard
        inter = len(toks1 & toks2) if toks1 and toks2 else 0
        X[i, 6] = inter / max(len(toks1), 1)                             # name_containment_s1
        X[i, 7] = inter / max(len(toks2), 1)                             # name_containment_s23

        ng1 = frozenset(nn1[k:k+3] for k in range(max(0, len(nn1)-2)))
        ng2 = frozenset(nn2[k:k+3] for k in range(max(0, len(nn2)-2)))
        X[i, 8] = _jaccard(ng1, ng2)                                     # name_3gram_jaccard

        ml = max(len(nn1), len(nn2), 1)
        pfx = 0
        for c1, c2 in zip(nn1, nn2):
            if c1 != c2: break
            pfx += 1
        X[i, 9] = pfx / ml                                               # name_prefix_ratio
        X[i, 10] = 1.0 if r1.get('name_sx') and r1['name_sx'] == r2.get('name_sx','') else 0.0  # soundex

        ln1, ln2 = len(nn1), len(nn2)
        X[i, 11] = min(ln1, ln2) / max(ln1, ln2, 1)                      # name_len_ratio

        # First significant token exact match
        ft1 = r1.get('name_first', '')
        ft2 = r2.get('name_first', '')
        X[i, 12] = 1.0 if ft1 and ft2 and ft1 == ft2 else 0.0           # name_first_exact
        X[i, 13] = abs(ln1 - ln2)                                        # name_len_diff

        # ----- Address features -----
        X[i, 14] = fuzz.ratio(an1, an2) / 100.0                          # addr_lev
        X[i, 15] = fuzz.token_sort_ratio(an1, an2) / 100.0               # addr_token_sort

        atoks1 = set(an1.split()) if an1 else set()
        atoks2 = set(an2.split()) if an2 else set()
        X[i, 16] = _jaccard(atoks1, atoks2)                              # addr_jaccard
        X[i, 17] = len(atoks1 & atoks2) / max(len(atoks1), 1)            # addr_containment

        nums1 = set(re.findall(r'\d+', an1))
        nums2 = set(re.findall(r'\d+', an2))
        X[i, 18] = _jaccard(nums1, nums2)                                # addr_num_jaccard
        X[i, 19] = len(nums1 & nums2) if nums1 and nums2 else 0          # addr_num_overlap

        p1, p2 = r1.get('postal', ''), r2.get('postal', '')
        X[i, 20] = 1.0 if (p1 and p2 and p1 == p2) else 0.0             # postal_exact
        X[i, 21] = 1.0 if (p1 and p2) else 0.0                           # postal_both_present
        X[i, 22] = 1.0 if (p1 and p2 and len(p1)>=3 and len(p2)>=3
                           and p1[:3] == p2[:3]) else 0.0                 # postal_prefix3

        la1, la2 = len(an1), len(an2)
        X[i, 23] = min(la1, la2) / max(la1, la2, 1)                      # addr_len_ratio
        X[i, 24] = 1.0 if not an2 else 0.0                               # addr2_empty

        # ----- Cross-field features -----
        X[i, 25] = 1.0 if r1.get('country_norm','') == r2.get('country_norm','') else 0.0
        X[i, 26] = X[i, 5] * X[i, 16]                                    # name_addr_prod
        X[i, 27] = X[i, 1] * X[i, 14]                                    # name_lev_addr_lev_prod
        X[i, 28] = X[i, 4] * X[i, 14]                                    # name_token_set * addr_lev
        X[i, 29] = max(X[i, 1], X[i, 3], X[i, 4])                        # max_name_sim

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
            'country_norm','name_first',
            'business_name','business_address']
    recs = {}
    for vals in zip(*(df[c] for c in cols)):
        d = dict(zip(cols, vals))
        recs[d['entity_id']] = d
    return recs


def make_pairs_and_labels(s1_ids, cands, gt_dict, s23_lookup, neg_ratio=5, rng=None):
    """Build (s1_id, s23_id, label) triples from candidates + ground truth.
    
    Includes ground-truth positives even when missed by blocking, so the
    model sees hard positives during training.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    pairs, labels = [], []
    for s1_id in s1_ids:
        true = gt_dict.get(s1_id, set())
        c = cands.get(s1_id, set())

        # Positives: ALL true matches that exist in s23_lookup
        # (includes those missed by blocking — critical for learning hard cases)
        pos = [m for m in true if m in s23_lookup]
        for m in pos:
            pairs.append((s1_id, m))
            labels.append(1)

        # Negatives: from candidates minus true
        negs = [m for m in (c - true) if m in s23_lookup]
        max_neg = max(len(pos) * neg_ratio, 5)
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

    _empty = {'name_norm':'','addr_norm':'','postal':'','name_sx':'',
              'country':'','country_norm':'','name_first':'',
              'business_name':'','business_address':''}

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
                s1_rows.append(_empty)
                s23_rows.append(_empty)
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
    blocker = Blocker(max_block=2000, tfidf_k=20)
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
    tr_pairs, y_tr = make_pairs_and_labels(list(train_ids), cands, gt_dict, s23_lookup, neg_ratio=5)
    vl_pairs, y_vl = make_pairs_and_labels(list(val_ids),   cands, gt_dict, s23_lookup, neg_ratio=5)
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
        'num_leaves': 127, 'learning_rate': 0.03,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 50, 'verbosity': -1, 'n_jobs': -1,
        'max_depth': -1,
        'lambda_l1': 0.1, 'lambda_l2': 1.0,
        'scale_pos_weight': float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1),
    }
    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    order = np.argsort(imp)[::-1]
    print("\nFeature importance (gain):")
    for idx in order[:20]:
        print(f"  {FEATURE_NAMES[idx]:30s}  {imp[idx]:.0f}")

    # ---- Threshold tuning on ALL validation candidates ----
    print("\nScoring all validation candidates for threshold tuning ...")
    val_all_pairs = []
    for sid in val_ids:
        for c in cands.get(sid, set()):
            if c in s23_lookup:
                val_all_pairs.append((sid, c))
    
    # Also include GT positives missed by blocking for comprehensive eval
    val_gt_extra = []
    for sid in val_ids:
        for m in gt_dict.get(sid, set()):
            if m in s23_lookup and m not in cands.get(sid, set()):
                val_gt_extra.append((sid, m))
    
    print(f"  Val candidate pairs: {len(val_all_pairs):,}")
    print(f"  Val GT pairs missed by blocking: {len(val_gt_extra):,}")

    X_vc, m_vc = featurize(val_all_pairs, s1_lookup, s23_lookup)
    X_vc = X_vc[m_vc]
    val_all_pairs = [p for p, ok in zip(val_all_pairs, m_vc) if ok]
    probs = model.predict(X_vc)

    # Fine-grained threshold search
    best_f05, best_thr = 0, 0.5
    for thr in np.arange(0.05, 0.98, 0.01):
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
