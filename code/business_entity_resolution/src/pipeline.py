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
import gc
import argparse
import pickle
import struct
import tempfile
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
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


def _drop_blocker_cols(df):
    """Drop columns only needed during Blocker.fit/transform to free RAM.

    These columns hold Python list/set objects per row (~80 bytes each overhead)
    and are not needed after the blocker has been built.
    Dropping on s23 (10M rows) saves ~2-3 GB.
    """
    to_drop = [c for c in ('name_toks', 'all_sx', 'addr_nums', 'name_pre3') if c in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)


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

    def __init__(self, max_block=2000, tfidf_k=50):
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
                # Country-agnostic full token (5-char for high precision)
                if len(tok) >= 5:
                    self.idx[f"nt_global:{tok}"].append(eid)
            
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
        import random
        rng = random.Random(42)
        pruned = 0
        for k in list(self.idx):
            if len(self.idx[k]) > self.max_block:
                pruned += 1
                self.idx[k] = rng.sample(self.idx[k], self.max_block)
        print(f"    keys: {len(self.idx):,}, sampled {pruned:,} blocks > {self.max_block}")

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
                # Country-agnostic full token (5-char for high precision)
                if len(tok) >= 5:
                    key = f"nt_global:{tok}"
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
    'name_jaro_winkler',      # 30: Jaro-Winkler on name
    'addr_jaro_winkler',      # 31: Jaro-Winkler on address
    'is_source2',             # 32: 1 if candidate is from S2, else 0
]

# Slot indices for lookup tuples:
#   (name_norm, addr_norm, postal, name_sx, country_norm, name_first)
_LU_NAME_NORM    = 0
_LU_ADDR_NORM    = 1
_LU_POSTAL       = 2
_LU_NAME_SX      = 3
_LU_COUNTRY_NORM = 4
_LU_NAME_FIRST   = 5

def _jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    inter = len(s1 & s2)
    return inter / (len(s1) + len(s2) - inter)

def compute_features_vec(s1_rows, s23_rows, s23_ids):
    """
    s1_rows, s23_rows: aligned lists of 6-tuples (same length).
    s23_ids: list of s23_ids for the is_source2 feature.
    Each tuple: (name_norm, addr_norm, postal, name_sx, country_norm, name_first)
    Returns np.ndarray of shape (n, num_features).
    """
    n = len(s1_rows)
    nf = len(FEATURE_NAMES)
    X = np.zeros((n, nf), dtype=np.float32)

    for i in range(n):
        r1 = s1_rows[i]
        r2 = s23_rows[i]

        nn1 = r1[_LU_NAME_NORM]
        nn2 = r2[_LU_NAME_NORM]
        an1 = r1[_LU_ADDR_NORM]
        an2 = r2[_LU_ADDR_NORM]

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

        sx1 = r1[_LU_NAME_SX]
        sx2 = r2[_LU_NAME_SX]
        X[i, 10] = 1.0 if sx1 and sx1 == sx2 else 0.0                   # soundex

        ln1, ln2 = len(nn1), len(nn2)
        X[i, 11] = min(ln1, ln2) / max(ln1, ln2, 1)                      # name_len_ratio

        # First significant token exact match
        ft1 = r1[_LU_NAME_FIRST]
        ft2 = r2[_LU_NAME_FIRST]
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

        p1, p2 = r1[_LU_POSTAL], r2[_LU_POSTAL]
        X[i, 20] = 1.0 if (p1 and p2 and p1 == p2) else 0.0             # postal_exact
        X[i, 21] = 1.0 if (p1 and p2) else 0.0                           # postal_both_present
        X[i, 22] = 1.0 if (p1 and p2 and len(p1)>=3 and len(p2)>=3
                           and p1[:3] == p2[:3]) else 0.0                 # postal_prefix3

        la1, la2 = len(an1), len(an2)
        X[i, 23] = min(la1, la2) / max(la1, la2, 1)                      # addr_len_ratio
        X[i, 24] = 1.0 if not an2 else 0.0                               # addr2_empty

        # ----- Cross-field features -----
        c1 = r1[_LU_COUNTRY_NORM]
        c2 = r2[_LU_COUNTRY_NORM]
        X[i, 25] = 1.0 if c1 == c2 else 0.0
        X[i, 26] = X[i, 5] * X[i, 16]                                    # name_addr_prod
        X[i, 27] = X[i, 1] * X[i, 14]                                    # name_lev_addr_lev_prod
        X[i, 28] = X[i, 4] * X[i, 14]                                    # name_token_set * addr_lev
        X[i, 29] = max(X[i, 1], X[i, 3], X[i, 4])                        # max_name_sim
        X[i, 30] = JaroWinkler.similarity(nn1, nn2) if nn1 and nn2 else 0.0
        X[i, 31] = JaroWinkler.similarity(an1, an2) if an1 and an2 else 0.0
        X[i, 32] = 1.0 if s23_ids[i].startswith('S2-') else 0.0

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
# DATA LOADING
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


###############################################################################
# LOOKUP BUILDING
###############################################################################
def build_lookup(df):
    """entity_id -> 6-tuple lookup.

    Stores ONLY the 6 fields actually consumed by compute_features_vec:
      (name_norm, addr_norm, postal, name_sx, country_norm, name_first)

    Using a tuple instead of a dict per record reduces CPython overhead
    from ~900 bytes/record to ~104 bytes/record (excluding string data),
    which saves ~8 GB at 10 M S2/S3 records compared to dict-of-dicts.
    business_name and business_address are NOT stored — they are never
    read by any feature computation.
    """
    cols = ['entity_id', 'name_norm', 'addr_norm', 'postal',
            'name_sx', 'country_norm', 'name_first']
    recs = {}
    for vals in zip(*(df[c] for c in cols)):
        # vals[0] = entity_id; vals[1:7] = the 6 feature fields as a tuple
        recs[vals[0]] = vals[1:]
    return recs


###############################################################################
# DISK-BACKED PAIR STORE
###############################################################################
# At full scale the accumulated Python lists for tr_pairs, vl_pairs, and
# val_all_pairs can exceed 4-5 GB combined. _DiskPairStore writes each
# (s1_id, s23_id) pair to a temp binary file as length-prefixed bytes and
# reads them back on demand, keeping only one batch in RAM at a time.

_PAIR_STRUCT = struct.Struct('>HH')   # two unsigned shorts (len_s1_id, len_s23_id)

class _DiskPairStore:
    """Write (s1_id, s23_id) string pairs to disk; stream them back."""

    def __init__(self):
        self._fh = tempfile.TemporaryFile()
        self._count = 0

    def append(self, s1_id, s23_id):
        b1 = s1_id.encode()
        b2 = s23_id.encode()
        self._fh.write(_PAIR_STRUCT.pack(len(b1), len(b2)))
        self._fh.write(b1)
        self._fh.write(b2)
        self._count += 1

    def extend_pairs(self, pair_list):
        for s1_id, s23_id in pair_list:
            self.append(s1_id, s23_id)

    def __len__(self):
        return self._count

    def iterate(self, batch_size=200_000):
        """Yield batches of [(s1_id, s23_id), ...] by seeking to 0 first."""
        self._fh.seek(0)
        hdr_size = _PAIR_STRUCT.size
        batch = []
        while True:
            hdr = self._fh.read(hdr_size)
            if len(hdr) < hdr_size:
                break
            l1, l2 = _PAIR_STRUCT.unpack(hdr)
            b1 = self._fh.read(l1)
            b2 = self._fh.read(l2)
            batch.append((b1.decode(), b2.decode()))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def close(self):
        self._fh.close()


###############################################################################
# FEATURIZATION HELPERS
###############################################################################
def featurize(pairs, s1_lookup, s23_lookup, batch_size=200_000):
    """Compute feature matrix for a list of (s1_id, s23_id) pairs (in RAM).

    Used only for moderate-sized pair sets in test inference where a chunk
    fits comfortably in memory.
    Returns (X, valid_mask).
    """
    n = len(pairs)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    _empty = ('', '', '', '', '', '')

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
        s23_ids_batch = [p[1] for p in pairs[start:end]]
        X[start:end] = compute_features_vec(s1_rows, s23_rows, s23_ids_batch)

        if start and start % (batch_size * 5) == 0:
            print(f"    featurize {start:,}/{n:,}", flush=True)

    return X, valid_mask


def _featurize_store_to_memmap(pair_store, s1_lookup, s23_lookup,
                                mm_path, batch_size=200_000):
    """Stream pairs from a _DiskPairStore, featurize in batches, write to memmap.

    Returns (mm, valid_mask) where mm is a numpy memmap (stays on disk) and
    valid_mask is a bool array in RAM (~1 bit per pair, negligible).
    This avoids materializing the full X matrix in RAM during featurization.
    """
    total_pairs = len(pair_store)
    nf = len(FEATURE_NAMES)
    mm = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(total_pairs, nf))
    valid_mask = np.ones(total_pairs, dtype=bool)
    _empty = ('', '', '', '', '', '')

    offset = 0
    for batch in pair_store.iterate(batch_size):
        blen = len(batch)
        s1_rows, s23_rows = [], []
        s23_ids_batch = []
        for k, (s1_id, s23_id) in enumerate(batch):
            r1 = s1_lookup.get(s1_id)
            r2 = s23_lookup.get(s23_id)
            s23_ids_batch.append(s23_id)
            if r1 is None or r2 is None:
                valid_mask[offset + k] = False
                s1_rows.append(_empty)
                s23_rows.append(_empty)
            else:
                s1_rows.append(r1)
                s23_rows.append(r2)
        mm[offset:offset + blen] = compute_features_vec(s1_rows, s23_rows, s23_ids_batch)
        offset += blen
        if offset % (batch_size * 5) == 0 and offset > 0:
            print(f"    featurize (memmap) {offset:,}/{total_pairs:,}", flush=True)

    mm.flush()
    return mm, valid_mask


###############################################################################
# PAIR + LABEL GENERATION
###############################################################################
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


###############################################################################
# TRAINING
###############################################################################
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

    # Preprocess all three sources
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    s3 = preprocess(s3)

    # --- RAM fix A: concat s2+s3 then immediately free s2 and s3 ---
    # pd.concat keeps s2 and s3 alive until explicitly deleted. At 10M rows
    # this triples the RAM needed for source data. Freeing them right after
    # concat saves ~2-3 GB.
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    print(f"  Combined S2+S3: {len(s23):,} (s2/s3 freed)")

    # --- Blocker build BEFORE dropping Python-object columns ---
    # blocker.fit() is the only consumer of name_toks, all_sx, addr_nums.
    blocker = Blocker(max_block=2000, tfidf_k=50)
    blocker.fit(s23)

    # --- RAM fix B: drop Python list/set columns from s23 after blocker.fit() ---
    # name_toks, all_sx, addr_nums are Python-object columns: each cell is a
    # CPython list/set object (~80 bytes overhead + 8 bytes per element).
    # At 10M rows with avg 3 tokens each, that's ~3 GB just for these columns.
    # They are no longer needed on s23 (blocker already consumed them).
    _drop_blocker_cols(s23)
    gc.collect()

    # --- RAM fix C: use tuple-based lookup (saves ~8 GB vs dict-of-dicts) ---
    # build_lookup now returns entity_id -> 6-tuple instead of entity_id -> dict.
    # A dict of 6 entries costs ~900 bytes in CPython; a 6-tuple costs ~104 bytes.
    # 10M records: 900 MB saved on tuple overhead alone (string data is shared).
    s1_lookup  = build_lookup(s1)
    s23_lookup = build_lookup(s23)

    # --- RAM fix D: free s23 DataFrame after lookup is built ---
    # s23_lookup now contains everything needed for feature computation;
    # the DataFrame itself is redundant.
    del s23
    gc.collect()

    # Train/val split (85/15 on S1 entity IDs)
    rng = np.random.RandomState(42)
    all_s1 = s1['entity_id'].values.copy()
    rng.shuffle(all_s1)
    split = int(len(all_s1) * 0.85)
    train_ids = set(all_s1[:split])
    val_ids   = set(all_s1[split:])
    del all_s1
    print(f"  Train S1: {len(train_ids):,}  Val S1: {len(val_ids):,}")

    # -----------------------------------------------------------------------
    # Blocking. Candidates are generated in CHUNKS of S1 and immediately
    # reduced before the chunk dict is discarded — same as the previous version.
    #
    # --- RAM fix E: write pairs to disk stores instead of Python lists ---
    # tr_pairs / vl_pairs / val_all_pairs accumulated as Python lists of tuples
    # can reach 4-5 GB combined at full scale. _DiskPairStore streams each
    # (s1_id, s23_id) pair to a temp binary file with negligible RAM overhead.
    # -----------------------------------------------------------------------
    CHUNK_SIZE = 20_000
    hits = total = 0
    total_cand_pairs = 0
    rng_tr = np.random.RandomState(42)
    rng_vl = np.random.RandomState(43)

    tr_store  = _DiskPairStore()  # training pairs
    vl_store  = _DiskPairStore()  # validation pairs (for lgb val set)
    vc_store  = _DiskPairStore()  # ALL val candidate pairs (for threshold tuning)

    _tr_label_parts = []
    _vl_label_parts = []

    n_s1 = len(s1)
    print("\nGenerating candidates + building train/val pairs (streamed by chunk) ...")
    for start in range(0, n_s1, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_s1)
        chunk_s1 = s1.iloc[start:end].reset_index(drop=True)
        chunk_cands = blocker.transform(chunk_s1)

        total_cand_pairs += sum(len(v) for v in chunk_cands.values())

        chunk_ids = chunk_s1['entity_id'].values
        chunk_train_ids = [sid for sid in chunk_ids if sid in train_ids]
        chunk_val_ids   = [sid for sid in chunk_ids if sid in val_ids]

        # Val blocking-recall stats + val_all_pairs -> disk
        for sid in chunk_val_ids:
            true = gt_dict.get(sid, set())
            c = chunk_cands.get(sid, set())
            for m in true:
                total += 1
                if m in c:
                    hits += 1
            for cid in c:
                if cid in s23_lookup:
                    vc_store.append(sid, cid)
            # GT pairs missed by blocking also go to vc_store
            # (same role as old val_gt_extra — they are needed so threshold
            #  tuning can observe these hard positives)
            for m in true:
                if m in s23_lookup and m not in c:
                    vc_store.append(sid, m)

        p, y = make_pairs_and_labels(chunk_train_ids, chunk_cands, gt_dict, s23_lookup,
                                     neg_ratio=5, rng=rng_tr)
        tr_store.extend_pairs(p)
        _tr_label_parts.append(y)

        p, y = make_pairs_and_labels(chunk_val_ids, chunk_cands, gt_dict, s23_lookup,
                                     neg_ratio=5, rng=rng_vl)
        vl_store.extend_pairs(p)
        _vl_label_parts.append(y)

        del chunk_cands, p, y
        gc.collect()
        print(f"    {end:,}/{n_s1:,} S1 processed", flush=True)

    # --- RAM fix F: drop Python-object columns from s1 after chunk loop ---
    # s1 still needs name_toks / all_sx / addr_nums during transform() on each
    # chunk slice, so we only drop them after all chunks are done.
    _drop_blocker_cols(s1)
    gc.collect()

    y_tr = np.concatenate(_tr_label_parts) if _tr_label_parts else np.array([], dtype=np.int8)
    y_vl = np.concatenate(_vl_label_parts) if _vl_label_parts else np.array([], dtype=np.int8)
    del _tr_label_parts, _vl_label_parts
    gc.collect()

    blocking_recall = hits / max(total, 1)
    max_possible = n_s1 * len(s23_lookup)
    reduction = 1.0 - total_cand_pairs / max(max_possible, 1)
    print(f"\n  Blocking recall (val): {blocking_recall:.4f}  ({hits:,}/{total:,})")
    print(f"  Reduction ratio: {reduction:.8f}")
    print(f"  Candidate pairs: {total_cand_pairs:,}")
    print(f"  Train: {len(tr_store):,} pairs  (pos={y_tr.sum():,}, neg={len(y_tr)-y_tr.sum():,})")
    print(f"  Val:   {len(vl_store):,} pairs  (pos={y_vl.sum():,}, neg={len(y_vl)-y_vl.sum():,})")

    # -----------------------------------------------------------------------
    # Featurize — write to memmap files so X_tr / X_vl never fully occupy RAM.
    # The valid rows are then compacted into in-RAM arrays for LightGBM.
    # At full scale X_tr ~840 MB and X_vl ~140 MB — these are acceptable once
    # the other large structures have been freed.
    # -----------------------------------------------------------------------
    tmpdir = tempfile.gettempdir()
    mm_tr_path = os.path.join(tmpdir, 'ber_X_tr.mmap')
    mm_vl_path = os.path.join(tmpdir, 'ber_X_vl.mmap')
    mm_vc_path = os.path.join(tmpdir, 'ber_X_vc.mmap')

    print("\nFeaturizing training pairs (disk -> memmap) ...")
    mm_tr, m_tr = _featurize_store_to_memmap(tr_store, s1_lookup, s23_lookup, mm_tr_path)
    tr_store.close()

    print("Featurizing validation pairs (disk -> memmap) ...")
    mm_vl, m_vl = _featurize_store_to_memmap(vl_store, s1_lookup, s23_lookup, mm_vl_path)
    vl_store.close()

    # Compact into in-RAM arrays (valid rows only). This is necessary because
    # LightGBM works best with a contiguous ndarray.
    print("  Compacting valid training/val rows into RAM ...")
    X_tr = np.array(mm_tr[m_tr], dtype=np.float32)
    y_tr = y_tr[m_tr]
    del mm_tr
    try:
        os.remove(mm_tr_path)
    except OSError:
        pass

    X_vl = np.array(mm_vl[m_vl], dtype=np.float32)
    y_vl = y_vl[m_vl]
    del mm_vl
    try:
        os.remove(mm_vl_path)
    except OSError:
        pass
    gc.collect()
    print(f"  X_tr shape: {X_tr.shape}  X_vl shape: {X_vl.shape}")

    # Train LightGBM
    print("\nTraining LightGBM ...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES, free_raw_data=True)
    dval   = lgb.Dataset(X_vl, label=y_vl, feature_name=FEATURE_NAMES, free_raw_data=True)

    params = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 255, 'learning_rate': 0.03,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_child_samples': 20, 'verbosity': -1, 'n_jobs': -1,
        'max_depth': -1,
        'lambda_l1': 0.1, 'lambda_l2': 1.0,
        'scale_pos_weight': float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1),
    }
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])

    # --- RAM fix G: free training matrices right after model is trained ---
    del X_tr, X_vl, y_tr, y_vl, dtrain, dval
    gc.collect()

    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    order = np.argsort(imp)[::-1]
    print("\nFeature importance (gain):")
    for idx in order[:20]:
        print(f"  {FEATURE_NAMES[idx]:30s}  {imp[idx]:.0f}")

    # -----------------------------------------------------------------------
    # Threshold tuning on ALL validation candidates.
    # vc_store holds all val candidate pairs on disk. We featurize them to a
    # memmap and then score them in streaming batches — X_vc is never loaded
    # fully into RAM.
    # -----------------------------------------------------------------------
    print("\nScoring validation candidates for threshold tuning ...")
    print(f"  Val candidate pairs (on disk): {len(vc_store):,}")

    print("  Featurizing val-all pairs (disk -> memmap) ...")
    mm_vc, m_vc = _featurize_store_to_memmap(vc_store, s1_lookup, s23_lookup, mm_vc_path)

    # Score in streaming batches; accumulate only (sid, cid, prob) in a dict.
    # This avoids materializing the full X_vc (~3 GB) and full val_all_pairs
    # list (~3.5 GB) simultaneously.
    print("  Scoring val candidates in streaming batches ...")
    scored_d = defaultdict(list)   # sid -> [(prob, cid), ...]
    batch_size_score = 200_000
    vc_offset = 0
    for batch in vc_store.iterate(batch_size_score):
        blen = len(batch)
        batch_X = mm_vc[vc_offset:vc_offset + blen]
        batch_mask = m_vc[vc_offset:vc_offset + blen]
        valid_X = np.array(batch_X[batch_mask], dtype=np.float32)
        if len(valid_X):
            probs_batch = model.predict(valid_X)
            vi = 0
            for j in range(blen):
                if batch_mask[j]:
                    sid, cid = batch[j]
                    scored_d[sid].append((float(probs_batch[vi]), cid))
                    vi += 1
        vc_offset += blen

    vc_store.close()
    del mm_vc, valid_X
    try:
        os.remove(mm_vc_path)
    except OSError:
        pass
    gc.collect()

    # Fine-grained threshold search
    best_f05, best_thr, best_k = 0, 0.5, 3
    for thr in np.arange(0.05, 0.98, 0.01):
        for k in [3, 5, 8, 12]:
            pred_d = defaultdict(set)
            for sid in val_ids:
                pred_d[sid] = set()
            for sid, cands_list in scored_d.items():
                filtered = [(prob, cid) for prob, cid in cands_list if prob >= thr]
                filtered.sort(reverse=True, key=lambda x: x[0])
                for _, cid in filtered[:k]:
                    pred_d[sid].add(cid)
            f = macro_f05(pred_d, gt_dict, list(val_ids))
            if f > best_f05:
                best_f05, best_thr, best_k = f, thr, k

    del scored_d
    gc.collect()

    print(f"\n  Best threshold: {best_thr:.2f}")
    print(f"  Best K: {best_k}")
    print(f"  Validation F_0.5:  {best_f05:.4f}")

    # Save
    artefact = {'model': model, 'threshold': best_thr, 'k_cap': best_k, 'features': FEATURE_NAMES,
                'blocker_cfg': {'max_block': blocker.max_block, 'tfidf_k': blocker.tfidf_k}}
    path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {path}")

    return dict(blocking_recall=blocking_recall, reduction_ratio=reduction,
                val_f05=best_f05, threshold=best_thr)


###############################################################################
# TEST INFERENCE
###############################################################################
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
    # --- RAM fix: free s2/s3 immediately after concat ---
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    print(f"  Combined S2+S3: {len(s23):,} (s2/s3 freed)")

    # Load model
    path = os.path.join(MODEL_DIR, 'model.pkl')
    with open(path, 'rb') as f:
        artefact = pickle.load(f)
    model = artefact['model']
    thr   = artefact['threshold']
    k_cap = artefact.get('k_cap', 5)
    print(f"  Model loaded, threshold={thr:.2f}, k_cap={k_cap}")

    # Lookups
    cfg = artefact['blocker_cfg']
    blocker = Blocker(**cfg)
    blocker.fit(s23)

    # --- RAM fix: drop blocker columns from s23 after fit ---
    _drop_blocker_cols(s23)
    gc.collect()

    s1_lookup  = build_lookup(s1)
    s23_lookup = build_lookup(s23)

    # --- RAM fix: free s23 DataFrame after lookup is built ---
    del s23
    gc.collect()

    # Blocking + scoring, streamed by S1 chunk. The old code called
    # blocker.transform(s1) once on the full test set (cands: a dict of
    # ~1.7M sets), then built all_pairs = [] as one list of every
    # (s1_id, s23_id) pair before scoring anything -- both are O(all_S1) in
    # memory and are what caused the OOM. This bounds peak memory to one
    # chunk at a time and writes results as it goes instead of holding
    # everything until the end.
    CHUNK_SIZE = 20_000
    n_s1 = len(s1)
    mp = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    cp = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    n_matched = 0
    n_links = 0

    print("\nScoring candidate pairs (streamed by S1 chunk) ...")
    with open(mp, 'w', encoding='utf-8') as fm, open(cp, 'w', encoding='utf-8') as fc:
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        fc.write('source1_entity_id\tcandidate_entity_ids\n')

        for start in range(0, n_s1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_s1)
            chunk_s1 = s1.iloc[start:end].reset_index(drop=True)
            chunk_ids = chunk_s1['entity_id'].values

            chunk_cands = blocker.transform(chunk_s1)
            pairs = [(sid, c) for sid in chunk_ids for c in chunk_cands.get(sid, set())]

            matches_scored = defaultdict(list)
            if pairs:
                X, mask = featurize(pairs, s1_lookup, s23_lookup)
                X = X[mask]
                valid_pairs = [p for p, ok in zip(pairs, mask) if ok]
                if len(X):
                    p = model.predict(X)
                    for i, (sid, cid) in enumerate(valid_pairs):
                        if p[i] >= thr:
                            matches_scored[sid].append((p[i], cid))
                del X, mask, valid_pairs

            for sid in chunk_ids:
                cl = matches_scored.get(sid, [])
                cl.sort(reverse=True, key=lambda x: x[0])
                match_ids = sorted({cid for _, cid in cl[:k_cap]})
                cand_ids = sorted(chunk_cands.get(sid, set()))

                fm.write(f"{sid}\t{','.join(match_ids)}\n")
                fc.write(f"{sid}\t{','.join(cand_ids)}\n")
                if match_ids:
                    n_matched += 1
                    n_links += len(match_ids)

            del chunk_cands, pairs, matches_scored
            gc.collect()
            print(f"    {end:,}/{n_s1:,} S1 processed", flush=True)

    print(f"  Matched S1: {n_matched:,}  Links: {n_links:,}  Singletons: {n_s1-n_matched:,}")
    print(f"  Written: {mp}")
    print(f"  Written: {cp}")


###############################################################################
# ENTRY POINT
###############################################################################
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
