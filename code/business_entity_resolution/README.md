# Business Entity Resolution Pipeline

## Overview

Entity resolution pipeline for matching business records across 3 independent data sources.
Given business records from Source 1 (deduplicated reference), Source 2, and Source 3 with
noisy/inconsistent fields, the pipeline determines which records refer to the same real-world
business entity.

**Approach:** Multi-key blocking + LightGBM classifier with string-similarity features.

## Requirements

- Python 3.9+
- Dependencies: see `requirements.txt`

Install:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── pipeline.py          # Main pipeline (blocking + features + model + inference)
│   ├── eda.py               # Exploratory data analysis script
│   └── eda2.py              # Additional EDA (matched pair analysis)
├── models/                  # Saved model artifacts (auto-created)
├── README.md                # This file
└── requirements.txt         # Pinned dependencies
```

## How to Reproduce

### Full pipeline (train + test inference):
```bash
cd code/business_entity_resolution
python src/pipeline.py --mode full
```

This will:
1. Load training data from `dataset/train/`
2. Preprocess and normalize names/addresses
3. Build blocking index and generate candidate pairs
4. Train a LightGBM binary classifier
5. Tune the decision threshold on a held-out validation set (F_0.5)
6. Load test data from `dataset/test/`
7. Run blocking + inference on test data
8. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`

### Quick development run (5% sample):
```bash
python src/pipeline.py --mode full --sample 0.05
```

### Train only:
```bash
python src/pipeline.py --mode train
```

### Test inference only (requires pre-trained model in `models/`):
```bash
python src/pipeline.py --mode test
```

## Pipeline Overview

### 1. Blocking / Candidate Generation
- **Multi-token inverted index:** every significant name token generates a blocking key (country-scoped)
- **Phonetic (Soundex) blocking:** on all significant tokens
- **Postal code blocking:** country-agnostic
- **Address number blocking:** 3+ digit numbers as blocking keys
- **TF-IDF ANN:** character 3-4gram TF-IDF, top-k=20 per country
- Max block size: 2000 (pruned above)

### 2. Feature Engineering (30 features)
- **Name features:** exact match, Levenshtein ratio, partial ratio, token sort ratio, token set ratio, Jaccard, containment (both directions), 3-gram Jaccard, prefix ratio, soundex match, length ratio, first-token match, length difference
- **Address features:** Levenshtein ratio, token sort ratio, Jaccard, containment, numeric token Jaccard, numeric overlap count, postal exact/partial/presence, length ratio, empty flag
- **Cross-field features:** country match, name×address products, max name similarity

### 3. Model
- LightGBM gradient-boosted trees (binary classifier)
- 127 leaves, learning rate 0.03, up to 1000 rounds with early stopping
- Scale-pos-weight for class imbalance
- Threshold tuned on validation F_0.5

### 4. Constraints
- No external data, APIs, or internet lookups
- Only MIT/Apache-2.0 licensed models (LightGBM: MIT)
- Output validated with `utils/validate_submission.py`
