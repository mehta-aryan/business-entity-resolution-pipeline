"""EDA script to profile train and test data."""
import pandas as pd
import sys

# --- Train data ---
print("=" * 80)
print("TRAIN DATA PROFILING")
print("=" * 80)

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

for name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3), ("Ground Truth", gt)]:
    print(f"\n--- {name} ---")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")
    print(f"  Null counts:\n{df.isnull().sum()}")
    print(f"  Null rates:\n{(df.isnull().mean() * 100).round(2)}")
    print(f"  Head:\n{df.head(3)}")

# Country distribution
print("\n--- Country Distribution (Train) ---")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"  {name}: {df['country'].value_counts().to_dict()}")

# Ground truth analysis
print("\n--- Ground Truth Analysis ---")
print(f"  Total S1 entities in GT: {len(gt)}")

# Parse matched_entity_ids
gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [] if pd.isna(x) or str(x).strip() == '' else str(x).split(',')
)
gt['match_count'] = gt['match_list'].apply(len)
singletons = (gt['match_count'] == 0).sum()
print(f"  Singletons (no match): {singletons} ({singletons/len(gt)*100:.1f}%)")
print(f"  Match count distribution:\n{gt['match_count'].value_counts().sort_index()}")

# Source distribution in matches
all_matches = [m for lst in gt['match_list'] for m in lst]
s2_matches = [m for m in all_matches if m.startswith('S2-')]
s3_matches = [m for m in all_matches if m.startswith('S3-')]
print(f"  Total match links: {len(all_matches)}")
print(f"    S2 matches: {len(s2_matches)}")
print(f"    S3 matches: {len(s3_matches)}")

# How many S1 match only S2, only S3, both
gt['has_s2'] = gt['match_list'].apply(lambda lst: any(m.startswith('S2-') for m in lst))
gt['has_s3'] = gt['match_list'].apply(lambda lst: any(m.startswith('S3-') for m in lst))
non_sing = gt[gt['match_count'] > 0]
print(f"  Non-singleton S1 entities: {len(non_sing)}")
only_s2 = ((non_sing['has_s2']) & (~non_sing['has_s3'])).sum()
only_s3 = ((~non_sing['has_s2']) & (non_sing['has_s3'])).sum()
both = ((non_sing['has_s2']) & (non_sing['has_s3'])).sum()
print(f"    Match only S2: {only_s2}")
print(f"    Match only S3: {only_s3}")
print(f"    Match both S2 & S3: {both}")

# --- Test data ---
print("\n" + "=" * 80)
print("TEST DATA PROFILING")
print("=" * 80)

ts1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
ts2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t")
ts3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t")

for name, df in [("Test Source1", ts1), ("Test Source2", ts2), ("Test Source3", ts3)]:
    print(f"\n--- {name} ---")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null counts:\n{df.isnull().sum()}")
    print(f"  Null rates:\n{(df.isnull().mean() * 100).round(2)}")

# Country distribution test
print("\n--- Country Distribution (Test) ---")
for name, df in [("TS1", ts1), ("TS2", ts2), ("TS3", ts3)]:
    print(f"  {name}: {df['country'].value_counts().to_dict()}")

# Name/address noise patterns - sample some records
print("\n--- Sample Records ---")
print("S1 samples:")
print(s1.sample(5, random_state=42).to_string())
print("\nS2 samples:")
print(s2.sample(5, random_state=42).to_string())
print("\nS3 samples:")
print(s3.sample(5, random_state=42).to_string())

# Look at some matched pairs to understand noise
print("\n--- Sample Matched Pairs ---")
sample_gt = gt[gt['match_count'] > 0].sample(10, random_state=42)
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1_rec = s1[s1['entity_id'] == s1_id]
    if len(s1_rec) == 0:
        continue
    print(f"\nS1: {s1_id}")
    print(f"  Name: {s1_rec.iloc[0]['business_name']}")
    print(f"  Addr: {s1_rec.iloc[0]['business_address']}")
    print(f"  Country: {s1_rec.iloc[0]['country']}")
    for mid in row['match_list'][:3]:
        if mid.startswith('S2-'):
            rec = s2[s2['entity_id'] == mid]
        else:
            rec = s3[s3['entity_id'] == mid]
        if len(rec) > 0:
            print(f"  -> {mid}")
            print(f"     Name: {rec.iloc[0]['business_name']}")
            print(f"     Addr: {rec.iloc[0]['business_address']}")
            print(f"     Country: {rec.iloc[0]['country']}")
