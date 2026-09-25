"""EDA script part 2 - matched pair analysis."""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

gt['match_list'] = gt['matched_entity_ids'].apply(
    lambda x: [] if pd.isna(x) or str(x).strip() == '' else str(x).split(',')
)
gt['match_count'] = gt['match_list'].apply(len)

merged = gt.merge(s1[['entity_id','country']], left_on='source1_entity_id', right_on='entity_id')

# Singleton rate per country
for c in merged['country'].unique():
    sub = merged[merged['country'] == c]
    singletons = (sub['match_count'] == 0).sum()
    print(f'Country={c}: total={len(sub)}, singletons={singletons} ({singletons/len(sub)*100:.1f}%)')
    dist = sub['match_count'].value_counts().sort_index().head(8).to_dict()
    print(f'  Match count dist: {dist}')

# Look at some matched pairs
print()
print('=== Sample Matched Pairs ===')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')

s2_idx = s2.set_index('entity_id')
s3_idx = s3.set_index('entity_id')

sample = merged[merged['match_count'] > 0].sample(12, random_state=42)
for _, row in sample.iterrows():
    s1_rec = s1[s1['entity_id'] == row['source1_entity_id']].iloc[0]
    print(f'S1={row["source1_entity_id"]} | {s1_rec["business_name"]} | {s1_rec["business_address"]} | {s1_rec["country"]}')
    for mid in row['match_list'][:2]:
        if mid.startswith('S2-') and mid in s2_idx.index:
            rec = s2_idx.loc[mid]
        elif mid in s3_idx.index:
            rec = s3_idx.loc[mid]
        else:
            continue
        print(f'  -> {mid} | {rec["business_name"]} | {rec["business_address"]} | {rec["country"]}')
    print()

# Check address patterns - postal codes
print('=== Address patterns ===')
# US addresses
us_s1 = s1[s1['country'] == 'US'].head(20)
for _, r in us_s1.iterrows():
    print(f'  US addr: {r["business_address"]}')

print()
# India addresses
in_s1 = s1[s1['country'] == 'India'].head(20)
for _, r in in_s1.iterrows():
    print(f'  India addr: {r["business_address"]}')
