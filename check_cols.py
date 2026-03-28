import pandas as pd
df = pd.read_parquet('/Users/sriramganeshalingam/Documents/hmda_pipeline/processed/hmda_cleaned_2007.parquet')
denial_cols = [c for c in df.columns if 'denial' in c.lower()]
print('Denial cols:', denial_cols)
print('All cols:', df.columns.tolist())
