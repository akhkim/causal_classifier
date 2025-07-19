import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def handle_missing(df, drop_threshold=0.5):
    n_rows = len(df)
    missing_per_row = df.isna().any(axis=1)
    n_missing_rows = missing_per_row.sum()
    frac_missing_rows = n_missing_rows / n_rows

    if frac_missing_rows <= drop_threshold:
        # Drop all rows that have any missing values
        df = df.loc[~missing_per_row].reset_index(drop=True)
        print(f"Dropped {n_missing_rows} incomplete rows ({frac_missing_rows:.1%} of data)")
    else:
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns

        if len(num_cols) > 0:
            num_imp = SimpleImputer(strategy="mean")
            df[num_cols] = num_imp.fit_transform(df[num_cols])
            print(f"Imputed numeric columns: {list(num_cols)}")

        if len(cat_cols) > 0:
            cat_imp = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
            print(f"Imputed categorical columns: {list(cat_cols)}")

        print(f"Imputed missing values in {n_missing_rows} rows ({frac_missing_rows:.1%} of data)")

    return df

def drop_highly_correlated(df):
    numeric_data = df.select_dtypes(include=np.number)
    corr = numeric_data.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"Dropped {len(to_drop)} highly correlated columns: {to_drop}")
    else:
        print("No highly correlated numeric columns to drop.")
    
    return df

def encode_categoricals(df, threshold=0.1):
    n = len(df)
    codebook = {}
    post_df = df.copy()

    non_num_cols = post_df.select_dtypes(exclude=[np.number]).columns

    # Determine low-cardinality columns
    low_card_cols = [
        col for col in non_num_cols
        if post_df[col].nunique(dropna=False) / n <= threshold
    ]

    for col in low_card_cols:
        # Normalize boolean-like values
        # treat any of ('true','yes','y','1') as positive, ('false','no','n','0') as negative
        if post_df[col].dtype == object or post_df[col].dtype.name == 'category':
            vals = post_df[col].astype(str).str.lower().str.strip()
            bool_pos = {'true','yes','y'}
            bool_neg = {'false','no','n'}
            if set(vals.dropna().unique()).issubset(bool_pos.union(bool_neg)):
                post_df[col] = vals.map(lambda x: 1 if x in bool_pos else 0).astype(int)
                codebook[col] = {'positive': 1, 'negative': 0}
                continue

        # Impute missing for categorical
        post_df[col] = SimpleImputer(strategy="most_frequent") \
                         .fit_transform(post_df[[col]]).ravel()

        # Build mapping for remaining categories
        unique_vals = sorted(post_df[col].dropna().unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        post_df[col] = post_df[col].map(mapping).astype(int)
        codebook[col] = mapping

    return post_df, codebook


def save_codebook(codebook):
    if codebook.items() == {}:
        print("No categorical columns to encode.")
        return
    
    lines = []
    for col, mapping in codebook.items():
        lines.append(f"Column: {col}")
        for val, idx in mapping.items():
            lines.append(f"  {idx} -> {val}")
        lines.append("")

    with open("codebook", "w") as f:
        f.write("\n".join(lines))

    print("Codebook saved to codebook")

def save_processed(df):
    df.to_csv("processed.csv", index=False)
    print(f"Preprocessed data saved to processed.csv")

def full_preprocess(df):
    df = handle_missing(df)
    df, codebook = encode_categoricals(df, threshold=0.05)
    df = drop_highly_correlated(df)
    save_processed(df)
    save_codebook(codebook)

    return df