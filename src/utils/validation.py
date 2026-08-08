import pandas as pd

def validate_dataframe(df: pd.DataFrame)-> None:
    """
    Validate a dataframe for duplicate rows and missing values.
    """
    print("="*60)
    print("Null & Duplicate Validation")
    print("="*60)
    dupes = df.duplicated().sum()
    nulls = df.isna().sum()[df.isna().sum() > 0]
    if dupes > 0:
        print(f"  Duplicate rows: {dupes:,}")
    else:
            print(f"✅ No duplicates")
    if len(nulls) > 0:
            print("\nMissing value:")
            print(nulls)
    else:
        print(f"  ✅ No missing values")