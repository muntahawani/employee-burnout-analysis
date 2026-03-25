import pandas as pd

# Load your dataset FIRST
df = pd.read_csv("C:/Users/S2/Desktop/Employee Burnout Analysis/dataset/train.csv")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.dropna(subset=["Burn Rate"])

    df["Resource Allocation"] = df["Resource Allocation"].fillna(
        df["Resource Allocation"].mean()
    )

    df["Mental Fatigue Score"] = df["Mental Fatigue Score"].fillna(
        df["Mental Fatigue Score"].mean()
    )

    df = df.drop(["Employee ID", "Date of Joining"], axis=1, errors="ignore")

    df["WFH Setup Available"] = df["WFH Setup Available"].map({
        "No": 0,
        "Yes": 1
    })

    df["Company Type"] = df["Company Type"].map({
        "Service": 0,
        "Product": 1
    })

    df["Gender"] = df["Gender"].str.strip().str.capitalize()
    df["Gender"] = df["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    df = df.dropna()

    return df

# Run preprocessing
df = pd.read_csv("C:/Users/S2/Desktop/Employee Burnout Analysis/dataset/train.csv")

cleaned_df = preprocess_data(df)

# Save cleaned data
cleaned_df.to_csv("C:/Users/S2/Desktop/Employee Burnout Analysis/dataset/cleaned_data.csv", index=False)

print("Preprocessing done. Cleaned data saved.")