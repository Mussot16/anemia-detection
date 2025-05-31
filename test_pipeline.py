import cloudpickle
import pandas as pd

with open("predictor/anemia_pipeline_final.joblib", "rb") as f:
    pipe = cloudpickle.load(f)

test_cases = [
    {"sex": 0, "red": 0.45, "green": 0.30, "blue": 0.25}, 
    {"sex": 1, "red": 0.48, "green": 0.32, "blue": 0.20}, 
    {"sex": 0, "red": 0.33, "green": 0.33, "blue": 0.34}, 
    {"sex": 1, "red": 0.25, "green": 0.25, "blue": 0.50}, 
    {"sex": 0, "red": 0.70, "green": 0.20, "blue": 0.10}, 
]

def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rg_ratio"] = df["red"] / (df["green"] + 1e-6)
    df["rb_ratio"] = df["red"] / (df["blue"] + 1e-6)
    df["gb_ratio"] = df["green"] / (df["blue"] + 1e-6)
    return df

for i, sample in enumerate(test_cases):
    df = pd.DataFrame([sample])
    X = augment_features(df)
    proba = pipe.predict_proba(X)
    prob_anemia = float(proba[0][1]) if proba.ndim > 1 else float(proba[0])
    diagnosis = "Anemia" if prob_anemia >= 0.5 else "No Anemia"
    print(f"\nCase {i+1}")
    print(f"  Input: {sample}")
    print(f"  Probability: {prob_anemia:.4f}")
    print(f"  Diagnosis:  {diagnosis}")
