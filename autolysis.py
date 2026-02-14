import sys
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(path):
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Dataset loaded using encoding: {encoding}")
            return df
        except Exception:
            continue

    print("Failed to load dataset with common encodings.")
    sys.exit(1)

def analyze_dataset(df):
    analysis = {}

    analysis["shape"] = df.shape
    analysis["columns"] = df.columns.tolist()
    analysis["dtypes"] = df.dtypes.astype(str).to_dict()
    analysis["missing"] = df.isnull().sum().to_dict()
    analysis["duplicates"] = int(df.duplicated().sum())

    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(include="object")

    analysis["summary"] = (
        df.describe(include="all").transpose().to_string()
        if not df.empty else "Dataset is empty."
    )

    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        analysis["correlation"] = corr_matrix.to_string()
    else:
        analysis["correlation"] = "Insufficient numeric columns."

    outlier_report = {}
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = numeric_df[(numeric_df[col] < Q1 - 1.5 * IQR) |
                              (numeric_df[col] > Q3 + 1.5 * IQR)]
        outlier_report[col] = len(outliers)

    analysis["outliers"] = outlier_report

    return analysis

def generate_visualizations(df, output_folder):
    images = []
    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(include="object")

    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(6, 6))
        sns.heatmap(numeric_df.corr(), cmap="viridis", annot=False)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        path = os.path.join(output_folder, "correlation.png")
        plt.savefig(path)
        plt.close()
        images.append("correlation.png")

    if numeric_df.shape[1] > 0:
        variances = numeric_df.var()
        col = variances.idxmax()
        plt.figure(figsize=(6, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        path = os.path.join(output_folder, "distribution.png")
        plt.savefig(path)
        plt.close()
        images.append("distribution.png")

    if categorical_df.shape[1] > 0:
        col = categorical_df.columns[0]
        plt.figure(figsize=(6, 6))
        df[col].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top 10 Categories in {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = os.path.join(output_folder, "categories.png")
        plt.savefig(path)
        plt.close()
        images.append("categories.png")

    return images
def generate_llm_report(analysis):
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        return "ERROR: AIPROXY_TOKEN not set."

    prompt = f"""
You are a senior data scientist preparing an automated analytical report.

Dataset Characteristics:
- Shape: {analysis["shape"]}
- Columns: {analysis["columns"]}
- Data Types: {analysis["dtypes"]}
- Missing Values: {analysis["missing"]}
- Duplicate Rows: {analysis["duplicates"]}
- Outlier Counts: {analysis["outliers"]}

Statistical Summary:
{analysis["summary"]}

Correlation Insights:
{analysis["correlation"]}

Write a professional Markdown report structured as:

# Automated Analysis Report
## 1. Data Overview
## 2. Analytical Approach
## 3. Insights and Patterns
## 4. Strategic Implications

Keep it analytical and avoid inventing data.
"""

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    try:
        response = requests.post(
            "https://api.aiproxy.io/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        result = response.json()

        if "error" in result:
            return f"LLM API Error: {result['error']['message']}"

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM API Error: {e}"
def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    os.makedirs(dataset_name, exist_ok=True)

    df = load_dataset(file_path)
    analysis = analyze_dataset(df)
    images = generate_visualizations(df, dataset_name)
    report = generate_llm_report(analysis)

    readme_path = os.path.join(dataset_name, "README.md")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n## Visualizations\n")
        for img in images:
            f.write(f"![{img}]({img})\n")

    print(f"Analysis completed. Results stored in '{dataset_name}' folder.")


if __name__ == "__main__":
    main()