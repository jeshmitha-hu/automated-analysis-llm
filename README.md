#Automated Analysis with LLM

This project implements an automated data analysis system that works with any CSV file. The script performs exploratory data analysis, generates visualizations, and produces an AI-generated analytical report using GPT-4o-mini.

Project Features

- Automatic CSV loading with multiple encoding support
- Summary statistics and data type detection
- Missing value and duplicate analysis
- Correlation analysis for numeric features
- Outlier detection using IQR method
- Automatic visualization generation (PNG format)
- AI-generated Markdown report
- Structured output folder for each dataset

How to Run

1. Set the environment variable (PowerShell):

$env:AIPROXY_TOKEN="your_token_here"

2. Run the script:

python -m uv run autolysis.py dataset.csv

Example:

python -m uv run autolysis.py media.csv

Repository Structure

- autolysis.py
- LICENSE
- goodreads/
- happiness/
- media/

Each dataset folder contains generated visualizations and a README report.

License
MIT License

