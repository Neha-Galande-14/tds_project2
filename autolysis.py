import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
import chardet
from tenacity import RetryError


# Constants
API_URL = "https://api.openai.com/v1/chat/completions"
AIPROXY_TOKEN = "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"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_openai_api(prompt):
    """Call OpenAI API with retries."""
    api_token = os.getenv("AI_PROXY")
    if not api_token:
        raise ValueError("API token not found. Set it in the AI_PROXY environment variable.")

    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()  # Raise exception for HTTP errors
    except httpx.RequestError as e:
        raise ValueError(f"Request error: {e}") from e
    except httpx.HTTPStatusError as e:
        raise ValueError(f"HTTP error {response.status_code}: {response.text}") from e

    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError as e:
        raise ValueError(f"Unexpected API response format: {response.text}") from e


def load_data(file_path):
    """Load CSV data with encoding detection."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    return pd.read_csv(file_path, encoding=encoding)

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
    }
    return analysis

def visualize_data(df, output_dir):
    """Generate and save visualizations in the output directory."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'))
        plt.close()

def generate_narrative(analysis):
    """Generate narrative using LLM with optimized prompt."""
    # Truncate or simplify the dataset summary
    truncated_summary = {k: v for i, (k, v) in enumerate(analysis['summary'].items()) if i < 5}
    truncated_missing_values = {k: v for i, (k, v) in enumerate(analysis['missing_values'].items()) if i < 5}
    truncated_correlation = {k: {kk: vv for j, (kk, vv) in enumerate(v.items()) if j < 3} 
                              for i, (k, v) in enumerate(analysis['correlation'].items()) if i < 3}

    prompt = (
        "Analyze the following dataset summary and provide a concise narrative with key insights: "
        f"\n\nSummary (truncated): {truncated_summary}\n\n"
        f"Missing Values (truncated): {truncated_missing_values}\n\n"
        f"Correlation (truncated): {truncated_correlation}"
    )

    # Check prompt size
    if len(prompt) > 4000:  # Adjust limit as per model's context size
        raise ValueError("Prompt exceeds token limit even after truncation. Further simplify the dataset.")

    return call_openai_api(prompt)


def main(file_path):
    # Create a folder based on the dataset's name (without the file extension)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.getcwd(), dataset_name)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        df = load_data(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load the file. {str(e)}")
        sys.exit(1)

    # Perform data analysis
    try:
        analysis = analyze_data(df)
    except Exception as e:
        print(f"Error: Data analysis failed. {str(e)}")
        sys.exit(1)

    # Generate visualizations and save them in the output directory
    try:
        visualize_data(df, output_dir)
    except Exception as e:
        print(f"Error: Visualization generation failed. {str(e)}")
        sys.exit(1)

    # Generate narrative
    try:
        narrative = generate_narrative(analysis)
        # Save the narrative in the output directory as README.md
        with open(os.path.join(output_dir, 'README.md'), 'w') as f:
            f.write(narrative)
    except RetryError as re:
        print(f"Error: Narrative generation failed after retries. {str(re)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Narrative generation failed. {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])