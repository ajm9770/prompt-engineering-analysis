# Prompt Engineering Analysis with Anthropic's Claude

A comprehensive toolkit for analyzing how prompt engineering techniques affect LLM output distributions, response characteristics, and behavior using Anthropic's Claude API.

## Overview

This project provides tools to:
- Compare different prompt engineering strategies (zero-shot, few-shot, chain-of-thought, etc.)
- Analyze temperature effects on output diversity
- Visualize response patterns and distributions
- Measure output consistency and creativity
- Understand how prompts influence model behavior

## Features

### Analysis Capabilities
- **Prompt Comparison**: Test multiple prompt variants side-by-side
- **Temperature Analysis**: Study how temperature affects output distribution
- **Diversity Metrics**: Measure response uniqueness and variability
- **Word Frequency Analysis**: Track vocabulary usage across different prompts
- **Statistical Insights**: Quantify differences between prompting strategies
- **Rich Visualizations**: Generate publication-ready charts and graphs

### Tools Included
1. **`prompt_engineering_analyzer.py`**: Standalone Python script with full analysis capabilities
2. **`prompt_analysis_notebook.ipynb`**: Interactive Jupyter notebook for exploration
3. **Comprehensive visualizations**: Automated chart generation

## Installation

### Prerequisites
- Python 3.9 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Setup

1. Clone or navigate to this directory:
```bash
cd /Users/amahajan/src/llm-from-scratch
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or add it to your `.bashrc`/`.zshrc`:
```bash
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Option 1: Python Script (Quick Start)

Run the example analysis:
```bash
python prompt_engineering_analyzer.py
```

This will:
- Compare 6 different prompt engineering techniques
- Analyze temperature effects from 0.0 to 1.5
- Generate visualizations
- Save results to CSV and JSON files

#### Using the Script Programmatically

```python
from prompt_engineering_analyzer import PromptEngineeringAnalyzer

# Initialize
analyzer = PromptEngineeringAnalyzer()

# Define your prompt variants
prompt_variants = {
    "baseline": {
        "prompt": "What is machine learning?",
        "system": ""
    },
    "expert": {
        "prompt": "What is machine learning?",
        "system": "You are an expert ML researcher."
    }
}

# Run comparison
df = analyzer.compare_prompts(
    base_question="What is machine learning?",
    prompt_variants=prompt_variants,
    temperature=0.7,
    num_samples=5
)

# Analyze and visualize
analyzer.visualize_prompt_comparison(df, save_path="results.png")
analyzer.save_results(df, "my_results")
```

### Option 2: Jupyter Notebook (Interactive)

For interactive exploration:
```bash
jupyter notebook prompt_analysis_notebook.ipynb
```

The notebook includes:
- Step-by-step tutorials
- Pre-built examples
- Custom experiment sections
- Detailed explanations of each technique

## Prompt Engineering Techniques Covered

### 1. Zero-Shot (Baseline)
Simple, direct prompts without additional context.
```python
{"prompt": "What are neural networks?", "system": ""}
```

### 2. System Prompts
Adding role/context via system messages.
```python
{
    "prompt": "What are neural networks?",
    "system": "You are an expert AI researcher."
}
```

### 3. Chain-of-Thought
Encouraging step-by-step reasoning.
```python
{
    "prompt": "What are neural networks?\n\nLet's think through this step by step:",
    "system": ""
}
```

### 4. Few-Shot Learning
Providing examples before the actual question.
```python
{
    "prompt": """Q: What is regression?
A: Regression is predicting continuous values.

Q: What are neural networks?
A:""",
    "system": ""
}
```

### 5. Structured Output
Requesting specific formatting.
```python
{
    "prompt": """What are neural networks?

Format:
1. Definition
2. Key components
3. Applications""",
    "system": ""
}
```

### 6. Role-Playing
Assigning specific personas.
```python
{
    "prompt": "What are neural networks?",
    "system": "You are Geoffrey Hinton explaining to a colleague."
}
```

## Output Examples

### Generated Visualizations

1. **Prompt Comparison Dashboard**
   - Response length distributions
   - Word count analysis
   - Token usage comparison
   - Output diversity metrics

2. **Temperature Effects**
   - Response length vs temperature
   - Diversity vs temperature
   - Distribution histograms
   - Boxplot comparisons

3. **Word Frequency Analysis**
   - Top words per variant
   - Vocabulary diversity
   - Pattern recognition

### Data Outputs

Results are saved in multiple formats:
- **CSV**: Easy to import into Excel, Google Sheets
- **JSON**: Structured data with full metadata
- **PNG**: High-resolution visualizations (300 DPI)

## Understanding the Metrics

### Uniqueness Ratio
Proportion of unique responses out of total samples.
- **1.0**: All responses are unique
- **0.5**: Half the responses are duplicates
- **Lower values**: More consistent/deterministic outputs

### Vocabulary Diversity
Ratio of unique words to total words used.
- **Higher values**: More varied language
- **Lower values**: Repetitive or focused vocabulary

### Response Length
Character count and word count metrics.
- Longer responses may indicate more detailed explanations
- Shorter responses might be more concise or restricted

### Token Usage
Input and output token counts.
- Important for API cost estimation
- Helps optimize prompt efficiency

## Example Results

Here's what you might discover:

**Temperature Effects:**
- **T=0.0**: Deterministic, identical responses
- **T=0.3**: Slight variations, mostly consistent
- **T=0.7**: Balanced creativity and coherence
- **T=1.0**: High diversity, some inconsistency
- **T=1.5**: Maximum creativity, potential incoherence

**Prompt Variants:**
- **Chain-of-thought**: Longer, more detailed responses
- **Few-shot**: Better format adherence
- **System prompts**: Tone and style shifts
- **Structured**: More consistent formatting

## Advanced Usage

### Custom Experiment

```python
# Test your own hypothesis
analyzer = PromptEngineeringAnalyzer()

# Define experimental conditions
conditions = {
    "control": {"prompt": "Explain X", "system": ""},
    "treatment": {"prompt": "Explain X in detail", "system": ""}
}

# Run experiment with multiple samples
results = analyzer.compare_prompts(
    base_question="Explain X",
    prompt_variants=conditions,
    temperature=0.7,
    num_samples=20  # More samples = better statistics
)

# Statistical analysis
diversity = analyzer.analyze_output_diversity(results)
print(diversity)
```

### Temperature Sweep

```python
# Fine-grained temperature analysis
df_temp = analyzer.analyze_temperature_effects(
    prompt="Be creative: describe a sunset",
    temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    num_samples=15
)

analyzer.visualize_temperature_effects(df_temp)
```

### Batch Analysis

```python
# Test multiple questions at once
questions = [
    "What is AI?",
    "Explain quantum computing",
    "How do vaccines work?"
]

all_results = []
for question in questions:
    df = analyzer.compare_prompts(
        base_question=question,
        prompt_variants=my_variants,
        temperature=0.7,
        num_samples=5
    )
    all_results.append(df)

# Combine and analyze
combined = pd.concat(all_results)
```

## Cost Estimation

API costs depend on token usage. Typical costs for Claude 3.5 Sonnet:
- Input: ~$3 per million tokens
- Output: ~$15 per million tokens

Example analysis (6 variants × 5 samples):
- ~30 API calls
- ~50,000 tokens total
- Estimated cost: ~$1-2

Use the saved CSV files to track exact token usage.

## Troubleshooting

### API Key Issues
```
Error: Invalid API key
```
**Solution**: Verify your API key is set correctly:
```bash
echo $ANTHROPIC_API_KEY
```

### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: The script includes automatic rate limiting (0.5s between calls). For heavy usage, adjust the sleep time in the code.

### Import Errors
```
ModuleNotFoundError: No module named 'anthropic'
```
**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

## File Structure

```
llm-from-scratch/
├── prompt_engineering_analyzer.py   # Main analysis script
├── prompt_analysis_notebook.ipynb   # Interactive notebook
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── input.py                         # Your existing code
│
└── Generated outputs:
    ├── prompt_comparison.png
    ├── temperature_effects.png
    ├── word_frequency_comparison.png
    ├── prompt_variants_results.csv
    ├── prompt_variants_results.json
    ├── temperature_analysis_results.csv
    └── temperature_analysis_results.json
```

## Contributing

This is a personal analysis toolkit, but feel free to extend it:
- Add new prompt engineering techniques
- Implement additional metrics
- Create new visualizations
- Test different Claude models

## References

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Claude Model Documentation](https://docs.anthropic.com/en/docs/models-overview)

## License

This project is provided as-is for educational and research purposes.

## Support

For issues with:
- **Anthropic API**: Check [Anthropic's documentation](https://docs.anthropic.com/)
- **This toolkit**: Review the code comments and docstrings
- **General ML questions**: The analysis itself can help answer them!

## Next Steps

1. Run the example script to see how it works
2. Open the Jupyter notebook for interactive exploration
3. Modify the prompt variants to test your own hypotheses
4. Analyze the results and draw insights about prompt engineering
5. Share your findings!

Happy prompt engineering!
