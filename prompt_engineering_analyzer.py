"""
Prompt Engineering Analyzer for Anthropic's Claude API

This script analyzes how different prompt engineering techniques affect
Claude's output distributions, token probabilities, and response characteristics.
"""

import anthropic
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from collections import Counter
import time


class PromptEngineeringAnalyzer:
    """Analyzes the impact of prompt engineering on LLM outputs."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the analyzer with Anthropic API.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.results = []

    def get_response(self, prompt: str, system: str = "", temperature: float = 1.0,
                     max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Get a response from Claude with detailed metrics.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing response and metadata
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        result = {
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "response_text": response.content[0].text,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": response.model,
        }

        return result

    def compare_prompts(self, base_question: str, prompt_variants: Dict[str, Dict[str, str]],
                       temperature: float = 1.0, num_samples: int = 5) -> pd.DataFrame:
        """
        Compare different prompt engineering techniques.

        Args:
            base_question: The core question to ask
            prompt_variants: Dict of variant_name -> {"prompt": str, "system": str}
            temperature: Sampling temperature
            num_samples: Number of samples per variant

        Returns:
            DataFrame with comparative results
        """
        results = []

        for variant_name, config in prompt_variants.items():
            print(f"Testing variant: {variant_name}")

            for sample_idx in range(num_samples):
                result = self.get_response(
                    prompt=config.get("prompt", base_question),
                    system=config.get("system", ""),
                    temperature=temperature
                )

                result["variant"] = variant_name
                result["sample_idx"] = sample_idx
                result["response_length"] = len(result["response_text"])
                result["word_count"] = len(result["response_text"].split())

                results.append(result)
                time.sleep(0.5)  # Rate limiting

        self.results = results
        return pd.DataFrame(results)

    def analyze_output_diversity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze diversity of outputs across variants.

        Args:
            df: DataFrame from compare_prompts

        Returns:
            Dictionary with diversity metrics
        """
        diversity_metrics = {}

        for variant in df['variant'].unique():
            variant_responses = df[df['variant'] == variant]['response_text'].tolist()

            # Calculate unique responses
            unique_responses = len(set(variant_responses))

            # Calculate average response length
            avg_length = df[df['variant'] == variant]['response_length'].mean()

            # Calculate average word count
            avg_words = df[df['variant'] == variant]['word_count'].mean()

            # Calculate vocabulary diversity (unique words / total words)
            all_words = []
            for response in variant_responses:
                all_words.extend(response.lower().split())
            vocab_diversity = len(set(all_words)) / len(all_words) if all_words else 0

            diversity_metrics[variant] = {
                "unique_responses": unique_responses,
                "total_samples": len(variant_responses),
                "uniqueness_ratio": unique_responses / len(variant_responses),
                "avg_response_length": avg_length,
                "avg_word_count": avg_words,
                "vocab_diversity": vocab_diversity,
                "total_vocabulary": len(set(all_words))
            }

        return diversity_metrics

    def analyze_temperature_effects(self, prompt: str, system: str = "",
                                   temperatures: List[float] = [0.0, 0.3, 0.7, 1.0, 1.5],
                                   num_samples: int = 10) -> pd.DataFrame:
        """
        Analyze how temperature affects outputs for a given prompt.

        Args:
            prompt: The prompt to test
            system: System prompt
            temperatures: List of temperatures to test
            num_samples: Samples per temperature

        Returns:
            DataFrame with results
        """
        results = []

        for temp in temperatures:
            print(f"Testing temperature: {temp}")

            for sample_idx in range(num_samples):
                result = self.get_response(
                    prompt=prompt,
                    system=system,
                    temperature=temp
                )

                result["temperature"] = temp
                result["sample_idx"] = sample_idx
                result["response_length"] = len(result["response_text"])
                result["word_count"] = len(result["response_text"].split())

                results.append(result)
                time.sleep(0.5)

        return pd.DataFrame(results)

    def visualize_prompt_comparison(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualizations comparing prompt variants.

        Args:
            df: DataFrame from compare_prompts
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Response length distribution
        df.boxplot(column='response_length', by='variant', ax=axes[0, 0])
        axes[0, 0].set_title('Response Length Distribution by Prompt Variant')
        axes[0, 0].set_xlabel('Prompt Variant')
        axes[0, 0].set_ylabel('Response Length (characters)')
        plt.sca(axes[0, 0])
        plt.xticks(rotation=45, ha='right')

        # 2. Word count distribution
        df.boxplot(column='word_count', by='variant', ax=axes[0, 1])
        axes[0, 1].set_title('Word Count Distribution by Prompt Variant')
        axes[0, 1].set_xlabel('Prompt Variant')
        axes[0, 1].set_ylabel('Word Count')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=45, ha='right')

        # 3. Token usage
        token_data = df.groupby('variant').agg({
            'input_tokens': 'mean',
            'output_tokens': 'mean'
        }).reset_index()

        x = np.arange(len(token_data))
        width = 0.35

        axes[1, 0].bar(x - width/2, token_data['input_tokens'], width, label='Input Tokens', alpha=0.8)
        axes[1, 0].bar(x + width/2, token_data['output_tokens'], width, label='Output Tokens', alpha=0.8)
        axes[1, 0].set_xlabel('Prompt Variant')
        axes[1, 0].set_ylabel('Average Token Count')
        axes[1, 0].set_title('Token Usage by Prompt Variant')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(token_data['variant'], rotation=45, ha='right')
        axes[1, 0].legend()

        # 4. Response diversity (unique responses per variant)
        diversity = df.groupby('variant')['response_text'].apply(lambda x: len(set(x)) / len(x))
        axes[1, 1].bar(diversity.index, diversity.values, alpha=0.8, color='coral')
        axes[1, 1].set_xlabel('Prompt Variant')
        axes[1, 1].set_ylabel('Uniqueness Ratio')
        axes[1, 1].set_title('Response Diversity (Unique Responses / Total Samples)')
        axes[1, 1].set_ylim(0, 1.1)
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def visualize_temperature_effects(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Visualize how temperature affects outputs.

        Args:
            df: DataFrame from analyze_temperature_effects
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Response length vs temperature
        temp_groups = df.groupby('temperature')['response_length'].agg(['mean', 'std']).reset_index()
        axes[0, 0].errorbar(temp_groups['temperature'], temp_groups['mean'],
                           yerr=temp_groups['std'], marker='o', capsize=5, capthick=2)
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Response Length (characters)')
        axes[0, 0].set_title('Response Length vs Temperature')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Response diversity vs temperature
        diversity = df.groupby('temperature')['response_text'].apply(lambda x: len(set(x)) / len(x))
        axes[0, 1].plot(diversity.index, diversity.values, marker='o', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Temperature')
        axes[0, 1].set_ylabel('Uniqueness Ratio')
        axes[0, 1].set_title('Response Diversity vs Temperature')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.1)

        # 3. Distribution of response lengths
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]['response_length']
            axes[1, 0].hist(temp_data, alpha=0.5, label=f'T={temp}', bins=15)
        axes[1, 0].set_xlabel('Response Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Response Length Distribution by Temperature')
        axes[1, 0].legend()

        # 4. Word count vs temperature
        df.boxplot(column='word_count', by='temperature', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Word Count')
        axes[1, 1].set_title('Word Count Distribution by Temperature')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def save_results(self, df: pd.DataFrame, output_path: str):
        """
        Save results to CSV and JSON.

        Args:
            df: DataFrame with results
            output_path: Base path for output files (without extension)
        """
        # Save CSV
        df.to_csv(f"{output_path}.csv", index=False)

        # Save JSON with full details
        df.to_json(f"{output_path}.json", orient='records', indent=2)

        print(f"Results saved to {output_path}.csv and {output_path}.json")


def example_usage():
    """Example usage of the PromptEngineeringAnalyzer."""

    # Initialize analyzer
    analyzer = PromptEngineeringAnalyzer()

    # Define a base question
    base_question = "What are the key principles of machine learning?"

    # Define different prompt engineering variants
    prompt_variants = {
        "zero_shot": {
            "prompt": base_question,
            "system": ""
        },
        "with_context": {
            "prompt": base_question,
            "system": "You are an expert machine learning educator who explains concepts clearly and concisely."
        },
        "chain_of_thought": {
            "prompt": f"{base_question}\n\nLet's think through this step by step:",
            "system": ""
        },
        "few_shot": {
            "prompt": f"""Here are some examples of explaining technical concepts:

Q: What are the key principles of databases?
A: The key principles include: 1) Data persistence and durability, 2) ACID properties for transactions, 3) Efficient querying through indexing, 4) Normalization to reduce redundancy.

Q: {base_question}
A:""",
            "system": ""
        },
        "structured_output": {
            "prompt": f"{base_question}\n\nPlease provide your answer in the following format:\n- Principle 1: [description]\n- Principle 2: [description]\n- Principle 3: [description]",
            "system": ""
        },
        "persona_expert": {
            "prompt": base_question,
            "system": "You are Andrew Ng, a world-renowned AI researcher and educator. Answer questions with deep technical insight while maintaining accessibility."
        }
    }

    print("=" * 80)
    print("PROMPT ENGINEERING ANALYSIS")
    print("=" * 80)
    print(f"\nBase Question: {base_question}\n")

    # Compare prompt variants
    print("\n--- Comparing Prompt Variants ---")
    df_variants = analyzer.compare_prompts(
        base_question=base_question,
        prompt_variants=prompt_variants,
        temperature=0.7,
        num_samples=5
    )

    # Analyze diversity
    print("\n--- Analyzing Output Diversity ---")
    diversity_metrics = analyzer.analyze_output_diversity(df_variants)

    print("\nDiversity Metrics:")
    for variant, metrics in diversity_metrics.items():
        print(f"\n{variant}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.3f}")
            else:
                print(f"  {metric_name}: {value}")

    # Visualize results
    print("\n--- Generating Visualizations ---")
    analyzer.visualize_prompt_comparison(df_variants, save_path="prompt_comparison.png")

    # Temperature analysis
    print("\n--- Analyzing Temperature Effects ---")
    df_temperature = analyzer.analyze_temperature_effects(
        prompt=base_question,
        system="You are a helpful AI assistant.",
        temperatures=[0.0, 0.3, 0.7, 1.0, 1.5],
        num_samples=10
    )

    analyzer.visualize_temperature_effects(df_temperature, save_path="temperature_effects.png")

    # Save results
    print("\n--- Saving Results ---")
    analyzer.save_results(df_variants, "prompt_variants_results")
    analyzer.save_results(df_temperature, "temperature_analysis_results")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

    return analyzer, df_variants, df_temperature


if __name__ == "__main__":
    # Run example analysis
    analyzer, df_variants, df_temperature = example_usage()
