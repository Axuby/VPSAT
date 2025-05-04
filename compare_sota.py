import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def compare_with_sota(your_results_path, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your results
    with open(your_results_path, 'r') as f:
        import json
        your_results = json.load(f)
    
    # SOTA methods and their reported performance
    # These are example values - replace with actual published numbers
    sota_methods = {
        'VPSAT': {
            'mean_error': your_results['mean_error'],
            'median_error': your_results['median_error'],
            'success_rate_5deg': your_results['success_rate_5deg'] * 100,
            'success_rate_10deg': your_results['success_rate_10deg'] * 100,
            'success_rate_20deg': your_results['success_rate_20deg'] * 100
        },
    'NeurVPS': {
        'mean_error': 3.4,       # degrees
        'median_error': 1.5,     # degrees
        'success_rate_5deg': 61.7,   # percentage
        'success_rate_10deg': 83.2,  # percentage
        'success_rate_20deg': 94.5   # percentage
    },
    'CONSAC': {
        'mean_error': 5.8,
        'median_error': 2.9,
        'success_rate_5deg': 54.3,
        'success_rate_10deg': 76.1,
        'success_rate_20deg': 90.2
    },
    'J-Linkage': {
        'mean_error': 8.7,
        'median_error': 4.5,
        'success_rate_5deg': 43.8,
        'success_rate_10deg': 65.4,
        'success_rate_20deg': 82.1
    }
}

    
    # Create DataFrame for plotting
    df = pd.DataFrame(sota_methods).T.reset_index()
    df = df.rename(columns={'index': 'Method'})
    
    # Plot error metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(df['Method']))
    width = 0.35
    
    plt.bar(x - width/2, df['mean_error'], width, label='Mean Error', color='steelblue')
    plt.bar(x + width/2, df['median_error'], width, label='Median Error', color='lightcoral')
    
    plt.xlabel('Method')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Error Comparison with State-of-the-Art Methods')
    plt.xticks(x, df['Method'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(df['mean_error']):
        plt.text(i - width/2, v + 0.5, f'{v:.2f}°', ha='center')
    
    for i, v in enumerate(df['median_error']):
        plt.text(i + width/2, v + 0.5, f'{v:.2f}°', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=300)
    
    # Plot success rates
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(df['Method']))
    width = 0.25
    
    plt.bar(x - width, df['success_rate_5deg'], width, label='< 5°', color='#5DA5DA')
    plt.bar(x, df['success_rate_10deg'], width, label='< 10°', color='#FAA43A')
    plt.bar(x + width, df['success_rate_20deg'], width, label='< 20°', color='#60BD68')
    
    plt.xlabel('Method')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison with State-of-the-Art Methods')
    plt.xticks(x, df['Method'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels
    for i, v in enumerate(df['success_rate_5deg']):
        plt.text(i - width, v + 1, f'{v:.1f}%', ha='center')
    
    for i, v in enumerate(df['success_rate_10deg']):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    for i, v in enumerate(df['success_rate_20deg']):
        plt.text(i + width, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=300)
    
    print(f"SOTA comparison plots saved to {output_dir}")

if __name__ == "__main__":
    your_results_path = "checkpoint/best_model/stats.json"  # Path to your model's stats
    output_dir = "sota_comparison"
    
    compare_with_sota(your_results_path, output_dir)
