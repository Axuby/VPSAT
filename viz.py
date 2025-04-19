import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from pathlib import Path
import json


class AdvancedVisualizer:
    def __init__(self, results_dir='./ablation_results'):
        """
        Initialize the advanced visualization tools.

        Args:
            results_dir: Directory containing ablation study results
        """
        self.results_dir = Path(results_dir)
        self.fig_dir = self.results_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # Custom color palette
        self.palette = sns.color_palette("viridis", 10)
        self.custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", self.palette)

        # Set seaborn style
        sns.set_theme(style="whitegrid", font_scale=1.2)

    def _load_results(self, experiment_name):
        """Load experiment results from file."""
        try:
            with open(self.results_dir / f"{experiment_name}_results.json", 'r') as f:
                return json.load(f)
        except:
            print(f"No results found for experiment {experiment_name}")
            return None

    def plot_angular_error_distribution(self, angular_errors, experiment_name, percentile_threshold=95):
        """
        Create a beautiful visualization of angular error distribution.

        Args:
            angular_errors: Array of angular errors in degrees
            experiment_name: Name of the experiment configuration
            percentile_threshold: Percentile threshold for highlighting errors
        """
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], width_ratios=[2, 1])

        # Compute statistics
        mean_error = np.mean(angular_errors)
        median_error = np.median(angular_errors)
        std_error = np.std(angular_errors)
        percentile_95 = np.percentile(angular_errors, 95)

        # 1. Main histogram in top left
        ax1 = fig.add_subplot(gs[0, 0])

        bins = np.linspace(0, 90, 36)  # 0 to 90 degrees in 2.5-degree bins
        n, bins, patches = ax1.hist(angular_errors, bins=bins, alpha=0.7,
                                    color=self.palette[0], edgecolor='black', linewidth=1.5)

        # Color the bins above threshold differently
        threshold_bin = np.searchsorted(bins, percentile_95)
        for i in range(threshold_bin, len(patches)):
            patches[i].set_facecolor(self.palette[-1])

        # Add vertical lines for mean and median
        ax1.axvline(mean_error, color='red', linestyle='-', linewidth=2,
                    label=f'Mean: {mean_error:.2f}°')
        ax1.axvline(median_error, color='green', linestyle='--', linewidth=2,
                    label=f'Median: {median_error:.2f}°')
        ax1.axvline(percentile_95, color='purple', linestyle='-.', linewidth=2,
                    label=f'95th Percentile: {percentile_95:.2f}°')

        ax1.set_title(f"Distribution of Angular Errors - {experiment_name}", fontsize=14, fontweight='bold')
        ax1.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.legend(fontsize=10, frameon=True)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. KDE plot in top right
        ax2 = fig.add_subplot(gs[0, 1])

        # Apply Gaussian KDE for smooth distribution
        kde = gaussian_kde(angular_errors)
        x_range = np.linspace(0, min(90, max(angular_errors) * 1.2), 1000)
        ax2.plot(x_range, kde(x_range), color=self.palette[2], linewidth=2.5)
        ax2.fill_between(x_range, kde(x_range), alpha=0.4, color=self.palette[2])

        # Add vertical lines for statistics
        ax2.axvline(mean_error, color='red', linestyle='-', linewidth=2,
                    label=f'Mean: {mean_error:.2f}°')
        ax2.axvline(median_error, color='green', linestyle='--', linewidth=2,
                    label=f'Median: {median_error:.2f}°')

        ax2.set_title("Density Estimation", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 3. Cumulative distribution in bottom row
        ax3 = fig.add_subplot(gs[1, :])

        # Sort errors for cumulative plot
        sorted_errors = np.sort(angular_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        # Plot CDF
        ax3.plot(sorted_errors, cumulative, color=self.palette[4], linewidth=3)

        # Add reference lines
        for p in [25, 50, 75, 90, 95]:
            percentile_val = np.percentile(angular_errors, p)
            ax3.axvline(percentile_val, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax3.axhline(p, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax3.text(percentile_val + 1, p - 2, f'{p}%: {percentile_val:.2f}°',
                     fontsize=9, ha='left', va='top')

        ax3.set_title("Cumulative Distribution of Angular Errors", fontsize=14, fontweight='bold')
        ax3.set_xlabel('Angular Error (degrees)', fontsize=12)
        ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim(0, min(90, max(angular_errors) * 1.2))
        ax3.set_ylim(0, 100)

        # Add statistics table as text
        stats_text = (
            f"Statistical Summary:\n"
            f"Mean Error: {mean_error:.2f}°\n"
            f"Median Error: {median_error:.2f}°\n"
            f"Std Deviation: {std_error:.2f}°\n"
            f"95th Percentile: {percentile_95:.2f}°\n"
            f"Max Error: {np.max(angular_errors):.2f}°\n"
            f"Min Error: {np.min(angular_errors):.2f}°"
        )
        ax3.text(0.02, 0.2, stats_text, transform=ax3.transAxes, fontsize=11,
                 va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Set the background color for all subplots
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#f9f9f9')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = self.fig_dir / f"{experiment_name}_angular_error_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return fig

    def create_parameter_radar_chart(self, experiment_results, metric='best_val_loss', title=None):
        """
        Create a radar chart comparing the best configurations for different parameters.

        Args:
            experiment_results: Dictionary mapping parameter names to best result values
            metric: Metric used for comparison
            title: Chart title
        """
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # Get parameter names and metric values
        params = list(experiment_results.keys())
        values = list(experiment_results.values())

        # Number of variables
        N = len(params)

        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Normalize values for the radar chart (0-1 scale)
        min_val = min(values)
        max_val = max(values)

        if min_val == max_val:
            normalized = [0.5] * len(values)
        else:
            if metric.endswith('loss'):  # Lower is better
                normalized = [1 - (v - min_val) / (max_val - min_val) for v in values]
            else:  # Higher is better
                normalized = [(v - min_val) / (max_val - min_val) for v in values]

        normalized += normalized[:1]  # Close the loop

        # Plot data
        ax.plot(angles, normalized, 'o-', linewidth=2.5, color=self.palette[0])
        ax.fill(angles, normalized, alpha=0.25, color=self.palette[0])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params, fontsize=12)

        # Add parameter values as text
        for i, (angle, param, value) in enumerate(zip(angles[:-1], params, values)):
            ha = 'left' if angle < np.pi else 'right'
            ax.text(angle, normalized[i] + 0.1, f"{value:.3f}",
                    fontsize=10, ha=ha, va='center', fontweight='bold')

        # Remove radial labels and set grid style
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set title
        plt.title(title or f"Parameter Comparison ({metric})", fontsize=14, fontweight='bold', pad=20)

        # Save figure
        fig_path = self.fig_dir / f"parameter_radar_chart_{metric}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return

    def fig_angular_error_comparison(self, experiment_configs, experiment_names=None):
        """
        Plot angular error distributions across multiple experiments.

        Args:
            experiment_configs: Dictionary mapping experiment names to file paths with angular errors
            experiment_names: Optional custom display names for the experiments
        """
        if not experiment_names:
            experiment_names = list(experiment_configs.keys())

        # Create figure
        plt.figure(figsize=(14, 10))

        # Set up violin plot parameters
        violin_parts = []
        error_data = []

        # Load error data
        for exp_name, error_path in experiment_configs.items():
            try:
                errors = np.load(error_path)
                error_data.append(errors)
            except:
                print(f"Could not load angular errors from {error_path}")
                return

        # Create violin plot
        positions = np.arange(len(experiment_names))
        violin_parts = plt.violinplot(error_data, positions=positions, vert=True,
                                      widths=0.8, showmeans=False, showextrema=False)

        # Customize violin plots
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(self.palette[i % len(self.palette)])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        # Add box plots inside violin plots for better statistics visualization
        box_parts = plt.boxplot(error_data, positions=positions, widths=0.15,
                                patch_artist=True, showfliers=False)

        # Customize box plots
        for i, box in enumerate(box_parts['boxes']):
            box.set(facecolor='white', alpha=0.8)
            box.set(edgecolor='black', linewidth=1.5)

        for i, median in enumerate(box_parts['medians']):
            median.set(color='firebrick', linewidth=2)

        # Add scatter points for individual data (with jitter)
        for i, data in enumerate(error_data):
            # Sample points if too many
            sample_size = min(100, len(data))
            sample_idx = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data[sample_idx]

            # Add jitter
            jitter = np.random.normal(0, 0.05, size=sample_size)
            plt.scatter(i + jitter, sample_data, alpha=0.4, s=20,
                        c=self.palette[i % len(self.palette)], edgecolor='none')

        # Add mean and median values as text
        for i, data in enumerate(error_data):
            mean_val = np.mean(data)
            median_val = np.median(data)
            plt.text(i, np.max(data) + 2, f"Mean: {mean_val:.2f}°", ha='center', fontweight='bold')
            plt.text(i, np.max(data) + 5, f"Median: {median_val:.2f}°", ha='center', fontweight='bold')

        # Set title and labels
        plt.title("Comparison of Angular Error Distributions", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Experiment Configuration", fontsize=14, labelpad=10)
        plt.ylabel("Angular Error (degrees)", fontsize=14, labelpad=10)

        # Set x-ticks
        plt.xticks(positions, experiment_names, fontsize=12, rotation=30, ha='right')
        plt.yticks(fontsize=12)

        # Add grid
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Add background color
        plt.gca().set_facecolor('#f9f9f9')

        # Add border
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = self.fig_dir / "angular_error_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_parameter_grid_search(self, x_param, y_param, results, metric='best_val_loss', title=None):
        """
        Plot a beautiful heatmap for 2D parameter grid search.

        Args:
            x_param: Parameter name for x-axis
            y_param: Parameter name for y-axis
            results: 2D array of results
            metric: Metric to visualize
            title: Plot title
        """
        # Format parameter names for display
        x_display = x_param.split('.')[-1].replace('_', ' ').title()
        y_display = y_param.split('.')[-1].replace('_', ' ').title()

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create a masked array to highlight the best result
        masked_results = np.ma.array(results, mask=False)
        if np.min(results) < 0:  # For metrics where higher is better
            best_idx = np.unravel_index(np.argmax(results), results.shape)
        else:  # For metrics where lower is better (like loss)
            best_idx = np.unravel_index(np.argmin(results), results.shape)

        # Create heatmap with more appealing colors
        ax = sns.heatmap(results, annot=True, fmt=".3f", cmap=self.custom_cmap,
                         cbar_kws={'label': metric.replace('_', ' ').title()})

        # Highlight the best result
        ax.add_patch(plt.Rectangle(best_idx[::-1], 1, 1, fill=False, edgecolor='white', lw=2, clip_on=False))

        # Set title and labels
        plt.title(title or f"Grid Search: {x_display} vs {y_display}", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(x_display, fontsize=14, labelpad=10)
        plt.ylabel(y_display, fontsize=14, labelpad=10)

        # Improve colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(metric.replace('_', ' ').title(), fontsize=14, labelpad=10)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = self.fig_dir / f"grid_search_{x_param.split('.')[-1]}_{y_param.split('.')[-1]}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_ablation_dashboard(self, experiment_names, sort_by='best_val_loss'):
        """
        Create a comprehensive dashboard of ablation study results.

        Args:
            experiment_names: List of experiment names to include
            sort_by: Metric to sort experiments by
        """
        # Create figure with grid layout
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1.5])

        all_results = []
        best_values = []
        best_metrics = []

        # Load results for all experiments
        for exp_name in experiment_names:
            results = self._load_results(exp_name)
            if results:
                all_results.append((exp_name, results))
                # Find best configuration
                if sort_by in results[0]:
                    if sort_by.startswith('best') or sort_by.endswith('loss'):
                        best_idx = np.argmin([r[sort_by] for r in results])
                    else:
                        best_idx = np.argmax([r[sort_by] for r in results])
                    best_values.append(results[best_idx]['value'])
                    best_metrics.append(results[best_idx][sort_by])

        # Sort experiments by performance
        if best_metrics:
            sorted_data = sorted(zip(experiment_names, best_values, best_metrics),
                                 key=lambda x: x[2])
            exp_names_sorted, best_values_sorted, best_metrics_sorted = zip(*sorted_data)
        else:
            exp_names_sorted = experiment_names
            best_values_sorted = [None] * len(experiment_names)
            best_metrics_sorted = [None] * len(experiment_names)

        # 1. Summary bar chart in top left
        ax1 = fig.add_subplot(gs[0, 0])
        param_names = [name.split('_')[-1].replace('_', ' ').title() for name in exp_names_sorted]

        # Create bars
        bars = ax1.bar(param_names, best_metrics_sorted, color=self.palette[:len(param_names)],
                       alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, best_values_sorted)):
            if value is not None:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(best_metrics_sorted),
                         f'{value}', ha='center', va='bottom', fontsize=10)

        ax1.set_title("Best Configuration Per Parameter", fontsize=14, fontweight='bold')
        ax1.set_ylabel(sort_by.replace('_', ' ').title(), fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # 2. Convergence plot in top right
        ax2 = fig.add_subplot(gs[0, 1])

        # For each experiment, plot convergence of best configuration
        for i, (exp_name, results) in enumerate(all_results):
            if sort_by.startswith('best') or sort_by.endswith('loss'):
                best_idx = np.argmin([r[sort_by] for r in results])
            else:
                best_idx = np.argmax([r[sort_by] for r in results])

            result = results[best_idx]
            param_name = exp_name.split('_')[-1].replace('_', ' ').title()

            if 'train_losses' in result and 'val_losses' in result:
                epochs = range(1, len(result['train_losses']) + 1)
                ax2.plot(epochs, result['val_losses'], '-', linewidth=2,
                         label=f"{param_name}={result['value']}",
                         color=self.palette[i % len(self.palette)])

        ax2.set_title("Validation Loss Convergence of Best Configurations", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=9, ncol=2)

        # 3. Detailed parameter comparisons (bottom row)
        for i, (exp_name, results) in enumerate(all_results[:2]):  # Show first two experiments in detail
            param_display = exp_name.split('_')[-1].replace('_', ' ').title()
            values = [r['value'] for r in results]
            metrics = [r[sort_by] for r in results]

            # Bottom left plot
            if i == 0:
                ax3 = fig.add_subplot(gs[1:, 0])

                # Create fancy violin plot
                if len(values) > 1:  # Only create violin plot if multiple values
                    vplot_data = []
                    for result in results:
                        if 'val_losses' in result:
                            # Use last 3 epochs for distribution
                            vplot_data.append(result['val_losses'][-3:])

                    if vplot_data:
                        violin_parts = ax3.violinplot(vplot_data, positions=range(len(values)),
                                                      vert=True, widths=0.8, showmeans=True)

                        # Customize violin parts
                        for j, pc in enumerate(violin_parts['bodies']):
                            pc.set_facecolor(self.palette[j % len(self.palette)])
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)

                # Add bar chart on top of violins
                bars = ax3.bar(range(len(values)), metrics, alpha=0.3, width=0.4,
                               color=self.palette[:len(values)], edgecolor='black')

                # Add value labels
                for j, (bar, value) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(metrics),
                             f'{value}', ha='center', va='bottom', fontsize=12)

                ax3.set_title(f"Detailed Analysis: {param_display}", fontsize=14, fontweight='bold')
                ax3.set_ylabel(sort_by.replace('_', ' ').title(), fontsize=12)
                ax3.set_xticks(range(len(values)))
                ax3.set_xticklabels([str(v) for v in values], fontsize=12)
                ax3.grid(axis='y', alpha=0.3, linestyle='--')

            # Bottom right plot
            elif i == 1:
                ax4 = fig.add_subplot(gs[1:, 1])

                # Create detailed convergence plot for all values
                for j, result in enumerate(results):
                    if 'train_losses' in result and 'val_losses' in result:
                        epochs = range(1, len(result['train_losses']) + 1)
                        val_line = ax4.plot(epochs, result['val_losses'], '-', linewidth=2,
                                            label=f"{param_display}={result['value']} (Val)",
                                            color=self.palette[j % len(self.palette)])
                        ax4.plot(epochs, result['train_losses'], '--', linewidth=1.5,
                                 label=f"{param_display}={result['value']} (Train)",
                                 color=val_line[0].get_color(), alpha=0.7)

                ax4.set_title(f"Convergence Behavior: {param_display}", fontsize=14, fontweight='bold')
                ax4.set_xlabel('Epoch', fontsize=12)
                ax4.set_ylabel('Loss', fontsize=12)
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.legend(fontsize=9, ncol=2)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = self.fig_dir / "ablation_dashboard.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_parameter_comparison(self, param_path, title=None, metric='best_val_loss', sort=True, log_scale=False):
        """
        Create a beautiful bar chart comparing different parameter values.

        Args:
            param_path: Parameter path (e.g., 'model.transformer.num_layers')
            title: Plot title
            metric: Metric to visualize ('best_val_loss' or 'val_loss')
            sort: Whether to sort values by performance
            log_scale: Whether to use log scale for y-axis
        """
        experiment_name = param_path.replace('.', '_')
        results = self._load_results(experiment_name)
        if not results:
            return

        # Format parameter name for display
        param_display = param_path.split('.')[-1].replace('_', ' ').title()

        # Extract values and metrics
        values = [str(r['value']) for r in results]
        metrics = [r[metric] for r in results]

        # Sort if requested
        if sort:
            value_metric_pairs = sorted(zip(values, metrics), key=lambda x: x[1])
            values, metrics = zip(*value_metric_pairs)

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create bar chart with custom styling
        bars = plt.bar(values, metrics, color=self.palette, alpha=0.8, width=0.6, edgecolor='black', linewidth=1.5)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(metrics),
                     f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Set log scale if requested
        if log_scale:
            plt.yscale('log')

        # Set title and labels with improved styling
        plt.title(title or f"Effect of {param_display} on Model Performance", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(param_display, fontsize=14, labelpad=10)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14, labelpad=10)

        # Add grid for easier reading
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Improve tick labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add subtle background color
        plt.gca().set_facecolor('#f9f9f9')

        # Add border
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = self.fig_dir / f"{experiment_name}_{metric}_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    def plot_convergence_comparison(self, param_path, values=None, metric='val_loss'):
        """
        Plot the convergence behavior of different parameter values.

        Args:
            param_path: Parameter path to compare
            values: Specific values to include (None for all)
            metric: Metric to plot ('train_loss' or 'val_loss')
        """
        experiment_name = param_path.replace('.', '_')
        results = self._load_results(experiment_name)
        if not results:
            return

        # Format parameter name for display
        param_display = param_path.split('.')[-1].replace('_', ' ').title()

        # Filter results if specific values are requested
        if values is not None:
            results = [r for r in results if r['value'] in values]

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot convergence for each parameter value
        for i, result in enumerate(results):
            value = result['value']
            epochs = range(1, len(result[f'{metric}s']) + 1)
            plt.plot(epochs, result[f'{metric}s'], '-', linewidth=2.5,
                     label=f"{param_display} = {value}", color=self.palette[i % len(self.palette)])

        # Set title and labels
        plt.title(f"Convergence Comparison for Different {param_display} Values", fontsize=16, fontweight='bold',
                  pad=20)
        plt.xlabel('Epoch', fontsize=14, labelpad=10)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14, labelpad=10)

        # Add grid and legend
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, frameon=True, framealpha=0.9, loc='upper right')

        # Improve tick labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add subtle background
        plt.gca().set_facecolor('#f9f9f9')

        # Add border
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        fig_path = self.fig_dir / f"{experiment_name}_convergence_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    # def plot