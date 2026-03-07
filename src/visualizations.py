import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self, style='darkgrid', figsize=(12, 6)):
        sns.set_style(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
    def plot_price_history(self, df, title="Stock Price History"):
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.5),
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        axes[0].plot(df.index, df['Close'], label='Close', color=self.colors[0], linewidth=2)
        if 'MA20' in df.columns:
            axes[0].plot(df.index, df['MA20'], label='20-day MA', color=self.colors[1], alpha=0.7)
        axes[0].set_title(title)
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume plot
        axes[1].bar(df.index, df['Volume'], color='gray', alpha=0.6)
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, y_true, y_pred, dates=None, title="Predictions vs Actual"):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Check if dates is a pandas DatetimeIndex and convert if needed
        if dates is not None:
            if isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Index):
                x = dates
                xlabel = 'Date'
            else:
                x = range(len(y_true))
                xlabel = 'Sample'
        else:
            x = range(len(y_true))
            xlabel = 'Sample'
        
        ax.plot(x, y_true, label='Actual', color=self.colors[0], linewidth=2, marker='o', markersize=4)
        ax.plot(x, y_pred, label='Predicted', color=self.colors[1], linewidth=2, marker='s', markersize=4)
        
        # Confidence interval
        errors = y_true - y_pred
        std_error = np.std(errors)
        ax.fill_between(range(len(y_pred)), y_pred - 2*std_error, y_pred + 2*std_error,
                       alpha=0.2, color=self.colors[1], label='95% CI')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate date labels if they're dates
        if dates is not None and (isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Index)):
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names, importance, title="Feature Importance", top_n=20):
        fig, ax = plt.subplots(figsize=(10, max(6, top_n*0.3)))
        
        indices = np.argsort(importance)[::-1][:top_n]
        y_pos = np.arange(len(indices))
        
        ax.barh(y_pos, importance[indices], color=self.colors[0])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
