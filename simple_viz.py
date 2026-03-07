import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.visualizations import Visualizer

# Load data
loader = DataLoader()
data = loader.load_from_yahoo('AAPL', '2024-01-01', '2024-06-30')

# Create visualization
viz = Visualizer()
fig = viz.plot_price_history(data, 'Apple (AAPL) Stock Price - First Half 2024')
plt.savefig('AAPL_visualization.png', dpi=100, bbox_inches='tight')
print('✅ Saved AAPL_visualization.png')

# Show basic stats
print(f'\n📊 Statistics:')
print(f'   Start Date: {data.index[0].strftime("%Y-%m-%d")}')
print(f'   End Date: {data.index[-1].strftime("%Y-%m-%d")}')
print(f'   Days: {len(data)}')
print(f'   Min Price: ')
print(f'   Max Price: ')
print(f'   Avg Price: ')
print(f'   Avg Volume: {int(data["Volume"].mean()):,}')
print(f'   First Close: ')
print(f'   Last Close: ')
print(f'   Total Change: {((data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100):.2f}%')
