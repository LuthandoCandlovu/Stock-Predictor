import sys
print('?? Verifying Stock Predictor installation...\n')

# Check imports
modules = [
    'src.data_loader',
    'src.preprocessor', 
    'src.feature_engineering',
    'src.models.random_forest',
    'src.models.lstm',
    'src.models.ensemble',
    'src.visualizations',
    'src.utils'
]

for module in modules:
    try:
        __import__(module)
        print(f'? {module}')
    except ImportError as e:
        print(f'? {module}: {e}')

print('\n?? Directory structure:')
import os
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files only
        print(f'{subindent}{file}')
