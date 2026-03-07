import subprocess
import time

print('🚀 Testing all models...\n')

models = ['random_forest', 'lstm', 'ensemble']
symbols = ['AAPL', 'MSFT']

for symbol in symbols:
    for model in models:
        print(f'\n📊 Testing {symbol} with {model}...')
        try:
            result = subprocess.run(
                ['python', 'main.py', '--symbol', symbol, '--model', model,
                 '--start', '2024-01-01', '--end', '2024-03-01'],
                capture_output=True, text=True, timeout=60
            )
            if '✅ Analysis complete' in result.stdout:
                print('   ✅ Success!')
            else:
                print('   ❌ Failed')
                if result.stderr:
                    print(f'   Error: {result.stderr[:200]}')
        except Exception as e:
            print(f'   ❌ Error: {str(e)}')

print('\n✅ Testing complete!')
