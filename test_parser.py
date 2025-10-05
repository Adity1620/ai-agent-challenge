import pandas as pd
from custom_parsers.icici_parser import parse

result = parse('data/icici/icici sample.pdf')
expected = pd.read_csv('data/icici/result.csv')

print(f'Got {len(result)} rows, expected {len(expected)}')
print('First few results:')
print(result.head(91))
print('Expected first few:')  
print(expected.head(91))
print('Are they equal?', result.iloc[0].equals(expected.iloc[0]) if len(result) > 0 else 'No data')
