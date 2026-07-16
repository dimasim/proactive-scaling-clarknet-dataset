import json
import pandas as pd
from io import StringIO
import re

with open('model_experiment_walkthrough.ipynb', 'r') as f:
    nb = json.load(f)

print("=== Ekstraksi Output ===")
for cell in nb['cells']:
    if 'outputs' in cell:
        for output in cell['outputs']:
            if output.get('output_type') == 'display_data' or output.get('output_type') == 'execute_result':
                if 'text/html' in output['data']:
                    html = "".join(output['data']['text/html'])
                    try:
                        df = pd.read_html(StringIO(html))[0]
                        print("\nDataframe Ditemukan:")
                        print(df.to_string())
                    except Exception as e:
                        pass
