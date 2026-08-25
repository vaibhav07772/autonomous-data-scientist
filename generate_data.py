import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'feature_1': np.random.rand(1000) * 100,
    'feature_2': np.random.rand(1000) * 50,
    'feature_3': np.random.randint(1, 10, 1000),
    'feature_4': np.random.rand(1000) * 200,
    'feature_5': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.rand(1000) * 500 + 100
})

df.to_csv('sample_data.csv', index=False)
print("✅ sample_data.csv created successfully!")