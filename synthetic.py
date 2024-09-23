import numpy as np
import pandas as pd

# Create a synthetic dataset with 2 features and 300 samples
np.random.seed(42)
X1 = np.random.normal(0, 1, (100, 2))
X2 = np.random.normal(5, 1, (100, 2))
X3 = np.random.normal(10, 1, (100, 2))

X = np.vstack((X1, X2, X3))
data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

# Save to CSV
data.to_csv('your_data.csv', index=False)

print("Synthetic data generated and saved as 'synthetic_data.csv'")
