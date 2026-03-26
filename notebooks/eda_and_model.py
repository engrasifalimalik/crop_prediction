# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/crop_data.csv")

# Show data
print(df.head())

# Basic info
print(df.info())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()
