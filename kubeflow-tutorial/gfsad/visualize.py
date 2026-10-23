import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('gfsad1000_tutorial_input.csv')

# Create a scatter plot of geographic coordinates
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df['longitude'], 
    df['latitude'], 
    c=df['landcover'], 
    cmap='viridis', 
    s=50, 
    edgecolor='k', 
    alpha=0.8
)

plt.colorbar(scatter, label='Landcover Class')
plt.title('Spatial Distribution of GFSAD Sample Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
