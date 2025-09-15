# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris_sklearn = load_iris()
    
    # Convert to pandas DataFrame
    iris = pd.DataFrame(data=iris_sklearn.data, columns=iris_sklearn.feature_names)
    iris['species'] = pd.Categorical.from_codes(iris_sklearn.target, iris_sklearn.target_names)
    
    print("First 5 rows of the Iris dataset:")
    display(iris.head())
    
    print("\nData types and missing values:")
    print(iris.info())
    print("\nMissing values per column:")
    print(iris.isnull().sum())
    
    # Since there are no missing values, no cleaning needed
except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis

# Basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
display(iris.describe())

# Group by species and compute mean of numerical columns
species_group = iris.groupby('species').mean()
print("\nMean values of numerical columns grouped by species:")
display(species_group)

# Observations:
print("\nObservations:")
print("- Setosa species has smaller petal length and width compared to Versicolor and Virginica.")
print("- Virginica tends to have the largest sepal and petal dimensions.")
print("- Sepal length and petal length increase from Setosa to Virginica.")

# Task 3: Data Visualization

# 1. Line chart showing trends over samples (not time-series, but index as proxy)
plt.figure(figsize=(10, 6))
for species in iris['species'].unique():
    subset = iris[iris['species'] == species]
    plt.plot(subset.index, subset['sepal length (cm)'], label=species)
plt.title('Sepal Length Trend by Sample Index and Species')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x=species_group.index, y=species_group['petal length (cm)'], palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram of sepal width to understand distribution
plt.figure(figsize=(8, 5))
sns.histplot(iris['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: sepal length vs petal length colored by species
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()