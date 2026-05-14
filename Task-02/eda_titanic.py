import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

# Create a directory to save the plots
os.makedirs("visualizations", exist_ok=True)

# 1. Load Data
file_path = r"c:\DS-prodigy-Task-1\data-science-datasets-main\Task 2\train.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 2. Data Cleaning
# Drop 'Cabin' because it has too many missing values
df = df.drop(columns=['Cabin'])

# Fill missing 'Age' values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' values with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 3. EDA and Visualizations

# 3.1 Overall Survival
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Overall Survival (0 = No, 1 = Yes)')
plt.savefig('visualizations/overall_survival.png', bbox_inches='tight')
plt.close()

# 3.2 Survival by Gender
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival by Gender')
plt.savefig('visualizations/survival_by_gender.png', bbox_inches='tight')
plt.close()

# 3.3 Survival by Passenger Class
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set2')
plt.title('Survival by Passenger Class')
plt.savefig('visualizations/survival_by_pclass.png', bbox_inches='tight')
plt.close()

# 3.4 Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.savefig('visualizations/age_distribution.png', bbox_inches='tight')
plt.close()

# 3.5 Survival by Age
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, palette='Set2')
plt.title('Survival Distribution by Age')
plt.xlabel('Age')
plt.savefig('visualizations/survival_by_age.png', bbox_inches='tight')
plt.close()

# 3.6 Correlation Heatmap
# Select only numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig('visualizations/correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("\nEDA completed. Visualizations saved in the 'visualizations' folder.")
