import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """# Titanic Dataset: Exploratory Data Analysis (EDA)
This notebook performs data cleaning and exploratory data analysis on the Titanic dataset."""

code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

# Load Data
file_path = r"c:\\DS-prodigy-Task-1\\data-science-datasets-main\\Task 2\\train.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\\nMissing values before cleaning:")
print(df.isnull().sum())
df.head()"""

text_2 = """## Data Cleaning
We drop the 'Cabin' column due to a high number of missing values. We fill missing 'Age' values with the median and 'Embarked' with the mode."""

code_2 = """# Drop 'Cabin' because it has too many missing values
df = df.drop(columns=['Cabin'])

# Fill missing 'Age' values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' values with the mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\\nMissing values after cleaning:")
print(df.isnull().sum())"""

text_3 = """## Exploratory Data Analysis & Visualizations"""

code_3 = """# Overall Survival
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='Set2', hue='Survived', legend=False)
plt.title('Overall Survival (0 = No, 1 = Yes)')
plt.show()"""

code_4 = """# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival by Gender')
plt.show()"""

code_5 = """# Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set2')
plt.title('Survival by Passenger Class')
plt.show()"""

code_6 = """# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.show()"""

code_7 = """# Survival by Age
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, palette='Set2')
plt.title('Survival Distribution by Age')
plt.xlabel('Age')
plt.show()"""

code_8 = """# Correlation Heatmap
numeric_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_code_cell(code_8)
]

with open('c:\\DS-prodigy-Task-2\\eda_titanic.ipynb', 'w') as f:
    nbf.write(nb, f)
