# Titanic Dataset Exploratory Data Analysis (EDA)

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
titanic = pd.read_csv(url)

# 3. Summary statistics
desc = titanic.describe(include='all').T
print(desc)

# 4. Visualizations
## Histograms for numeric features
num_features = ["Age", "Fare", "SibSp", "Parch"]
fig, axes = plt.subplots(2, 2, figsize=(14,8))
for i, feature in enumerate(num_features):
    sns.histplot(titanic[feature].dropna(), kde=True, bins=30, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f"Histogram of {feature}")
plt.tight_layout()
plt.show()

## Boxplots for numeric features
fig, axes = plt.subplots(2, 2, figsize=(14,8))
for i, feature in enumerate(num_features):
    sns.boxplot(y=titanic[feature], ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f"Boxplot of {feature}")
plt.tight_layout()
plt.show()

## Pairplot for feature relationships
sns.pairplot(titanic[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]].dropna(), diag_kind="hist")
plt.show()

## Correlation matrix heatmap
corr = titanic[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation matrix")
plt.show()

# 5. Observations & Insights
print("""
1. Age and Fare variables are right-skewed.
2. There are several outliers in Fare and Age, as seen from boxplots.
3. Pairplot and correlation matrix show weak positive correlation between Fare and Survived.
4. Pclass is negatively correlated with Survived.

""")
