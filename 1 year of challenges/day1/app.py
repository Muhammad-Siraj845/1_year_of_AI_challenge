import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv("titanic.csv")  # Adjust path if titanic.csv is in Day1/
    print(df.info())
    print(df.describe())
except FileNotFoundError:
    print("Error: titanic.csv not found in Day1 folder")
    exit()

# Visualize survival by gender
sns.countplot(x="Survived", hue="Sex", data=df)
plt.savefig("survival_by_gender.png")
plt.close()

# Visualize age distribution
sns.histplot(df["Age"].dropna())
plt.savefig("age_distribution.png")
plt.close()

# Visualize survival by class
sns.countplot(x="Survived", hue="Pclass", data=df)
plt.savefig("survival_by_class.png")
plt.close()