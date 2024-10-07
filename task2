import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visualization
sns.set(style="whitegrid")

# Load the Titanic dataset 
df = pd.read_csv('/content/titanic.csv')
df

df.head()
# Check for missing values
df.isnull().sum()

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df['Age']
df['Embarked']

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
df.head()
df.describe()

#Exploratory Data Analysis (EDA)
# Get summary statistics
print(df.describe())


# Histogram of 'Age'
plt.hist(df['Age'], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()


# Correlation matrix
numerical_df = df.select_dtypes(include=['number'])  
correlation_matrix = numerical_df.corr()  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# Survival rate by gender
survival_rate_by_gender = df.groupby('Sex')['Survived'].mean() 
print(survival_rate_by_gender)

# Survival rate by class
survival_rate_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_rate_by_class)

# Pair plot to visualize relationships
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived')
plt.show()


# Scatter plot for 'Age' vs 'Fare'
plt.scatter(df['Age'], df['Fare'], c=df['Survived'], cmap='coolwarm', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare colored by Survival')
plt.colorbar(label='Survived')
plt.show()
