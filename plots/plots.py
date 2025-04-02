import json
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import pickle
except ImportError:  # Python 3.x
    import pickle

with open('plots/CCD_results.p', 'rb') as pkl_file:
    CCD_results = pickle.load(pkl_file)

df = pd.DataFrame(CCD_results)
df[['p', 'g']] = df['file'].str.extract(r'p_([\d\.]+)_and_g_([\d\.]+)').astype(float)
df = df.round(3)

print(df)



plt.figure(figsize=(12, 6))
sns.boxplot(x=df['n'], y=df['roc_auc'], hue=df['p'])
plt.xlabel('Sample Size (n)')
plt.ylabel('ROC AUC')
plt.title('Impact of Sample Size on ROC AUC (Grouped by Class Imbalance (p))')
plt.legend(title='Class Imbalance (p)')
plt.grid(True)
plt.show()

# Example: Heatmap for the impact of n and p on Balanced Accuracy
pivot_df = df.pivot_table(index='n', columns='p', values='balanced_accuracy')
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Heatmap: Balanced Accuracy vs. Sample Size and Class Imbalance')
plt.show()

with open('plots/sklearn_results.p', 'rb') as pkl_file:
    sklearn_results = pickle.load(pkl_file)

df2 = pd.DataFrame(CCD_results)
df2[['p', 'g']] = df2['file'].str.extract(r'p_([\d\.]+)_and_g_([\d\.]+)').astype(float)
df2 = df2.round(3)

print(df2)
plt.figure(figsize=(12, 6))
sns.boxplot(x=df2['n'], y=df2['roc_auc'], hue=df2['g'])
plt.xlabel('Sample Size (n)')
plt.ylabel('ROC AUC')
plt.title('Impact of Sample Size on ROC AUC of Logistic Rgression from Scikit-learn (Grouped by Covariances(g))')
plt.legend(title='Covariances(g)')
plt.grid(True)
plt.show()

# Example: Heatmap for the impact of n and p on Balanced Accuracy
pivot_df = df2.pivot_table(index='n', columns='p', values='balanced_accuracy')
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Heatmap: Balanced Accuracy vs. Sample Size and Class Imbalance of Logistic Rgression from Scikit-learn')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=df2['n'], y=df2['roc_auc'], hue=df2['p'])
plt.xlabel('Sample Size (n)')
plt.ylabel('ROC AUC')
plt.title('Impact of Sample Size on ROC AUC of Logistic Rgression from Scikit-learn (Grouped by Class Imbalance (p))')
plt.legend(title='Class Imbalance (p)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x=df2['n'], y=df2['roc_auc'], hue=df2['d'])
plt.xlabel('Sample Size (n)')
plt.ylabel('ROC AUC')
plt.title('Impact of Sample Size on ROC AUC of Logistic Rgression from Scikit-learn (Grouped by Number of features (d)))')
plt.legend(title='Number of features (d)')
plt.grid(True)
plt.show()
