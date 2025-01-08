# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:19:09 2024

@author: MOGIC
"""
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and prepare the data
def load_data(file_path):
    groceries_df = pd.read_csv(file_path, header=None)
    groceries_df = groceries_df.iloc[1:, 1:]  # Remove first row and column
    groceries_df.fillna(0, inplace=True)  # Replace NaN with 0
    return groceries_df

# Step 2: One-hot encode the products
def one_hot_encode(data):
    all_products = set()
    for _, row in data.iterrows():
        all_products.update(row)  # Update set with each row's items
    
    all_products.discard(0)  # Remove '0' from the product list
    encoded_vals = []

    for _, row in data.iterrows():
        rowset = {item: 0 for item in all_products}
        rowset.update({item: 1 for item in row if item != 0})
        encoded_vals.append(rowset)

    return pd.DataFrame(encoded_vals)

# Step 3: Generate candidate itemsets
def generate_candidates(frequent_itemsets, length):

    items = set(item for itemset in frequent_itemsets for item in itemset)
    return set(combinations(items, length))

# Step 4: Implement the pruning algorithm
def prune_candidates(data, candidates, min_support):

    pruned_candidates = {}
    for itemset in candidates:
        support = data[list(itemset)].prod(axis=1).sum() / len(data.index)
        if support >= min_support:
            pruned_candidates[itemset] = support
    return pruned_candidates

# Step 5: Implement the Apriori algorithm
def my_apriori(data, min_support=0.04, max_length=3):

    support = {}
    current_L = set([(item,) for item in data.columns]) # Start with single-item sets

    for length in range(1, max_length + 1):
        candidates = generate_candidates(current_L, length)
        
        pruned_candidates = prune_candidates(data, candidates, min_support)
        support.update(pruned_candidates)
        
        current_L = set(pruned_candidates.keys())
        if not current_L:  # Exit early if no frequent itemsets are found
            break

    return pd.DataFrame(list(support.items()), columns=["Items", "Support"])

# Step 6: Visualization of frequent itemsets
def plot_frequent_itemsets(itemset_df, top_n=10):
    top_itemsets = itemset_df.nlargest(top_n, 'Support')
    top_itemsets = itemset_df.nlargest(top_n, 'Support')

    # Horizontal Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_itemsets['Items'].astype(str), x=top_itemsets['Support'], orient='h')
    plt.title(f'Top {top_n} Frequent Itemsets by Support')
    plt.xlabel('Support')
    plt.ylabel('Itemsets')
    plt.show()

    # Support Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(itemset_df['Support'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Itemset Support')
    plt.xlabel('Support')
    plt.ylabel('Frequency')
    plt.show()

    # Itemset Support Matrix
    plt.figure(figsize=(12, 10))
    itemset_matrix = itemset_df.pivot_table(index='Items', values='Support', aggfunc='mean')
    sns.heatmap(itemset_matrix, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Support'})
    plt.title('Frequent Itemset Support Matrix')
    plt.show()

# Main execution
    groceries_df = load_data("groceries.csv")
    print("Size of table: ", groceries_df.shape)
    print("No of Transactions: ", groceries_df.shape[0])
    print("No of items: ", groceries_df.shape[1])

    # Adjust min_support to capture more itemsets
    encoded_vals_df = one_hot_encode(groceries_df)
    my_freq_itemset = my_apriori(encoded_vals_df, min_support=0.5, max_length=3)

    sorted_freq_itemset = my_freq_itemset.sort_values(by='Support', ascending=False)
    print("Frequent Itemsets:\n", sorted_freq_itemset)
    print("Number of Frequent Itemsets:", sorted_freq_itemset.count(0))

    # Visualize the top 10 itemsets
    plot_frequent_itemsets(sorted_freq_itemset, top_n=10)
