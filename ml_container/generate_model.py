import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# 1. Load the dataset
df = pd.read_csv("/data/2023_spotify_ds1.csv")

# 2. Group songs by playlist id using the entire dataset
transactions = df.groupby("pid")["track_name"].apply(list).tolist()

# 3. Transform transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

# 4. Extract frequent itemsets using the Apriori algorithm
# Lowering min_support to 0.03 to include less frequent items
frequent_itemsets = apriori(df_trans, min_support=0.04, use_colnames=True)

# 5. Generate association rules with a minimum confidence threshold of 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[rules['lift'] > 1]  # Optionally filter by lift

# Sort the rules by lift in descending order and return the top 10 (for inspection)
top_rules = rules.sort_values(by='lift', ascending=False).head(10)

# 6. Persist the generated model (association rules)
# Use the mounted shared volume path (e.g., /shared) instead of a relative path
with open("/shared/model_full.pickle", "wb") as f:
    pickle.dump(rules, f)

print(top_rules)
