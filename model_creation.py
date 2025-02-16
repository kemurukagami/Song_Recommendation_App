import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# 1. Load the dataset
# Assumes the CSV has columns "playlist_id" and "song"
df = pd.read_csv("2023_spotify_ds1.csv")

# 2. Group songs by playlist id using the entire dataset
transactions = df.groupby("pid")["track_name"].apply(list).tolist()

# 3. Transform transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

# 4. Extract frequent itemsets using the Apriori algorithm
# Adjust min_support as needed; a higher threshold may reduce memory usage
frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)

# 5. Generate association rules with a minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[rules['lift'] > 1]  # Optionally filter by lift

# 6. Persist the generated model (association rules)
with open("model_full.pickle", "wb") as f:
    pickle.dump(rules, f)

print(rules.head())