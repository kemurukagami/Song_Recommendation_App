import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# 1. Load the dataset
# Assumes the CSV has columns "playlist_id" and "song"
df = pd.read_csv("/home/datasets/spotify/2023_spotify_ds1.csv")

# For testing: use a subset of the data (e.g., first 5000 playlists)
sample_pids = df['pid'].unique()[:5000]
df_sample = df[df['pid'].isin(sample_pids)]

# 2. Group songs by playlist id
transactions = df_sample.groupby("pid")["track_name"].apply(list).tolist()

# 3. Transform transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

# 4. Increase min_support to 0.05 (adjust as needed) to reduce memory usage
frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)

# 5. Generate association rules with a minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[rules['lift'] > 1]  # Optionally filter by lift

# 6. Persist the generated model (association rules)
with open("model_sample.pickle", "wb") as f:
    pickle.dump(rules, f)

print(rules.head())