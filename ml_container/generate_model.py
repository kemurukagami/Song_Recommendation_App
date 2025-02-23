import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import os

# Check if running inside Docker (Docker sets the `container` environment variable)
IN_DOCKER = os.path.exists("/app")

# Use appropriate path
DATA_DIR = "/app/data" if IN_DOCKER else "./data"
MODEL_PATH = "/app/model/model.pickle" if IN_DOCKER else "./model/model_full.pickle"

# Ensure directories exist (only needed locally)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load dataset
df = pd.read_csv(f"{DATA_DIR}/2023_spotify_ds1.csv")

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
with open(MODEL_PATH, "wb") as f:
    pickle.dump(rules, f)

print(top_rules)
