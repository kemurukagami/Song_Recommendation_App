import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import os

# 📌 Read dataset path from environment variable (set by Kubernetes ConfigMap)
DATASET_FILE = os.getenv("DATASET_FILE", "/app/shared/data/2023_spotify_ds1.csv")  # Default to ds1 if missing

# Check if running inside Docker
IN_DOCKER = os.path.exists("/app")

# 🔹 Define paths (Updated to use `/app/shared/data/` as dataset path)
DATA_DIR = "/app/shared/data" if IN_DOCKER else "./data"
MODEL_PATH = "/app/shared/model/model.pickle" if IN_DOCKER else "./model/model_full.pickle"

# Ensure directories exist (only needed locally)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 🔹 Load dataset (from Kubernetes ConfigMap path)
print(f"Loading dataset from: {DATASET_FILE}")

# Ensure dataset file exists
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

df = pd.read_csv(DATASET_FILE)

# 🔹 Group songs by playlist ID using the entire dataset
transactions = df.groupby("pid")["track_name"].apply(list).tolist()

# 🔹 Transform transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

# 🔹 Extract frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df_trans, min_support=0.04, use_colnames=True)

# 🔹 Generate association rules with minimum confidence threshold of 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[rules['lift'] > 1]  # Optionally filter by lift

# Sort the rules by lift in descending order (top 10 for inspection)
top_rules = rules.sort_values(by='lift', ascending=False).head(10)

# 🔹 Persist the generated model (association rules)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(rules, f)

print(f"Model saved at: {MODEL_PATH}")
print(top_rules)
