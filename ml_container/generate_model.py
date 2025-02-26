import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import os

DATASET_FILE = os.getenv("DATASET_FILE", "/app/shared/data/2023_spotify_ds1.csv")  # Default to ds1 if missing

IN_DOCKER = os.path.exists("/app")

DATA_DIR = "/app/shared/data" if IN_DOCKER else "./data"
MODEL_PATH = "/app/shared/model/model.pickle" if IN_DOCKER else "./model/model_full.pickle"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print(f"Loading dataset from: {DATASET_FILE}")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

df = pd.read_csv(DATASET_FILE)

transactions = df.groupby("pid")["track_name"].apply(list).tolist()

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_trans, min_support=0.03, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

top_rules = rules.sort_values(by='lift', ascending=False).head(10)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(rules, f)

print(f"Model generated from: {DATASET_FILE}")
print(f"Model saved at: {MODEL_PATH}")
print(top_rules)
