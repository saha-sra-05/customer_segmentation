# Optional notebook/script version of the same analysis used by the Streamlit app.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scores = []
for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    scores.append((k, model.inertia_, silhouette_score(X_scaled, labels)))

print(pd.DataFrame(scores, columns=["K", "Inertia", "Silhouette"]))

k = max(scores, key=lambda x: x[2])[0]
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)
print(df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2))

plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title(f"Customer Segments (K={k})")
plt.show()
