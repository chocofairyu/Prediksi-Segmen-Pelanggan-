import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

df = pd.read_csv("marketing_campaign.csv", sep="\t")
df["Income"] = df["Income"].fillna(df["Income"].median())
df = df[df["Income"] <= 200000]

df["TotalMnt"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)
df["TotalPurchases"] = (
    df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
)

features = ["Income", "Recency", "TotalMnt", "TotalPurchases", "NumDealsPurchases", "NumWebVisitsMonth"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["KMeans_Cluster"] = labels

sil = silhouette_score(X_scaled, labels)
dbi = davies_bouldin_score(X_scaled, labels)
print("best_k:", best_k, "silhouette:", sil, "dbi:", dbi)

joblib.dump(scaler, "scaler_pelanggan.joblib")
joblib.dump(kmeans, "kmeans_pelanggan.joblib")

cluster_profile = df.groupby("KMeans_Cluster")[features].mean().round(2)
cluster_profile["Jumlah_Pelanggan"] = df["KMeans_Cluster"].value_counts().sort_index()
cluster_profile.to_csv("cluster_profile.csv")
print(cluster_profile)
