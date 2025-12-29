import numpy as np
from sklearn.cluster import KMeans
import psycopg2

def load_embeddings_from_db(db_config):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("SELECT user_id, embedding FROM grievances WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    user_ids = [r[0] for r in rows]
    embeddings = [np.array(r[1]) for r in rows]
    return user_ids, np.vstack(embeddings) if embeddings else None

def cluster_embeddings(embeddings, n_clusters=5):
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings available for clustering.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

def assign_clusters_to_db(user_ids, clusters, db_config):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    for user_id, cluster_label in zip(user_ids, clusters):
        cur.execute('UPDATE grievances SET cluster_id = %s WHERE user_id = %s', (int(cluster_label), int(user_id)))
    conn.commit()
    cur.close()
    conn.close()
