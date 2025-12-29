# # Prototype Code for AI Grievance Management System Workflow
# # This is a simplified, self-contained prototype using free tools: SBERT for embeddings, HDBSCAN for clustering,
# # PostgreSQL with pgVector for storage, FastAPI for backend, and Streamlit for frontend/dashboard.
# # Assumptions: PostgreSQL with pgVector extension installed locally. Run with dummy data for testing.
# # Requirements: Install via pip: sentence-transformers, hdbscan, psycopg2-binary, pandas, numpy, plotly, fastapi, uvicorn, streamlit, geopy
# # Usage: Run ` main:uvicornpp --reload` for backend, and `streamlit run frontend.py` for dashboard (separate file).

# # main.py (Backend with FastAPI)
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import psycopg2
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import hdbscan
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# from geopy.geocoders import Nominatim
# from typing import List, Dict
# import uvicorn

# app = FastAPI()

# # SBERT Model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Database Config (Replace with your details; assumes pgVector installed)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': '5432'
# }

# # Connect to DB
# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Create table if not exists
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             id SERIAL PRIMARY KEY,
#             user_id TEXT,
#             complaint_text TEXT,
#             timestamp TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             cluster_danger INT,
#             cluster_population INT,
#             cluster_socioeconomic INT,
#             score_danger FLOAT,
#             score_population FLOAT,
#             score_socioeconomic FLOAT,
#             final_score FLOAT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # A. Submit Complaint
# class Complaint(BaseModel):
#     user_id: str
#     complaint_text: str
#     timestamp: str
#     latitude: float = None
#     longitude: float = None

# @app.post("/submit")
# def submit_complaint(complaint: Complaint):
#     embedding = model.encode(complaint.complaint_text).tolist()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (user_id, complaint_text, timestamp, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
#     ''', (complaint.user_id, complaint.complaint_text, complaint.timestamp, complaint.latitude, complaint.longitude, embedding))
#     gr_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"id": gr_id, "message": "Complaint submitted"}

# # B. Clustering Job (Run manually or schedule)
# @app.post("/run_clustering")
# def run_clustering():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT id, embedding, latitude, longitude FROM grievances;")
#     rows = cur.fetchall()
#     if not rows:
#         return {"message": "No data to cluster"}
    
#     df = pd.DataFrame(rows, columns=['id', 'embedding', 'lat', 'lon'])
#     embeddings = np.array(df['embedding'].tolist())
    
#     # Criterion: Danger (embedding clustering)
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
#     danger_labels = clusterer.fit_predict(embeddings)
#     scaler = MinMaxScaler()
#     danger_scores = scaler.fit_transform(danger_labels.reshape(-1, 1)).flatten() * 10  # Scale to 0-10
    
#     # Criterion: Population (simple geo clustering with DBSCAN)
#     from sklearn.cluster import DBSCAN
#     geo_coords = df[['lat', 'lon']].dropna().values
#     if len(geo_coords) > 0:
#         geo_clusterer = DBSCAN(eps=0.5, min_samples=2)
#         pop_labels = geo_clusterer.fit_predict(geo_coords)
#         pop_scores = scaler.fit_transform(pop_labels.reshape(-1, 1)).flatten() * 10
#     else:
#         pop_scores = np.zeros(len(df))
    
#     # Criterion: Socioeconomic (dummy; replace with real data lookup)
#     socio_scores = np.random.uniform(0, 10, len(df))  # Placeholder
    
#     # Update DB
#     for i, row in df.iterrows():
#         cur.execute('''
#             UPDATE grievances 
#             SET cluster_danger = %s, score_danger = %s,
#                 cluster_population = %s, score_population = %s,
#                 cluster_socioeconomic = %s, score_socioeconomic = %s
#             WHERE id = %s;
#         ''', (int(danger_labels[i]), danger_scores[i], int(pop_labels[i] if i < len(pop_labels) else -1), pop_scores[i] if i < len(pop_scores) else 0, -1, socio_scores[i], row['id']))
    
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Clustering completed"}

# # C. Real-time Query and Scoring
# @app.get("/issues")
# def get_issues(state: str = None, country: str = None):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT * FROM grievances;")
#     rows = cur.fetchall()
#     columns = [desc[0] for desc in cur.description]
#     df = pd.DataFrame(rows, columns=columns)
#     cur.close()
#     conn.close()
    
#     # Filter by state/country using Geopy
#     geolocator = Nominatim(user_agent="grievance_app")
#     if state or country:
#         df['location'] = df.apply(lambda row: geolocator.reverse((row['latitude'], row['longitude'])) if row['latitude'] else None, axis=1)
#         df = df[df['location'].str.contains(state or '', na=False) & df['location'].str.contains(country or '', na=False)]
    
#     # Compute final score (weighted)
#     weights = {'danger': 0.4, 'population': 0.3, 'socioeconomic': 0.3}
#     df['final_score'] = (df['score_danger'] * weights['danger'] +
#                          df['score_population'] * weights['population'] +
#                          df['score_socioeconomic'] * weights['socioeconomic'])
    
#     # Sort descending
#     df = df.sort_values('final_score', ascending=False)
#     return df.to_dict(orient='records')

# # Run backend
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py - FastAPI backend
# This file handles the server-side logic: complaint submission, geolocation, embedding generation, and data retrieval.
# Run with: uvicorn main:app --reload (assuming FastAPI and dependencies are installed).

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     # Get latitude/longitude from location
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     # Generate embedding
#     embedding = model.encode(c.complaint_text).tolist()
    
#     # Auto-generate timestamp
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Endpoint to get all complaints (for dashboard)
# @app.get('/get_complaints')
# def get_complaints():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     complaints = [
#         {
#             "user_id": row[0],
#             "name": row[1],
#             "complaint_text": row[2],
#             "timestamp": row[3].isoformat(),
#             "city": row[4],
#             "state": row[5],
#             "country": row[6],
#             "latitude": row[7],
#             "longitude": row[8]
#         } for row in rows
#     ]
#     return {"complaints": complaints}

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# main.py - FastAPI backend with simulated geocoding workaround
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# import psycopg2
# import uvicorn

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Simulated geocoding (workaround for 403 error; replace with real geopy in production)
# def get_coordinates(city: str, state: str, country: str):
#     # Hardcoded dictionary for testing (add more as needed)
#     known_locations = {
#         ("hyderabad", "telangana", "india"): (17.385044, 78.486671),
#         ("mumbai", "maharashtra", "india"): (19.076090, 72.877426),
#         ("delhi", "delhi", "india"): (28.613939, 77.209023),
#         # Add other locations here
#     }
#     key = (city.lower(), state.lower(), country.lower())
#     if key in known_locations:
#         return known_locations[key]
#     else:
#         raise HTTPException(status_code=400, detail="Location not found in simulation (add to dictionary or use real geocoding)")

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     latitude, longitude = get_coordinates(c.city, c.state, c.country)
    
#     # Generate embedding
#     embedding = model.encode(c.complaint_text).tolist()
    
#     # Auto-generate timestamp
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Endpoint to get all complaints (for dashboard)
# @app.get('/get_complaints')
# def get_complaints():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     complaints = [
#         {
#             "user_id": row[0],
#             "name": row[1],
#             "complaint_text": row[2],
#             "timestamp": row[3].isoformat(),
#             "city": row[4],
#             "state": row[5],
#             "country": row[6],
#             "latitude": row[7],
#             "longitude": row[8]
#         } for row in rows
#     ]
#     return {"complaints": complaints}

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# main.py - FastAPI backend
# Updated to use 'all-mpnet-base-v2' for 768-dimensional embeddings to match VECTOR(768)
# This resolves the "expected 768 dimensions, not 384" error

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn

# app = FastAPI()

# # Load SBERT model (changed to 'all-mpnet-base-v2' for 768 dims)
# model = SentenceTransformer('all-mpnet-base-v2')

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     # Get latitude/longitude from location
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     # Generate embedding (now 768 dims)
#     embedding = model.encode(c.complaint_text).tolist()
    
#     # Auto-generate timestamp
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Endpoint to get all complaints (for dashboard)
# @app.get('/get_complaints')
# def get_complaints():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     complaints = [
#         {
#             "user_id": row[0],
#             "name": row[1],
#             "complaint_text": row[2],
#             "timestamp": row[3].isoformat(),
#             "city": row[4],
#             "state": row[5],
#             "country": row[6],
#             "latitude": row[7],
#             "longitude": row[8]
#         } for row in rows
#     ]
#     return {"complaints": complaints}

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# main.py - FastAPI backend with simulated geocoding to avoid 403 errors
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import time  # For rate limiting

# app = FastAPI()

# # Load SBERT model (using 768-dim model to match VECTOR(768))
# model = SentenceTransformer('all-mpnet-base-v2')

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Simulated geocoding dictionary (expand with more locations)
# KNOWN_LOCATIONS = {
#     ("hyderabad", "telangana", "india"): (17.385044, 78.486671),
#     ("mumbai", "maharashtra", "india"): (19.076090, 72.877426),
#     ("delhi", "delhi", "india"): (28.613939, 77.209023),
#     ("bangalore", "karnataka", "india"): (12.971599, 77.594566),
#     # Add more as needed
# }

# def get_coordinates(city: str, state: str, country: str, use_real: bool = False):
#     key = (city.lower(), state.lower(), country.lower())
#     if use_real:
#         time.sleep(1)  # Rate limit to avoid 403
#         geolocator = Nominatim(user_agent="grievance_app_unique_user_agent")  # Unique User-Agent
#         location = geolocator.geocode(f"{city}, {state}, {country}")
#         if location:
#             return location.latitude, location.longitude
#         else:
#             raise HTTPException(status_code=400, detail="Invalid location (real geocoding failed)")
#     else:
#         # Simulation mode
#         if key in KNOWN_LOCATIONS:
#             return KNOWN_LOCATIONS[key]
#         else:
#             raise HTTPException(status_code=400, detail="Location not found in simulation (add to dictionary or enable real mode)")

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Endpoint to submit complaint (use simulation by default; set use_real=True for production)
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     latitude, longitude = get_coordinates(c.city, c.state, c.country, use_real=False)  # Change to True for real geocoding
    
#     # Generate embedding
#     embedding = model.encode(c.complaint_text).tolist()
    
#     # Auto-generate timestamp
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Endpoint to get all complaints (for dashboard)
# @app.get('/get_complaints')
# def get_complaints():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     complaints = [
#         {
#             "user_id": row[0],
#             "name": row[1],
#             "complaint_text": row[2],
#             "timestamp": row[3].isoformat(),
#             "city": row[4],
#             "state": row[5],
#             "country": row[6],
#             "latitude": row[7],
#             "longitude": row[8]
#         } for row in rows
#     ]
#     return {"complaints": complaints}

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# main.py - FastAPI backend with multi-metric ranking
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from geopy.distance import geodesic

# app = FastAPI()

# # Load SBERT model
# model = SentenceTransformer('all-mpnet-base-v2')

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Endpoint to get complaints with rankings by metric
# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, complaint_text, timestamp, latitude, longitude, embedding
#         FROM grievances;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['id', 'text', 'timestamp', 'lat', 'lon', 'embedding'])
#     df['embedding'] = df['embedding'].apply(np.array)  # Convert list to array
    
#     # Compute scores based on metric
#     if metric == "semantic":
#         embeddings = np.vstack(df['embedding'])
#         sim_matrix = cosine_similarity(embeddings)
#         scores = [(np.sum(row) - 1) / (len(row)-1) for row in sim_matrix]
#     elif metric == "location":
#         def loc_sim(c1, c2):
#             coords1 = (c1['lat'], c1['lon'])
#             coords2 = (c2['lat'], c2['lon'])
#             dist = geodesic(coords1, coords2).km
#             return max(0, 1 - dist/10)
#         sim_matrix = np.array([[loc_sim(df.iloc[i], df.iloc[j]) for j in range(len(df))] for i in range(len(df))])
#         scores = [(np.sum(row) - 1) / (len(row)-1) for row in sim_matrix]
#     elif metric == "keyword":
#         keywords = ["road", "traffic", "pothole", "sinkhole", "rain", "street", "light"]
#         def kw_count(text):
#             return sum(text.lower().count(k) for k in keywords)
#         counts = df['text'].apply(kw_count)
#         max_cnt = counts.max() if counts.max() > 0 else 1
#         scores = counts / max_cnt
#     elif metric == "temporal":
#         window_hours = 6
#         scores = []
#         for i, ts1 in enumerate(df['timestamp']):
#             count = sum(abs((ts1 - ts2).total_seconds() / 3600) <= window_hours for ts2 in df['timestamp'])
#             scores.append(count / len(df))
#     else:
#         raise HTTPException(status_code=400, detail="Invalid metric")
    
#     df['score'] = scores
#     ranked_df = df.sort_values(by='score', ascending=False)
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# main.py - FastAPI backend with category-based urgency ranking
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd

# app = FastAPI()

# # Load SBERT model (optional for future semantic features)
# model = SentenceTransformer('all-mpnet-base-v2')

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}


# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accident', 'transport'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []  
# }

# URGENCY_MAP = {
#     'emergency': 10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'road': 4, 'area': 3
# }


# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]  
#     return max(scores) if scores else 0

# # Endpoint to get ranked complaints by metric or overall
# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     # Compute scores for each category
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    
#     # 'Other' as max of unassigned or low scores
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    
#     # Overall urgency (average of category scores)
#     category_cols = list(CATEGORIES.keys()) + ['Other']
#     df['overall'] = df[category_cols].mean(axis=1)
    
#     # Sort by selected metric or overall
#     if metric in category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='overall', ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid metric")
    
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# main.py - FastAPI backend with fine-tuned BERT for multi-class complaint classification
# Assumes you have a fine-tuned BERT model saved in './finetuned_bert' (see fine-tuning code below)
# For this prototype, I'll include a placeholder; replace with actual fine-tuned model loading

# main.py - FastAPI backend with Photon geocoder to avoid 403 errors
# Uses fine-tuned BERT for classification (load from local './finetuned_bert')

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from geopy.geocoders import Photon
# import time  # For rate limiting
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline

# app = FastAPI()

# # Load fine-tuned BERT classifier (assumes './finetuned_bert' exists with model files)
# classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Categories (match fine-tuning labels)
# CATEGORIES = ['location', 'medical and health care', 'environment', 'road transport', 'public safety', 'crime', 'cyber crime', 'other']

# # Database configuration (update with your details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Photon geocoder with user-agent
# geolocator = Photon(user_agent="grievance_app/1.0 (ashlinjoel30sizzin@gmail.com)")

# # Endpoint to submit complaint (with rate limiting and Photon)
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     time.sleep(1)  # Rate limit to 1 request/second
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Endpoint to get ranked complaints by metric or overall
# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     # Classify using fine-tuned BERT
#     predictions = classifier(df['text'].tolist())
#     df['category'] = [pred['label'] for pred in predictions]
#     df['urgency'] = [pred['score'] * 10 for pred in predictions]  # Scale confidence to urgency 0-10
    
#     # For overall, sort by urgency
#     if metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     else:
#         # Filter by category and sort by urgency
#         ranked_df = df[df['category'] == metric].sort_values(by='urgency', ascending=False)
    
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time  # For rate limiting

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories, keywords, and urgency map
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accident', 'transport'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []  # Catch-all
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'road': 4, 'area': 3
#     # Add more as needed
# }

# # Function to compute urgency for a category
# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]  # Default 3 if not in map
#     return max(scores) if scores else 0

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     time.sleep(1)  # Rate limit to avoid 403
#     geolocator = Nominatim(user_agent="grievance_app/1.0 (your.email@example.com)")  # Unique user-agent
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     # --- START OF CHANGES ---

#     # Create sanitized, URL-friendly keys for categories
#     sanitized_categories = {key.lower().replace(' ', '_'): val for key, val in CATEGORIES.items()}

#     # Compute keyword-based scores using the original keys
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    
#     # 'Other' as default if no matches
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    
#     # NEW: Explicitly create the 'keyword_category' column for the frontend
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     # BERT classification to enhance accuracy
#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]  # Scale to 0-10
    
#     # Combine keyword and BERT
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row[row['final_category']] + row['bert_score']) / 2, axis=1)
    
#     # Overall urgency (average of all category scores)
#     category_cols = list(CATEGORIES.keys()) + ['Other']
#     df['overall'] = df[category_cols].mean(axis=1)
    
#     # Now, rename the original category columns to the sanitized versions for sorting
#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     # Sort by selected metric or overall using sanitized keys
#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']
#     if metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='overall', ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")
    
#     # --- END OF CHANGES ---
    
#     return ranked_df.to_dict(orient='records')


# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import threading

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Database configuration (update with your actual details)
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768)
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories, keywords, and urgency map
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accident', 'transport'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacking', 'phishing', 'malware', 'data breach', 'hacked'],
#     'Other': []  # Catch-all
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'road': 4, 'area': 3
#     # Add more as needed
# }

# # Function to compute urgency for a category
# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]  # Default 3 if not in map
#     return max(scores) if scores else 0

# # Endpoint to submit complaint
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     time.sleep(1)  # Rate limit to avoid 403
#     geolocator = Nominatim(user_agent="grievance_app/1.0 (your.email@example.com)")  # Unique user-agent
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if location is None:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     latitude, longitude = location.latitude, location.longitude
    
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (c.name, c.complaint_text, timestamp, c.city, c.state, c.country, latitude, longitude, embedding))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # Integrated SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         self.weights = {
#             'safety_risk': 0.30,
#             'population_impact': 0.25,
#             'community_validation': 0.20,
#             'economic_impact': 0.15,
#             'environmental_impact': 0.05,
#             'accessibility_impact': 0.05
#         }

#         self.safety_multipliers = {
#             'school_zone': 1.5,
#             'hospital_area': 1.4,
#             'major_road': 1.3,
#             'residential_high_density': 1.2,
#             'commercial_area': 1.1
#         }

#     def calculate_safety_risk(self, complaint_data):
#         description = complaint_data.get('complaint_text', '').lower()
#         location_type = ''  # You can map from city/state if needed
#         base_score = 0.0

#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }

#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)

#         multiplier = self.safety_multipliers.get(location_type, 1.0)
#         return min(1.0, base_score * multiplier)

#     def calculate_population_impact(self, complaint_data):
#         direct_affected = 0  # Extend DB if needed for real data
#         daily_commuters = 0
#         nearby_facilities = []

#         if direct_affected > 0:
#             base_score = min(1.0, np.log10(direct_affected + 1) / 4)
#         else:
#             base_score = 0.1

#         if daily_commuters > 100:
#             commuter_bonus = min(0.3, daily_commuters / 10000)
#             base_score += commuter_bonus

#         facility_bonus = len(nearby_facilities) * 0.1
#         base_score += facility_bonus

#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = 0
#         photo_evidence = 0
#         similar_reports = 0
#         expert_validation = False

#         confirmation_score = min(0.6, confirmations / 50)
#         photo_bonus = min(0.2, photo_evidence / 10)
#         reports_bonus = min(0.15, similar_reports / 20)
#         expert_bonus = 0.25 if expert_validation else 0.0

#         total_score = confirmation_score + photo_bonus + reports_bonus + expert_bonus
#         return min(1.0, total_score)

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = 0
#         business_disruption = 0
#         repair_urgency = 30

#         if estimated_damage_cost > 0:
#             cost_score = min(0.5, np.log10(estimated_damage_cost + 1) / 10)
#         else:
#             cost_score = 0.0

#         disruption_score = min(0.3, business_disruption / 10)
#         urgency_score = min(0.2, (30 - repair_urgency) / 30) if repair_urgency <= 30 else 0

#         return min(1.0, cost_score + disruption_score + urgency_score)

#     def calculate_environmental_impact(self, complaint_data):
#         description = complaint_data.get('complaint_text', '').lower()

#         environmental_terms = {
#             'sewage overflow': 0.8, 'water contamination': 0.9, 'toxic spill': 1.0,
#             'air pollution': 0.6, 'noise pollution': 0.3, 'waste overflow': 0.5,
#             'drainage blockage': 0.4
#         }

#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)

#         return max_score

#     def calculate_accessibility_impact(self, complaint_data):
#         description = complaint_data.get('complaint_text', '').lower()

#         accessibility_terms = {
#             'wheelchair access': 0.8, 'ramp broken': 0.7, 'elevator out': 0.9,
#             'sidewalk blocked': 0.6, 'tactile paving': 0.5, 'handrail missing': 0.4
#         }

#         max_score = 0.0
#         for term, score in accessibility_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)

#         return max_score

#     def calculate_overall_severity(self, complaint_data):
#         try:
#             safety_score = self.calculate_safety_risk(complaint_data)
#             population_score = self.calculate_population_impact(complaint_data)
#             community_score = self.calculate_community_validation(complaint_data)
#             economic_score = self.calculate_economic_impact(complaint_data)
#             environmental_score = self.calculate_environmental_impact(complaint_data)
#             accessibility_score = self.calculate_accessibility_impact(complaint_data)

#             overall_severity = (
#                 self.weights['safety_risk'] * safety_score +
#                 self.weights['population_impact'] * population_score +
#                 self.weights['community_validation'] * community_score +
#                 self.weights['economic_impact'] * economic_score +
#                 self.weights['environmental_impact'] * environmental_score +
#                 self.weights['accessibility_impact'] * accessibility_score
#             )

#             if overall_severity >= 0.8:
#                 severity_level = "CRITICAL"
#             elif overall_severity >= 0.6:
#                 severity_level = "HIGH"
#             elif overall_severity >= 0.4:
#                 severity_level = "MEDIUM"
#             else:
#                 severity_level = "LOW"

#             return {
#                 'safety_risk': round(safety_score, 3),
#                 'population_impact': round(population_score, 3),
#                 'community_validation': round(community_score, 3),
#                 'economic_impact': round(economic_score, 3),
#                 'environmental_impact': round(environmental_score, 3),
#                 'accessibility_impact': round(accessibility_score, 3),
#                 'overall_severity': round(overall_severity, 3),
#                 'severity_level': severity_level
#             }
#         except Exception as e:
#             print(f"Severity calculation error: {e}")
#             return {'overall_severity': 0.0, 'severity_level': 'UNKNOWN'}

# severity_calculator = SeverityCalculator()

# @app.on_event("startup")
# def analyze_existing_complaints():
#     def run_analysis():
#         conn = get_db_conn()
#         cur = conn.cursor()
#         cur.execute('''
#             SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#             FROM grievances ORDER BY timestamp DESC;
#         ''')
#         rows = cur.fetchall()
#         cur.close()
#         conn.close()

#         if not rows:
#             print("No existing complaints for severity analysis.")
#             return

#         df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])

#         print("Starting SeverityCalculator analysis on existing complaints...")

#         for idx, row in df.iterrows():
#             complaint_data = row.to_dict()
#             sev_report = severity_calculator.calculate_overall_severity(complaint_data)
#             print(f"UserID {row['user_id']}: Severity Level = {sev_report['severity_level']}, Overall Score = {sev_report['overall_severity']}")
#             print(f"Component Scores: {sev_report}")

#         print("Severity analysis complete.")

#     threading.Thread(target=run_analysis).start()

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude
#         FROM grievances ORDER BY timestamp DESC;
#     ''')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     # --- START OF CHANGES ---

#     # Create sanitized, URL-friendly keys for categories
#     sanitized_categories = {key.lower().replace(' ', '_'): val for key, val in CATEGORIES.items()}

#     # Compute keyword-based scores using the original keys
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    
#     # 'Other' as default if no matches
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    
#     # NEW: Explicitly create the 'keyword_category' column for the frontend
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     # BERT classification to enhance accuracy
#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]  # Scale to 0-10
    
#     # Combine keyword and BERT
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row[row['final_category']] + row['bert_score']) / 2, axis=1)
    
#     # Overall urgency (average of all category scores)
#     category_cols = list(CATEGORIES.keys()) + ['Other']
#     df['overall'] = df[category_cols].mean(axis=1)
    
#     # Now, rename the original category columns to the sanitized versions for sorting
#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     # Sort by selected metric or overall using sanitized keys
#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']
#     if metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='overall', ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")

#     # Compute SeverityCalculator analysis for returned complaints
#     for idx, row in ranked_df.iterrows():
#         complaint_data = row.to_dict()
#         sev_report = severity_calculator.calculate_overall_severity(complaint_data)
#         ranked_df.at[idx, 'safety_risk'] = sev_report['safety_risk']
#         ranked_df.at[idx, 'population_impact'] = sev_report['population_impact']
#         ranked_df.at[idx, 'community_validation'] = sev_report['community_validation']
#         ranked_df.at[idx, 'economic_impact'] = sev_report['economic_impact']
#         ranked_df.at[idx, 'environmental_impact'] = sev_report['environmental_impact']
#         ranked_df.at[idx, 'accessibility_impact'] = sev_report['accessibility_impact']
#         ranked_df.at[idx, 'overall_severity'] = sev_report['overall_severity']
#         ranked_df.at[idx, 'severity_level'] = sev_report['severity_level']

#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Initialize geolocator
# geolocator = Nominatim(user_agent="grievance_app")

# # Database configuration
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     # --- MODIFIED: Removed accessibility_impact column ---
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories and keywords
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accident', 'transport'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'road': 4, 'area': 3
# }

# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0

# # SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         # --- MODIFIED: Removed accessibility_impact weight ---
#         self.weights = {
#             'safety_risk': 0.35,          # Increased weight
#             'population_impact': 0.30,      # Increased weight
#             'community_validation': 0.15, # Decreased weight
#             'economic_impact': 0.10,      # Decreased weight
#             'environmental_impact': 0.10      # Increased weight
#         }

#     # --- MODIFIED: Added 'category' parameter and conditional logic ---
#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']:
#             return 0.9
        
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }
#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)
#         return min(1.0, base_score)

#     # --- MODIFIED: Added 'category' parameter and conditional logic ---
#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']:
#             return 0.9
#         # Default logic remains for other cases
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0

#     # --- MODIFIED: Added 'category' parameter and conditional logic ---
#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)
#         return max_score

#     # --- MODIFIED: Updated to handle new logic and removed accessibility ---
#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)

#         overall_severity = (
#             self.weights['safety_risk'] * safety_score +
#             self.weights['population_impact'] * population_score +
#             self.weights['community_validation'] * community_score +
#             self.weights['economic_impact'] * economic_score +
#             self.weights['environmental_impact'] * environmental_score
#         )

#         if overall_severity >= 0.7:
#             severity_level = "CRITICAL"
#         elif overall_severity >= 0.5:
#             severity_level = "HIGH"
#         elif overall_severity >= 0.3:
#             severity_level = "MEDIUM"
#         else:
#             severity_level = "LOW"

#         return {
#             'safety_risk': round(safety_score, 3),
#             'population_impact': round(population_score, 3),
#             'community_validation': round(community_score, 3),
#             'economic_impact': round(economic_score, 3),
#             'environmental_impact': round(environmental_score, 3),
#             'overall_severity': round(overall_severity, 3),
#             'severity_level': severity_level
#         }

# severity_calculator = SeverityCalculator()

# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
    
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
    
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
    
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
    
#     return {
#         'keyword_category': keyword_category,
#         'bert_category': bert_category,
#         'final_category': final_category,
#         'urgency': urgency,
#         **severity_report
#     }

# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")

#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)

#     conn = get_db_conn()
#     cur = conn.cursor()
#     # --- MODIFIED: Removed accessibility_impact from INSERT statement ---
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact']
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# # --- MODIFIED: Reverted to original on-the-fly calculation logic ---
# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     sanitized_categories = {key.lower().replace(' ', '_').replace('_and_health_care', '_care'): val for key, val in CATEGORIES.items()}

#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
    
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)
    
#     category_cols = list(CATEGORIES.keys()) + ['Other']
#     df['overall'] = df[category_cols].mean(axis=1)
    
#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']
#     if metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='overall', ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")
    
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Initialize geolocator
# geolocator = Nominatim(user_agent="grievance_app")

# # Database configuration
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories and keywords
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'water', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accident', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'hack':4, 'hacking':4, 'hack':4, 'road': 4, 'area': 3
# }

# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0

# # SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         self.weights = {
#             'safety_risk': 0.35,
#             'population_impact': 0.30,
#             'community_validation': 0.15,
#             'economic_impact': 0.10,
#             'environmental_impact': 0.10
#         }

#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }
#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)
#         return min(1.0, base_score)

#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']:
#             return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0

#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)
#         return max_score

#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)

#         overall_severity = (
#             self.weights['safety_risk'] * safety_score +
#             self.weights['population_impact'] * population_score +
#             self.weights['community_validation'] * community_score +
#             self.weights['economic_impact'] * economic_score +
#             self.weights['environmental_impact'] * environmental_score
#         )

#         if overall_severity >= 0.7:
#             severity_level = "CRITICAL"
#         elif overall_severity >= 0.5:
#             severity_level = "HIGH"
#         elif overall_severity >= 0.3:
#             severity_level = "MEDIUM"
#         else:
#             severity_level = "LOW"

#         return {
#             'safety_risk': round(safety_score, 3),
#             'population_impact': round(population_score, 3),
#             'community_validation': round(community_score, 3),
#             'economic_impact': round(economic_score, 3),
#             'environmental_impact': round(environmental_score, 3),
#             'overall_severity': round(overall_severity, 3),
#             'severity_level': severity_level
#         }

# severity_calculator = SeverityCalculator()

# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
    
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
    
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
    
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
    
#     return {
#         'keyword_category': keyword_category,
#         'bert_category': bert_category,
#         'final_category': final_category,
#         'urgency': urgency,
#         **severity_report
#     }

# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")

#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)

#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact']
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     sanitized_categories = {key.lower().replace(' ', '_').replace('_and_health_care', '_care'): val for key, val in CATEGORIES.items()}

#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
    
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)
    
#     category_cols = list(CATEGORIES.keys()) + ['Other']
#     df['overall'] = df[category_cols].mean(axis=1)
    
#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']
    
#     # --- MODIFIED: Added special sorting for the 'location' metric ---
#     if metric == 'location':
#         # Sort by location score first, then by urgency for tie-breaking
#         ranked_df = df.sort_values(by=['location', 'urgency'], ascending=[False, False])
#     elif metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='overall', ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")
    
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Initialize geolocator
# geolocator = Nominatim(user_agent="grievance_app")

# # Database configuration
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories and keywords
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street'],
#     'Medical and Health Care': ['hospital', 'medicine', 'meds', 'health', 'doctor', 'emergency', 'death'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accident', 'transport'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'road': 4, 'area': 3
# }

# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0

# # SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         self.weights = {
#             'safety_risk': 0.35,
#             'population_impact': 0.30,
#             'community_validation': 0.15,
#             'economic_impact': 0.10,
#             'environmental_impact': 0.10
#         }

#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }
#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)
#         return min(1.0, base_score)

#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']:
#             return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0

#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)
#         return max_score

#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)

#         overall_severity = (
#             self.weights['safety_risk'] * safety_score +
#             self.weights['population_impact'] * population_score +
#             self.weights['community_validation'] * community_score +
#             self.weights['economic_impact'] * economic_score +
#             self.weights['environmental_impact'] * environmental_score
#         )

#         if overall_severity >= 0.7:
#             severity_level = "CRITICAL"
#         elif overall_severity >= 0.5:
#             severity_level = "HIGH"
#         elif overall_severity >= 0.3:
#             severity_level = "MEDIUM"
#         else:
#             severity_level = "LOW"

#         return {
#             'safety_risk': round(safety_score, 3),
#             'population_impact': round(population_score, 3),
#             'community_validation': round(community_score, 3),
#             'economic_impact': round(economic_score, 3),
#             'environmental_impact': round(environmental_score, 3),
#             'overall_severity': round(overall_severity, 3),
#             'severity_level': severity_level
#         }

# severity_calculator = SeverityCalculator()

# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
    
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
    
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
    
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
    
#     return {
#         'keyword_category': keyword_category,
#         'bert_category': bert_category,
#         'final_category': final_category,
#         'urgency': urgency,
#         **severity_report
#     }

# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")

#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)

#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact']
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
    
#     if not rows:
#         return {"ranked_complaints": []}
    
#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])
    
#     sanitized_categories = {key.lower().replace(' ', '_').replace('_and_health_care', '_care'): val for key, val in CATEGORIES.items()}

#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
    
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)
    
#     category_cols = list(CATEGORIES.keys()) + ['Other']
#     df['overall'] = df[category_cols].mean(axis=1)
    
#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']
    
#     # --- MODIFIED: Corrected the sorting logic for the 'location' metric ---
#     if metric == 'location':
#         # Sort by city name first, then by urgency within each city
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='overall', ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")
    
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Initialize geolocator
# geolocator = Nominatim(user_agent="grievance_app")

# # Database configuration
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories and keywords
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'water':6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'hack':4, 'hacking':4, 'hack':4, 'road': 4, 'area': 3
# }

# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0

# # SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         self.weights = {
#             'safety_risk': 0.35,
#             'population_impact': 0.30,
#             'community_validation': 0.15,
#             'economic_impact': 0.10,
#             'environmental_impact': 0.10
#         }

#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }
#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)
#         return min(1.0, base_score)

#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']:
#             return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0

#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)
#         return max_score

#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)

#         overall_severity = (
#             self.weights['safety_risk'] * safety_score +
#             self.weights['population_impact'] * population_score +
#             self.weights['community_validation'] * community_score +
#             self.weights['economic_impact'] * economic_score +
#             self.weights['environmental_impact'] * environmental_score
#         )

#         if overall_severity >= 0.7:
#             severity_level = "CRITICAL"
#         elif overall_severity >= 0.5:
#             severity_level = "HIGH"
#         elif overall_severity >= 0.3:
#             severity_level = "MEDIUM"
#         else:
#             severity_level = "LOW"

#         return {
#             'safety_risk': round(safety_score, 3),
#             'population_impact': round(population_score, 3),
#             'community_validation': round(community_score, 3),
#             'economic_impact': round(economic_score, 3),
#             'environmental_impact': round(environmental_score, 3),
#             'overall_severity': round(overall_severity, 3),
#             'severity_level': severity_level
#         }

# severity_calculator = SeverityCalculator()

# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)

#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10

#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2

#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)

#     return {
#         'keyword_category': keyword_category,
#         'bert_category': bert_category,
#         'final_category': final_category,
#         'urgency': urgency,
#         **severity_report
#     }

# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")

#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)

#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact']
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     if not rows:
#         return {"ranked_complaints": []}

#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])

#     sanitized_categories = {key.lower().replace(' ', '_').replace('_and_health_care', '_care'): val for key, val in CATEGORIES.items()}

#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))

#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)

#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]

#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)

#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']

#     if metric == 'location':
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     # --- MODIFIED: 'overall' now sorts by 'urgency' ---
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     elif metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")

#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import json # Import json for pretty printing

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Initialize geolocator
# geolocator = Nominatim(user_agent="grievance_app")

# # Database configuration
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories and keywords
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths'],
#     'Environment': ['pollution', 'waste', 'dumping', 'contamination', 'air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'assault', 'robbery', 'violence'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'accident': 8, 'crime': 7, 'cyber': 7, 'hacking': 7, 'pollution': 6,
#     'hospital': 6, 'theft': 6, 'water':6, 'traffic': 5, 'pothole': 5, 'waste': 4, 'hack':4, 'hacking':4, 'hack':4, 'road': 4, 'area': 3
# }

# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0

# # SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         self.weights = {
#             'safety_risk': 0.35,
#             'population_impact': 0.30,
#             'community_validation': 0.15,
#             'economic_impact': 0.10,
#             'environmental_impact': 0.10
#         }

#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }
#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)
#         return min(1.0, base_score)

#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']:
#             return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0

#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)
#         return max_score

#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)

#         overall_severity = (
#             self.weights['safety_risk'] * safety_score +
#             self.weights['population_impact'] * population_score +
#             self.weights['community_validation'] * community_score +
#             self.weights['economic_impact'] * economic_score +
#             self.weights['environmental_impact'] * environmental_score
#         )

#         if overall_severity >= 0.7:
#             severity_level = "CRITICAL"
#         elif overall_severity >= 0.5:
#             severity_level = "HIGH"
#         elif overall_severity >= 0.3:
#             severity_level = "MEDIUM"
#         else:
#             severity_level = "LOW"

#         return {
#             'safety_risk': round(safety_score, 3),
#             'population_impact': round(population_score, 3),
#             'community_validation': round(community_score, 3),
#             'economic_impact': round(economic_score, 3),
#             'environmental_impact': round(environmental_score, 3),
#             'overall_severity': round(overall_severity, 3),
#             'severity_level': severity_level
#         }

# severity_calculator = SeverityCalculator()

# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)

#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10

#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2

#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)

#     return {
#         'keyword_category': keyword_category,
#         'bert_category': bert_category,
#         'final_category': final_category,
#         'urgency': urgency,
#         **severity_report
#     }

# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")

#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)
    
#     # --- MODIFICATION: Print analysis to the console ---
#     print("\n" + "="*50)
#     print(f"Severity Analysis for Complaint: '{c.complaint_text}'")
#     print("="*50)
#     print(json.dumps(analysis, indent=4))
#     print("="*50 + "\n")
#     # --- END OF MODIFICATION ---

#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact']
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     if not rows:
#         return {"ranked_complaints": []}

#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])

#     sanitized_categories = {key.lower().replace(' ', '_').replace('_and_health_care', '_care'): val for key, val in CATEGORIES.items()}

#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))

#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)

#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]

#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)

#     df = df.rename(columns={orig: san for orig, san in zip(CATEGORIES.keys(), sanitized_categories.keys())})

#     sanitized_category_cols = list(sanitized_categories.keys()) + ['other']

#     if metric == 'location':
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     elif metric in sanitized_category_cols:
#         ranked_df = df.sort_values(by=metric, ascending=False)
#     else:
#         raise HTTPException(status_code=400, detail=f"Invalid metric. Valid metrics are: {sanitized_category_cols + ['overall']}")

#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import json

# app = FastAPI()

# # Load SBERT model for embeddings
# model = SentenceTransformer('all-mpnet-base-v2')

# # Load fine-tuned BERT classifier (replace with your actual model path)
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")

# # Initialize geolocator
# geolocator = Nominatim(user_agent="grievance_app")

# # Database configuration
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin',
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# # Initialize DB table
# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # Input model for complaint
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # Define categories and keywords
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths', 'flu'],
#     'Environment': ['pollution', 'waste', 'dumping', 'smoke', 'tree', 'deforestation', 'trees', 'contaminate','contamination', 'contaminating','air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'criminal','assault', 'robbery', 'violence', 'violent', 'murder', 'murderer'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }

# URGENCY_MAP = {
#     'emergency': 10, 'death': 10, 'deaths':10, 
#     'fire': 9, 'violence':9, 'violent':9, 'murder':9, 'murderer':9,
#     'accident': 8,'pollution': 8, 'deforestation':8, 'assault':8,
#     'trees': 7,'crime': 7,'doctor': 7, 'cyber': 7, 'hacking': 7, 'crime':7, 'theft':7, 'criminal':7, 'robbery':7,
#     'hospital': 6, 'theft': 6, 'water':6, 'data breach':6,
#     'traffic': 5, 'pothole': 5, 'waste': 5, 'contamination':5, 'smoke':5, 'malware':5,
#     'dumping':4,'hack':4, 'hacking':4, 'road': 4, 'phishing':4, 'flu':4,
#     'area': 3
# }

# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0

# # SeverityCalculator class
# class SeverityCalculator:
#     def __init__(self):
#         self.weights = {
#             'safety_risk': 0.35,
#             'population_impact': 0.30,
#             'community_validation': 0.15,
#             'economic_impact': 0.10,
#             'environmental_impact': 0.10
#         }

#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = {
#             'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95,
#             'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8,
#             'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7
#         }
#         for term, score in safety_critical_terms.items():
#             if term in description:
#                 base_score = max(base_score, score)
#         return min(1.0, base_score)

#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']:
#             return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)

#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))

#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0

#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']:
#             return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description:
#                 max_score = max(max_score, score)
#         return max_score

#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)

#         overall_severity = (
#             self.weights['safety_risk'] * safety_score +
#             self.weights['population_impact'] * population_score +
#             self.weights['community_validation'] * community_score +
#             self.weights['economic_impact'] * economic_score +
#             self.weights['environmental_impact'] * environmental_score
#         )

#         if overall_severity >= 0.7:
#             severity_level = "CRITICAL"
#         elif overall_severity >= 0.5:
#             severity_level = "HIGH"
#         elif overall_severity >= 0.3:
#             severity_level = "MEDIUM"
#         else:
#             severity_level = "LOW"

#         return {
#             'safety_risk': round(safety_score, 3),
#             'population_impact': round(population_score, 3),
#             'community_validation': round(community_score, 3),
#             'economic_impact': round(economic_score, 3),
#             'environmental_impact': round(environmental_score, 3),
#             'overall_severity': round(overall_severity, 3),
#             'severity_level': severity_level
#         }

# severity_calculator = SeverityCalculator()

# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)

#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10

#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2

#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)

#     return {
#         'keyword_category': keyword_category,
#         'bert_category': bert_category,
#         'final_category': final_category,
#         'urgency': urgency,
#         **severity_report
#     }

# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")

#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)
    
#     print("\n" + "="*50)
#     print(f"Severity Analysis for Complaint: '{c.complaint_text}'")
#     print("="*50)
#     print(json.dumps(analysis, indent=4))
#     print("="*50 + "\n")

#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact']
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()

#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()

#     if not rows:
#         return []

#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon'])

#     # --- Start of Analysis Block ---
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))

#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)

#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]

#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)
#     # --- End of Analysis Block ---

#     # --- Start of Sorting Logic ---
#     if metric == 'location':
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     else:
#         # --- MODIFICATION: Filter by category, then sort by urgency ---
#         # Find the original category name from the sanitized metric
#         original_category_name = None
#         for cat_name in CATEGORIES.keys():
#             sanitized_name = cat_name.lower().replace(' ', '_').replace('_and_health_care', '_care')
#             if sanitized_name == metric:
#                 original_category_name = cat_name
#                 break
        
#         if original_category_name:
#             # Filter the DataFrame to only include complaints of the chosen category
#             filtered_df = df[df['final_category'] == original_category_name]
#             # Sort the filtered list by urgency
#             ranked_df = filtered_df.sort_values(by='urgency', ascending=False)
#         else:
#             # Fallback for 'other' or if metric is invalid
#             ranked_df = df.sort_values(by='urgency', ascending=False)
            
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# from psycopg2 import sql
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import json

# app = FastAPI()

# # --- Model Loading and Initial Setup ---
# model = SentenceTransformer('all-mpnet-base-v2')
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")
# geolocator = Nominatim(user_agent="grievance_app")

# # --- Database Configuration ---
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin', # Replace with your actual password
#     'host': 'localhost',
#     'port': 5432
# }

# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)

# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT,
#             support_count INT NOT NULL DEFAULT 0
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()

# init_db()

# # --- Pydantic Models ---
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str

# # --- Analysis Constants and Classes (omitted for brevity, same as before) ---
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths', 'flu'],
#     'Environment': ['pollution', 'waste', 'dumping', 'smoke', 'tree', 'deforestation', 'trees', 'contaminate','contamination', 'contaminating','air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'criminal','assault', 'robbery', 'violence', 'violent', 'murder', 'murderer'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }
# URGENCY_MAP = {'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'violence':9, 'violent':9, 'murder':9, 'murderer':9, 'accident': 8,'pollution': 8, 'deforestation':8, 'assault':8, 'trees': 7,'crime': 7,'doctor': 7, 'cyber': 7, 'hacking': 7, 'crime':7, 'theft':7, 'criminal':7, 'robbery':7, 'hospital': 6, 'theft': 6, 'water':6, 'data breach':6, 'traffic': 5, 'pothole': 5, 'waste': 5, 'contamination':5, 'smoke':5, 'malware':5, 'dumping':4,'hack':4, 'hacking':4, 'road': 4, 'phishing':4, 'flu':4, 'area': 3}
# class SeverityCalculator:
#     def __init__(self): self.weights = {'safety_risk': 0.35, 'population_impact': 0.30, 'community_validation': 0.15, 'economic_impact': 0.10, 'environmental_impact': 0.10}
#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = { 'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95, 'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8, 'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7 }
#         for term, score in safety_critical_terms.items():
#             if term in description: base_score = max(base_score, score)
#         return min(1.0, base_score)
#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']: return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)
#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))
#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0
#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description: max_score = max(max_score, score)
#         return max_score
#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)
#         overall_severity = (self.weights['safety_risk'] * safety_score + self.weights['population_impact'] * population_score + self.weights['community_validation'] * community_score + self.weights['economic_impact'] * economic_score + self.weights['environmental_impact'] * environmental_score)
#         if overall_severity >= 0.7: severity_level = "CRITICAL"
#         elif overall_severity >= 0.5: severity_level = "HIGH"
#         elif overall_severity >= 0.3: severity_level = "MEDIUM"
#         else: severity_level = "LOW"
#         return {'safety_risk': round(safety_score, 3), 'population_impact': round(population_score, 3), 'community_validation': round(community_score, 3), 'economic_impact': round(economic_score, 3), 'environmental_impact': round(environmental_score, 3), 'overall_severity': round(overall_severity, 3), 'severity_level': severity_level}

# # --- Analysis Functions ---
# severity_calculator = SeverityCalculator()
# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0
# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
#     return {'keyword_category': keyword_category, 'bert_category': bert_category, 'final_category': final_category, 'urgency': urgency, **severity_report}

# # --- API Endpoints ---
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact, support_count
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact'],
#         0 # Initial support count is 0
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}

# @app.post('/support_complaint/{complaint_id}')
# def support_complaint(complaint_id: int):
#     try:
#         conn = get_db_conn()
#         cur = conn.cursor()
#         cur.execute(
#             sql.SQL("UPDATE grievances SET support_count = support_count + 1 WHERE user_id = %s"),
#             (complaint_id,)
#         )
#         if cur.rowcount == 0:
#             raise HTTPException(status_code=404, detail="Complaint not found")
#         conn.commit()
#         cur.close()
#         conn.close()
#         return {"message": f"Successfully supported complaint {complaint_id}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude, support_count FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     if not rows:
#         return []

#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon', 'support_count'])

#     # Re-run analysis for dynamic ranking
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)
#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)

#     # Sorting logic based on the selected metric from the frontend
#     if metric == 'most_supported':
#         ranked_df = df.sort_values(by='support_count', ascending=False)
#     elif metric == 'location':
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     else:
#         original_category_name = None
#         for cat_name in CATEGORIES.keys():
#             sanitized_name = cat_name.lower().replace(' ', '_').replace('_and_health_care', '_care')
#             if sanitized_name == metric:
#                 original_category_name = cat_name
#                 break
#         if original_category_name:
#             filtered_df = df[df['final_category'] == original_category_name]
#             ranked_df = filtered_df.sort_values(by='urgency', ascending=False)
#         else:
#             ranked_df = df.sort_values(by='urgency', ascending=False)
            
#     return ranked_df.to_dict(orient='records')

# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# from psycopg2 import sql
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import json
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder

# app = FastAPI()


# # --- Model Loading and Initial Setup ---
# model = SentenceTransformer('all-mpnet-base-v2')
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")
# geolocator = Nominatim(user_agent="grievance_app")


# # --- Database Configuration ---
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin', # Replace with your actual password
#     'host': 'localhost',
#     'port': 5432
# }


# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)


# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT,
#             support_count INT NOT NULL DEFAULT 0
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()


# init_db()


# # --- Pydantic Models ---
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str


# # --- Analysis Constants and Classes (omitted for brevity, same as before) ---
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths', 'flu'],
#     'Environment': ['pollution', 'waste', 'dumping', 'smoke', 'tree', 'deforestation', 'trees', 'contaminate','contamination', 'contaminating','air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'criminal','assault', 'robbery', 'violence', 'violent', 'murder', 'murderer'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }
# URGENCY_MAP = {'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'violence':9, 'violent':9, 'murder':9, 'murderer':9, 'accident': 8,'pollution': 8, 'deforestation':8, 'assault':8, 'trees': 7,'crime': 7,'doctor': 7, 'cyber': 7, 'hacking': 7, 'crime':7, 'theft':7, 'criminal':7, 'robbery':7, 'hospital': 6, 'theft': 6, 'water':6, 'data breach':6, 'traffic': 5, 'pothole': 5, 'waste': 5, 'contamination':5, 'smoke':5, 'malware':5, 'dumping':4,'hack':4, 'hacking':4, 'road': 4, 'phishing':4, 'flu':4, 'area': 3}
# class SeverityCalculator:
#     def __init__(self): self.weights = {'safety_risk': 0.35, 'population_impact': 0.30, 'community_validation': 0.15, 'economic_impact': 0.10, 'environmental_impact': 0.10}
#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = { 'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95, 'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8, 'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7 }
#         for term, score in safety_critical_terms.items():
#             if term in description: base_score = max(base_score, score)
#         return min(1.0, base_score)
#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']: return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)
#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))
#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0
#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description: max_score = max(max_score, score)
#         return max_score
#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)
#         overall_severity = (self.weights['safety_risk'] * safety_score + self.weights['population_impact'] * population_score + self.weights['community_validation'] * community_score + self.weights['economic_impact'] * economic_score + self.weights['environmental_impact'] * environmental_score)
#         if overall_severity >= 0.7: severity_level = "CRITICAL"
#         elif overall_severity >= 0.5: severity_level = "HIGH"
#         elif overall_severity >= 0.3: severity_level = "MEDIUM"
#         else: severity_level = "LOW"
#         return {'safety_risk': round(safety_score, 3), 'population_impact': round(population_score, 3), 'community_validation': round(community_score, 3), 'economic_impact': round(economic_score, 3), 'environmental_impact': round(environmental_score, 3), 'overall_severity': round(overall_severity, 3), 'severity_level': severity_level}


# # --- Analysis Functions ---
# severity_calculator = SeverityCalculator()
# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0
# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
#     return {'keyword_category': keyword_category, 'bert_category': bert_category, 'final_category': final_category, 'urgency': urgency, **severity_report}


# # --- API Endpoints ---
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact, support_count
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact'],
#         0 # Initial support count is 0
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}


# @app.post('/support_complaint/{complaint_id}')
# def support_complaint(complaint_id: int):
#     try:
#         conn = get_db_conn()
#         cur = conn.cursor()
#         cur.execute(
#             sql.SQL("UPDATE grievances SET support_count = support_count + 1 WHERE user_id = %s"),
#             (complaint_id,)
#         )
#         if cur.rowcount == 0:
#             raise HTTPException(status_code=404, detail="Complaint not found")
#         conn.commit()
#         cur.close()
#         conn.close()
#         return {"message": f"Successfully supported complaint {complaint_id}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")


# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude, support_count FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     if not rows:
#         return []


#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon', 'support_count'])


#     # Re-run analysis for dynamic ranking
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)
#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)


#     # Impute missing values in categorical columns with 'Unknown'
#     categorical_cols = ['city', 'state', 'country', 'final_category']  # Adjust as needed
#     imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
#     df[categorical_cols] = pd.DataFrame(imputer.fit_transform(df[categorical_cols]), columns=categorical_cols)

#     # One-hot encode the imputed categorical columns
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     encoded = encoder.fit_transform(df[categorical_cols])
#     encoded_cols = encoder.get_feature_names_out(categorical_cols)
#     encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

#     # Print the one-hot encoded data to console for all complaints
#     print("One-Hot Encoded Categorical Data for Complaints:")
#     print(encoded_df.to_string(index=False))
#     print('-' * 80)  # Separator

#     # Concatenate encoded features back to df (drop original categoricals if not needed, but keep for frontend)
#     df = pd.concat([df, encoded_df], axis=1)  # Keep original cols for frontend display

#     # Sorting logic based on the selected metric from the frontend
#     if metric == 'most_supported':
#         ranked_df = df.sort_values(by='support_count', ascending=False)
#     elif metric == 'location':
#         # Since 'city' is still in df, sort by 'city' and 'urgency'
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     else:
#         original_category_name = None
#         for cat_name in CATEGORIES.keys():
#             sanitized_name = cat_name.lower().replace(' ', '_').replace('_and_health_care', '_care')
#             if sanitized_name == metric:
#                 original_category_name = cat_name
#                 break
#         if original_category_name:
#             filtered_df = df[df['final_category'] == original_category_name]
#             ranked_df = filtered_df.sort_values(by='urgency', ascending=False)
#         else:
#             ranked_df = df.sort_values(by='urgency', ascending=False)
            
#     return ranked_df.to_dict(orient='records')


# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# from psycopg2 import sql
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import json
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# import re  # For text processing

# app = FastAPI()


# # --- Model Loading and Initial Setup ---
# model = SentenceTransformer('all-mpnet-base-v2')
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")
# geolocator = Nominatim(user_agent="grievance_app")


# # --- Database Configuration ---
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin', # Replace with your actual password
#     'host': 'localhost',
#     'port': 5432
# }


# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)


# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT,
#             support_count INT NOT NULL DEFAULT 0
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()


# init_db()


# # --- Pydantic Models ---
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str


# # --- Analysis Constants and Classes (omitted for brevity, same as before) ---
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths', 'flu'],
#     'Environment': ['pollution', 'waste', 'dumping', 'smoke', 'tree', 'deforestation', 'trees', 'contaminate','contamination', 'contaminating','air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'criminal','assault', 'robbery', 'violence', 'violent', 'murder', 'murderer'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }
# URGENCY_MAP = {'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'violence':9, 'violent':9, 'murder':9, 'murderer':9, 'accident': 8,'pollution': 8, 'deforestation':8, 'assault':8, 'trees': 7,'crime': 7,'doctor': 7, 'cyber': 7, 'hacking': 7, 'crime':7, 'theft':7, 'criminal':7, 'robbery':7, 'hospital': 6, 'theft': 6, 'water':6, 'data breach':6, 'traffic': 5, 'pothole': 5, 'waste': 5, 'contamination':5, 'smoke':5, 'malware':5, 'dumping':4,'hack':4, 'hacking':4, 'road': 4, 'phishing':4, 'flu':4, 'area': 3}
# class SeverityCalculator:
#     def __init__(self): self.weights = {'safety_risk': 0.35, 'population_impact': 0.30, 'community_validation': 0.15, 'economic_impact': 0.10, 'environmental_impact': 0.10}
#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = { 'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95, 'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8, 'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7 }
#         for term, score in safety_critical_terms.items():
#             if term in description: base_score = max(base_score, score)
#         return min(1.0, base_score)
#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']: return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)
#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))
#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0
#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description: max_score = max(max_score, score)
#         return max_score
#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)
#         overall_severity = (self.weights['safety_risk'] * safety_score + self.weights['population_impact'] * population_score + self.weights['community_validation'] * community_score + self.weights['economic_impact'] * economic_score + self.weights['environmental_impact'] * environmental_score)
#         if overall_severity >= 0.7: severity_level = "CRITICAL"
#         elif overall_severity >= 0.5: severity_level = "HIGH"
#         elif overall_severity >= 0.3: severity_level = "MEDIUM"
#         else: severity_level = "LOW"
#         return {'safety_risk': round(safety_score, 3), 'population_impact': round(population_score, 3), 'community_validation': round(community_score, 3), 'economic_impact': round(economic_score, 3), 'environmental_impact': round(environmental_score, 3), 'overall_severity': round(overall_severity, 3), 'severity_level': severity_level}


# # --- Analysis Functions ---
# severity_calculator = SeverityCalculator()
# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0
# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
#     return {'keyword_category': keyword_category, 'bert_category': bert_category, 'final_category': final_category, 'urgency': urgency, **severity_report}


# # --- API Endpoints ---
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact, support_count
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact'],
#         0 # Initial support count is 0
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}


# @app.post('/support_complaint/{complaint_id}')
# def support_complaint(complaint_id: int):
#     try:
#         conn = get_db_conn()
#         cur = conn.cursor()
#         cur.execute(
#             sql.SQL("UPDATE grievances SET support_count = support_count + 1 WHERE user_id = %s"),
#             (complaint_id,)
#         )
#         if cur.rowcount == 0:
#             raise HTTPException(status_code=404, detail="Complaint not found")
#         conn.commit()
#         cur.close()
#         conn.close()
#         return {"message": f"Successfully supported complaint {complaint_id}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude, support_count FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     if not rows:
#         return []


#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon', 'support_count'])


#     # Re-run analysis for dynamic ranking (your existing code here)
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)
#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)


#     # Impute missing values in categorical columns with 'Unknown'
#     categorical_cols = ['city', 'state', 'country', 'final_category']  # Adjust as needed
#     imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
#     df[categorical_cols] = pd.DataFrame(imputer.fit_transform(df[categorical_cols]), columns=categorical_cols)

#     # One-hot encode the imputed categorical columns
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     encoded = encoder.fit_transform(df[categorical_cols])
#     encoded_cols = encoder.get_feature_names_out(categorical_cols)
#     encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

#     # Print the one-hot encoded data to console for all complaints
#     print("One-Hot Encoded Categorical Data for Complaints:")
#     print(encoded_df.to_string(index=False))
#     print('-' * 80)  # Separator

#     # Concatenate encoded features back to df (drop original categoricals if not needed, but keep for frontend)
#     df = pd.concat([df, encoded_df], axis=1)  # Keep original cols for frontend display

#     # Basic Feature Engineering
#     print("[Feature Engineering] Starting basic feature extraction...")
#     df['word_count'] = df['text'].apply(lambda t: len(t.split()))  # Word count
#     df['char_count'] = df['text'].apply(len)  # Character count
#     df['urgent_keyword_count'] = df['text'].apply(lambda t: sum(1 for kw in URGENCY_MAP if kw in t.lower()))  # Count urgent keywords
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['hour_of_day'] = df['timestamp'].dt.hour  # Hour from timestamp
#     df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
#     print("[Feature Engineering] Added features: word_count, char_count, urgent_keyword_count, hour_of_day, day_of_week")
#     print(df[['user_id', 'word_count', 'char_count', 'urgent_keyword_count', 'hour_of_day', 'day_of_week']].to_string(index=False))
#     print("[Feature Engineering] Completed.")
#     print('-' * 80)

#     # Sorting logic based on the selected metric from the frontend
#     if metric == 'most_supported':
#         ranked_df = df.sort_values(by='support_count', ascending=False)
#     elif metric == 'location':
#         # Since 'city' is still in df, sort by 'city' and 'urgency'
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     else:
#         original_category_name = None
#         for cat_name in CATEGORIES.keys():
#             sanitized_name = cat_name.lower().replace(' ', '_').replace('_and_health_care', '_care')
#             if sanitized_name == metric:
#                 original_category_name = cat_name
#                 break
#         if original_category_name:
#             filtered_df = df[df['final_category'] == original_category_name]
#             ranked_df = filtered_df.sort_values(by='urgency', ascending=False)
#         else:
#             ranked_df = df.sort_values(by='urgency', ascending=False)
            
#     return ranked_df.to_dict(orient='records')


# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from geopy.geocoders import Nominatim
# import psycopg2
# from psycopg2 import sql
# import uvicorn
# import pandas as pd
# from transformers import pipeline
# import numpy as np
# import time
# import json
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# import re  # For text processing
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# from sklearn.cluster import HDBSCAN


# app = FastAPI()



# # --- Model Loading and Initial Setup ---
# model = SentenceTransformer('all-mpnet-base-v2')
# bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")
# geolocator = Nominatim(user_agent="grievance_app")



# # --- Database Configuration ---
# DB_CONFIG = {
#     'dbname': 'grievance_db',
#     'user': 'postgres',
#     'password': 'AshlinJoelSizzin', # Replace with your actual password
#     'host': 'localhost',
#     'port': 5432
# }



# def get_db_conn():
#     return psycopg2.connect(**DB_CONFIG)



# def init_db():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         CREATE EXTENSION IF NOT EXISTS vector;
#         CREATE TABLE IF NOT EXISTS grievances (
#             user_id SERIAL PRIMARY KEY,
#             name TEXT,
#             complaint_text TEXT,
#             timestamp TIMESTAMP,
#             city TEXT,
#             state TEXT,
#             country TEXT,
#             latitude FLOAT,
#             longitude FLOAT,
#             embedding VECTOR(768),
#             keyword_category TEXT,
#             bert_category TEXT,
#             final_category TEXT,
#             urgency DOUBLE PRECISION,
#             safety_risk FLOAT,
#             population_impact FLOAT,
#             community_validation FLOAT,
#             economic_impact FLOAT,
#             environmental_impact FLOAT,
#             overall_severity FLOAT,
#             severity_level TEXT,
#             support_count INT NOT NULL DEFAULT 0
#         );
#     ''')
#     conn.commit()
#     cur.close()
#     conn.close()



# init_db()



# # --- Pydantic Models ---
# class ComplaintIn(BaseModel):
#     name: str
#     complaint_text: str
#     city: str
#     state: str
#     country: str



# # --- Analysis Constants and Classes (omitted for brevity, same as before) ---
# CATEGORIES = {
#     'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
#     'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths', 'flu'],
#     'Environment': ['pollution', 'waste', 'dumping', 'smoke', 'tree', 'deforestation', 'trees', 'contaminate','contamination', 'contaminating','air'],
#     'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
#     'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
#     'Crime': ['crime', 'theft', 'criminal','assault', 'robbery', 'violence', 'violent', 'murder', 'murderer'],
#     'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
#     'Other': []
# }
# URGENCY_MAP = {'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'violence':9, 'violent':9, 'murder':9, 'murderer':9, 'accident': 8,'pollution': 8, 'deforestation':8, 'assault':8, 'trees': 7,'crime': 7,'doctor': 7, 'cyber': 7, 'hacking': 7, 'crime':7, 'theft':7, 'criminal':7, 'robbery':7, 'hospital': 6, 'theft': 6, 'water':6, 'data breach':6, 'traffic': 5, 'pothole': 5, 'waste': 5, 'contamination':5, 'smoke':5, 'malware':5, 'dumping':4,'hack':4, 'hacking':4, 'road': 4, 'phishing':4, 'flu':4, 'area': 3}
# class SeverityCalculator:
#     def __init__(self): self.weights = {'safety_risk': 0.35, 'population_impact': 0.30, 'community_validation': 0.15, 'economic_impact': 0.10, 'environmental_impact': 0.10}
#     def calculate_safety_risk(self, complaint_data, category: str):
#         if category in ['Crime', 'Cyber Crime', 'Public Safety']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         base_score = 0.0
#         safety_critical_terms = { 'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95, 'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8, 'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7 }
#         for term, score in safety_critical_terms.items():
#             if term in description: base_score = max(base_score, score)
#         return min(1.0, base_score)
#     def calculate_population_impact(self, complaint_data, category: str):
#         if category in ['Road Transport', 'Public Safety']: return 0.9
#         direct_affected = complaint_data.get('direct_affected_count', 0)
#         base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
#         return min(1.0, base_score)
#     def calculate_community_validation(self, complaint_data):
#         confirmations = complaint_data.get('neighbor_confirmations', 0)
#         photo_evidence = complaint_data.get('photo_count', 0)
#         return min(1.0, (confirmations / 50) + (photo_evidence / 10))
#     def calculate_economic_impact(self, complaint_data):
#         estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
#         return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0
#     def calculate_environmental_impact(self, complaint_data, category: str):
#         if category in ['Environment', 'Medical and Health Care']: return 0.9
#         description = complaint_data.get('complaint_text', '').lower()
#         environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
#         max_score = 0.0
#         for term, score in environmental_terms.items():
#             if term in description: max_score = max(max_score, score)
#         return max_score
#     def calculate_overall_severity(self, complaint_data, final_category: str):
#         safety_score = self.calculate_safety_risk(complaint_data, final_category)
#         population_score = self.calculate_population_impact(complaint_data, final_category)
#         community_score = self.calculate_community_validation(complaint_data)
#         economic_score = self.calculate_economic_impact(complaint_data)
#         environmental_score = self.calculate_environmental_impact(complaint_data, final_category)
#         overall_severity = (self.weights['safety_risk'] * safety_score + self.weights['population_impact'] * population_score + self.weights['community_validation'] * community_score + self.weights['economic_impact'] * economic_score + self.weights['environmental_impact'] * environmental_score)
#         if overall_severity >= 0.7: severity_level = "CRITICAL"
#         elif overall_severity >= 0.5: severity_level = "HIGH"
#         elif overall_severity >= 0.3: severity_level = "MEDIUM"
#         else: severity_level = "LOW"
#         return {'safety_risk': round(safety_score, 3), 'population_impact': round(population_score, 3), 'community_validation': round(community_score, 3), 'economic_impact': round(economic_score, 3), 'environmental_impact': round(environmental_score, 3), 'overall_severity': round(overall_severity, 3), 'severity_level': severity_level}



# # --- Analysis Functions ---
# severity_calculator = SeverityCalculator()
# def compute_urgency(text: str, keywords: list):
#     text_lower = text.lower()
#     scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
#     return max(scores) if scores else 0
# def enhanced_analysis(complaint_text, extra_data=None):
#     extra_data = extra_data or {}
#     text_lower = complaint_text.lower()
#     keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
#     keyword_category = max(keyword_scores, key=keyword_scores.get)
#     bert_pred = bert_classifier(complaint_text)[0]
#     bert_category = bert_pred['label']
#     bert_score = bert_pred['score'] * 10
#     final_category = bert_category if bert_score > 7 else keyword_category
#     urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
#     complaint_data = {'complaint_text': complaint_text, **extra_data}
#     severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
#     return {'keyword_category': keyword_category, 'bert_category': bert_category, 'final_category': final_category, 'urgency': urgency, **severity_report}



# # --- API Endpoints ---
# @app.post('/submit_complaint')
# def submit_complaint(c: ComplaintIn):
#     location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
#     if not location:
#         raise HTTPException(status_code=400, detail="Invalid location details")
#     embedding = model.encode(c.complaint_text).tolist()
#     timestamp = datetime.utcnow()
#     analysis = enhanced_analysis(c.complaint_text)
    
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('''
#         INSERT INTO grievances (
#             name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
#             keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
#             safety_risk, population_impact, community_validation,
#             economic_impact, environmental_impact, support_count
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
#     ''', (
#         c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
#         location.latitude, location.longitude, embedding,
#         analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
#         analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
#         analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
#         analysis['economic_impact'], analysis['environmental_impact'],
#         0 # Initial support count is 0
#     ))
#     user_id = cur.fetchone()[0]
#     conn.commit()
#     cur.close()
#     conn.close()
#     return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}



# @app.post('/support_complaint/{complaint_id}')
# def support_complaint(complaint_id: int):
#     try:
#         conn = get_db_conn()
#         cur = conn.cursor()
#         cur.execute(
#             sql.SQL("UPDATE grievances SET support_count = support_count + 1 WHERE user_id = %s"),
#             (complaint_id,)
#         )
#         if cur.rowcount == 0:
#             raise HTTPException(status_code=404, detail="Complaint not found")
#         conn.commit()
#         cur.close()
#         conn.close()
#         return {"message": f"Successfully supported complaint {complaint_id}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database error: {e}")


# # Sample labeled test data (expanded to 20 entries for better evaluation)
# sample_test_data = [
#     {'text': 'There was a theft in Mumbai, causing panic in the neighborhood.', 'true_category': 'Crime'},
#     {'text': 'Pollution levels have increased drastically in Delhi, affecting health.', 'true_category': 'Environment'},
#     {'text': 'The hospital delayed emergency treatment, risking lives.', 'true_category': 'Medical and Health Care'},
#     {'text': 'Broken traffic signals caused an accident on the highway.', 'true_category': 'Road Transport'},
#     {'text': 'Fire outbreak in the market caused major damage and safety hazards.', 'true_category': 'Public Safety'},
#     {'text': 'My account was hacked and funds were stolen.', 'true_category': 'Cyber Crime'},
#     {'text': 'Local water supply was contaminated leading to health issues.', 'true_category': 'Environment'},
#     {'text': 'Robbery incident reported near the city park.', 'true_category': 'Crime'},
#     {'text': 'Heavy vehicle traffic causing traffic jams and potholes.', 'true_category': 'Road Transport'},
#     {'text': 'Malware attack affected many local computers.', 'true_category': 'Cyber Crime'},
#     {'text': 'Illegal dumping of waste near river contaminates water.', 'true_category': 'Environment'},
#     {'text': 'Fireworks created safety concerns in residential areas.', 'true_category': 'Public Safety'},
#     {'text': 'Medical supplies are running out in health centers.', 'true_category': 'Medical and Health Care'},
#     {'text': 'Illegal substances found in local school.', 'true_category': 'Crime'},
#     {'text': 'Phishing emails are being sent to employees.', 'true_category': 'Cyber Crime'},
#     {'text': 'Traffic congestion worsens around urban areas.', 'true_category': 'Road Transport'},
#     {'text': 'Community health programs are effective.', 'true_category': 'Medical and Health Care'},
#     {'text': 'Roadworks causing hazards and accidents.', 'true_category': 'Road Transport'},
#     {'text': 'Data breach exposed personal information.', 'true_category': 'Cyber Crime'},
#     {'text': 'Power outage caused disruption in services.', 'true_category': 'Public Safety'}
# ]
# df_test = pd.DataFrame(sample_test_data)

# # Function to evaluate classification
# def evaluate_classification(df_test):
#     # Predict categories using your enhanced_analysis
#     df_test['predicted_category'] = df_test['text'].apply(lambda t: enhanced_analysis(t)['final_category'])
    
#     y_true = df_test['true_category']
#     y_pred = df_test['predicted_category']
    
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
#     cm = confusion_matrix(y_true, y_pred)
    
#     print("[Evaluation] Classification Metrics:")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall (Sensitivity): {recall:.2f}")
#     print(f"F1-Score: {f1:.2f}")
#     print("Confusion Matrix:\n", cm)
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm.tolist()}

# # Function to evaluate clustering (assume embeddings from model)
# def evaluate_clustering(df_test):
#     # Generate embeddings
#     embeddings = np.array([model.encode(t) for t in df_test['text']])
    
#     # Apply HDBSCAN (your clustering algo)
#     clusterer = HDBSCAN(min_cluster_size=2)
#     labels = clusterer.fit_predict(embeddings)
    
#     if len(set(labels)) > 1:  # Need at least 2 clusters for metrics
#         silhouette = silhouette_score(embeddings, labels)
#         db_index = davies_bouldin_score(embeddings, labels)
#         print("[Evaluation] Clustering Metrics:")
#         print(f"Silhouette Score: {silhouette:.2f} (higher is better, range -1 to 1)")
#         print(f"Davies-Bouldin Index: {db_index:.2f} (lower is better)")
#         return {"silhouette": silhouette, "davies_bouldin": db_index}
#     else:
#         print("[Evaluation] Insufficient clusters for metrics.")
#         return {}

# # New endpoint to compute and return metrics
# @app.get('/evaluate_models')
# def evaluate_models():
#     class_metrics = evaluate_classification(df_test)
#     cluster_metrics = evaluate_clustering(df_test)
#     return {"classification": class_metrics, "clustering": cluster_metrics}

# @app.get('/get_ranked_complaints')
# def get_ranked_complaints(metric: str = 'overall'):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude, support_count FROM grievances ORDER BY timestamp DESC;')
#     rows = cur.fetchall()
#     cur.close()
#     conn.close()
#     if not rows:
#         return []


#     df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon', 'support_count'])


#     # Re-run analysis for dynamic ranking
#     for cat, kws in CATEGORIES.items():
#         df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
#     df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
#     df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)
#     bert_predictions = bert_classifier(df['text'].tolist())
#     df['bert_category'] = [pred['label'] for pred in bert_predictions]
#     df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
#     df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
#     df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)


#     # Impute missing values in categorical columns with 'Unknown'
#     categorical_cols = ['city', 'state', 'country', 'final_category']  # Adjust as needed
#     imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
#     df[categorical_cols] = pd.DataFrame(imputer.fit_transform(df[categorical_cols]), columns=categorical_cols)

#     # One-hot encode the imputed categorical columns
#     encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#     encoded = encoder.fit_transform(df[categorical_cols])
#     encoded_cols = encoder.get_feature_names_out(categorical_cols)
#     encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

#     # Print the one-hot encoded data to console for all complaints
#     print("One-Hot Encoded Categorical Data for Complaints:")
#     print(encoded_df.to_string(index=False))
#     print('-' * 80)  # Separator

#     # Concatenate encoded features back to df (drop original categoricals if not needed, but keep for frontend)
#     df = pd.concat([df, encoded_df], axis=1)  # Keep original cols for frontend display

#     # Basic Feature Engineering
#     print("[Feature Engineering] Starting basic feature extraction...")
#     df['word_count'] = df['text'].apply(lambda t: len(t.split()))  # Word count
#     df['char_count'] = df['text'].apply(len)  # Character count
#     df['urgent_keyword_count'] = df['text'].apply(lambda t: sum(1 for kw in URGENCY_MAP if kw in t.lower()))  # Count urgent keywords
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df['hour_of_day'] = df['timestamp'].dt.hour  # Hour from timestamp
#     df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
#     print("[Feature Engineering] Added features: word_count, char_count, urgent_keyword_count, hour_of_day, day_of_week")
#     print(df[['user_id', 'word_count', 'char_count', 'urgent_keyword_count', 'hour_of_day', 'day_of_week']].to_string(index=False))
#     print("[Feature Engineering] Completed.")
#     print('-' * 80)


#     # Sorting logic based on the selected metric from the frontend
#     if metric == 'most_supported':
#         ranked_df = df.sort_values(by='support_count', ascending=False)
#     elif metric == 'location':
#         # Since 'city' is still in df, sort by 'city' and 'urgency'
#         ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
#     elif metric == 'overall':
#         ranked_df = df.sort_values(by='urgency', ascending=False)
#     else:
#         original_category_name = None
#         for cat_name in CATEGORIES.keys():
#             sanitized_name = cat_name.lower().replace(' ', '_').replace('_and_health_care', '_care')
#             if sanitized_name == metric:
#                 original_category_name = cat_name
#                 break
#         if original_category_name:
#             filtered_df = df[df['final_category'] == original_category_name]
#             ranked_df = filtered_df.sort_values(by='urgency', ascending=False)
#         else:
#             ranked_df = df.sort_values(by='urgency', ascending=False)
            
#     return ranked_df.to_dict(orient='records')



# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
import psycopg2
from psycopg2 import sql
import uvicorn
import pandas as pd
from transformers import pipeline
import numpy as np
import time
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import re  # For text processing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import HDBSCAN


app = FastAPI()



# --- Model Loading and Initial Setup ---
model = SentenceTransformer('all-mpnet-base-v2')
bert_classifier = pipeline("text-classification", model="../scripts/finetuned_bert", tokenizer="../scripts/finetuned_bert")
geolocator = Nominatim(user_agent="grievance_app")



# --- Database Configuration ---
DB_CONFIG = {
    'dbname': 'grievance_db',
    'user': 'postgres',
    'password': 'AshlinJoelSizzin', # Replace with your actual password
    'host': 'localhost',
    'port': 5432
}



def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)



def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute('''
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS grievances (
            user_id SERIAL PRIMARY KEY,
            name TEXT,
            complaint_text TEXT,
            timestamp TIMESTAMP,
            city TEXT,
            state TEXT,
            country TEXT,
            latitude FLOAT,
            longitude FLOAT,
            embedding VECTOR(768),
            keyword_category TEXT,
            bert_category TEXT,
            final_category TEXT,
            urgency DOUBLE PRECISION,
            safety_risk FLOAT,
            population_impact FLOAT,
            community_validation FLOAT,
            economic_impact FLOAT,
            environmental_impact FLOAT,
            overall_severity FLOAT,
            severity_level TEXT,
            support_count INT NOT NULL DEFAULT 0
        );
    ''')
    conn.commit()
    cur.close()
    conn.close()



init_db()



# --- Pydantic Models ---
class ComplaintIn(BaseModel):
    name: str
    complaint_text: str
    city: str
    state: str
    country: str



# --- Analysis Constants and Classes (omitted for brevity, same as before) ---
CATEGORIES = {
    'Location': ['area', 'city', 'neighborhood', 'street', 'unclean', 'dirty, sanitation'],
    'Medical and Health Care': ['hospital', 'medicine', 'health', 'doctor', 'emergency', 'death', 'dead', 'deaths', 'flu'],
    'Environment': ['pollution', 'waste', 'dumping', 'smoke', 'tree', 'deforestation', 'trees', 'contaminate','contamination', 'contaminating','air'],
    'Road Transport': ['road', 'traffic', 'pothole', 'accidents', 'transport', 'blocking'],
    'Public Safety': ['safety', 'fire', 'hazard', 'water', 'emergency', 'accident'],
    'Crime': ['crime', 'theft', 'criminal','assault', 'robbery', 'violence', 'violent', 'murder', 'murderer'],
    'Cyber Crime': ['cyber', 'hacked', 'hack', 'hacking', 'phishing', 'malware', 'data breach'],
    'Other': []
}
URGENCY_MAP = {'emergency': 10, 'death': 10, 'deaths':10, 'fire': 9, 'violence':9, 'violent':9, 'murder':9, 'murderer':9, 'accident': 8,'pollution': 8, 'deforestation':8, 'assault':8, 'trees': 7,'crime': 7,'doctor': 7, 'cyber': 7, 'hacking': 7, 'crime':7, 'theft':7, 'criminal':7, 'robbery':7, 'hospital': 6, 'theft': 6, 'water':6, 'data breach':6, 'traffic': 5, 'pothole': 5, 'waste': 5, 'contamination':5, 'smoke':5, 'malware':5, 'dumping':4,'hack':4, 'hacking':4, 'road': 4, 'phishing':4, 'flu':4, 'area': 3}
class SeverityCalculator:
    def __init__(self): self.weights = {'safety_risk': 0.35, 'population_impact': 0.30, 'community_validation': 0.15, 'economic_impact': 0.10, 'environmental_impact': 0.10}
    def calculate_safety_risk(self, complaint_data, category: str):
        if category in ['Crime', 'Cyber Crime', 'Public Safety']: return 0.9
        description = complaint_data.get('complaint_text', '').lower()
        base_score = 0.0
        safety_critical_terms = { 'gas leak': 1.0, 'electrical hazard': 0.9, 'sinkhole': 0.95, 'structural collapse': 1.0, 'traffic accident': 0.8, 'accident': 0.8, 'broken signal': 0.75, 'major pothole': 0.6, 'water pipe burst': 0.7 }
        for term, score in safety_critical_terms.items():
            if term in description: base_score = max(base_score, score)
        return min(1.0, base_score)
    def calculate_population_impact(self, complaint_data, category: str):
        if category in ['Road Transport', 'Public Safety']: return 0.9
        direct_affected = complaint_data.get('direct_affected_count', 0)
        base_score = min(1.0, np.log10(direct_affected + 1) / 4) if direct_affected > 0 else 0.1
        return min(1.0, base_score)
    def calculate_community_validation(self, complaint_data):
        confirmations = complaint_data.get('neighbor_confirmations', 0)
        photo_evidence = complaint_data.get('photo_count', 0)
        return min(1.0, (confirmations / 50) + (photo_evidence / 10))
    def calculate_economic_impact(self, complaint_data):
        estimated_damage_cost = complaint_data.get('estimated_damage_cost', 0)
        return min(1.0, np.log10(estimated_damage_cost + 1) / 10) if estimated_damage_cost > 0 else 0.0
    def calculate_environmental_impact(self, complaint_data, category: str):
        if category in ['Environment', 'Medical and Health Care']: return 0.9
        description = complaint_data.get('complaint_text', '').lower()
        environmental_terms = {'sewage': 0.8, 'contamination': 0.9, 'pollution': 0.6, 'waste': 0.5}
        max_score = 0.0
        for term, score in environmental_terms.items():
            if term in description: max_score = max(max_score, score)
        return max_score
    def calculate_overall_severity(self, complaint_data, final_category: str):
        safety_score = self.calculate_safety_risk(complaint_data, final_category)
        population_score = self.calculate_population_impact(complaint_data, final_category)
        community_score = self.calculate_community_validation(complaint_data)
        economic_score = self.calculate_economic_impact(complaint_data)
        environmental_score = self.calculate_environmental_impact(complaint_data, final_category)
        overall_severity = (self.weights['safety_risk'] * safety_score + self.weights['population_impact'] * population_score + self.weights['community_validation'] * community_score + self.weights['economic_impact'] * economic_score + self.weights['environmental_impact'] * environmental_score)
        if overall_severity >= 0.7: severity_level = "CRITICAL"
        elif overall_severity >= 0.5: severity_level = "HIGH"
        elif overall_severity >= 0.3: severity_level = "MEDIUM"
        else: severity_level = "LOW"
        return {'safety_risk': round(safety_score, 3), 'population_impact': round(population_score, 3), 'community_validation': round(community_score, 3), 'economic_impact': round(economic_score, 3), 'environmental_impact': round(environmental_score, 3), 'overall_severity': round(overall_severity, 3), 'severity_level': severity_level}



# --- Analysis Functions ---
severity_calculator = SeverityCalculator()
def compute_urgency(text: str, keywords: list):
    text_lower = text.lower()
    scores = [URGENCY_MAP.get(kw, 3) for kw in keywords if kw in text_lower]
    return max(scores) if scores else 0
def enhanced_analysis(complaint_text, extra_data=None):
    extra_data = extra_data or {}
    text_lower = complaint_text.lower()
    keyword_scores = {cat: compute_urgency(text_lower, kws) for cat, kws in CATEGORIES.items()}
    keyword_category = max(keyword_scores, key=keyword_scores.get)
    bert_pred = bert_classifier(complaint_text)[0]
    bert_category = bert_pred['label']
    bert_score = bert_pred['score'] * 10
    final_category = bert_category if bert_score > 7 else keyword_category
    urgency = (keyword_scores.get(final_category, 0) + bert_score) / 2
    complaint_data = {'complaint_text': complaint_text, **extra_data}
    severity_report = severity_calculator.calculate_overall_severity(complaint_data, final_category)
    return {'keyword_category': keyword_category, 'bert_category': bert_category, 'final_category': final_category, 'urgency': urgency, **severity_report}



# --- API Endpoints ---
@app.post('/submit_complaint')
def submit_complaint(c: ComplaintIn):
    location = geolocator.geocode(f"{c.city}, {c.state}, {c.country}")
    if not location:
        raise HTTPException(status_code=400, detail="Invalid location details")
    embedding = model.encode(c.complaint_text).tolist()
    timestamp = datetime.utcnow()
    analysis = enhanced_analysis(c.complaint_text)
    
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO grievances (
            name, complaint_text, timestamp, city, state, country, latitude, longitude, embedding,
            keyword_category, bert_category, final_category, urgency, severity_level, overall_severity,
            safety_risk, population_impact, community_validation,
            economic_impact, environmental_impact, support_count
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id;
    ''', (
        c.name, c.complaint_text, timestamp, c.city, c.state, c.country,
        location.latitude, location.longitude, embedding,
        analysis['keyword_category'], analysis['bert_category'], analysis['final_category'],
        analysis['urgency'], analysis['severity_level'], analysis['overall_severity'],
        analysis['safety_risk'], analysis['population_impact'], analysis['community_validation'],
        analysis['economic_impact'], analysis['environmental_impact'],
        0 # Initial support count is 0
    ))
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Complaint submitted", "user_id": user_id, "timestamp": timestamp.isoformat()}



@app.post('/support_complaint/{complaint_id}')
def support_complaint(complaint_id: int):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            sql.SQL("UPDATE grievances SET support_count = support_count + 1 WHERE user_id = %s"),
            (complaint_id,)
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Complaint not found")
        conn.commit()
        cur.close()
        conn.close()
        return {"message": f"Successfully supported complaint {complaint_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# Sample labeled test data (expanded to 20 entries for better evaluation)
sample_test_data = [
    {'text': 'There was a theft in Mumbai, causing panic in the neighborhood.', 'true_category': 'Crime'},
    {'text': 'Pollution levels have increased drastically in Delhi, affecting health.', 'true_category': 'Environment'},
    {'text': 'The hospital delayed emergency treatment, risking lives.', 'true_category': 'Medical and Health Care'},
    {'text': 'Broken traffic signals caused an accident on the highway.', 'true_category': 'Road Transport'},
    {'text': 'Fire outbreak in the market caused major damage and safety hazards.', 'true_category': 'Public Safety'},
    {'text': 'My account was hacked and funds were stolen.', 'true_category': 'Cyber Crime'},
    {'text': 'Local water supply was contaminated leading to health issues.', 'true_category': 'Environment'},
    {'text': 'Robbery incident reported near the city park.', 'true_category': 'Crime'},
    {'text': 'Heavy vehicle traffic causing traffic jams and potholes.', 'true_category': 'Road Transport'},
    {'text': 'Malware attack affected many local computers.', 'true_category': 'Cyber Crime'},
    {'text': 'Illegal dumping of waste near river contaminates water.', 'true_category': 'Environment'},
    {'text': 'Fireworks created safety concerns in residential areas.', 'true_category': 'Public Safety'},
    {'text': 'Medical supplies are running out in health centers.', 'true_category': 'Medical and Health Care'},
    {'text': 'Illegal substances found in local school.', 'true_category': 'Crime'},
    {'text': 'Phishing emails are being sent to employees.', 'true_category': 'Cyber Crime'},
    {'text': 'Traffic congestion worsens around urban areas.', 'true_category': 'Road Transport'},
    {'text': 'Community health programs are effective.', 'true_category': 'Medical and Health Care'},
    {'text': 'Roadworks causing hazards and accidents.', 'true_category': 'Road Transport'},
    {'text': 'Data breach exposed personal information.', 'true_category': 'Cyber Crime'},
    {'text': 'Power outage caused disruption in services.', 'true_category': 'Public Safety'}
]
df_test = pd.DataFrame(sample_test_data)

# Function to evaluate classification
def evaluate_classification(df_test):
    # Predict categories using your enhanced_analysis
    df_test['predicted_category'] = df_test['text'].apply(lambda t: enhanced_analysis(t)['final_category'])
    
    y_true = df_test['true_category']
    y_pred = df_test['predicted_category']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("[Evaluation] Classification Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:\n", cm)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm.tolist()}

# Function to evaluate clustering (assume embeddings from model)
def evaluate_clustering(df_test):
    # Generate embeddings
    embeddings = np.array([model.encode(t) for t in df_test['text']])
    
    # Apply HDBSCAN (your clustering algo)
    clusterer = HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(embeddings)
    
    if len(set(labels)) > 1:  # Need at least 2 clusters for metrics
        silhouette = silhouette_score(embeddings, labels)
        db_index = davies_bouldin_score(embeddings, labels)
        print("[Evaluation] Clustering Metrics:")
        print(f"Silhouette Score: {silhouette:.2f} ")
        print(f"Davies-Bouldin Index: {db_index:.2f}")
        return {"silhouette": silhouette, "davies_bouldin": db_index}
    else:
        print("[Evaluation] Insufficient clusters for metrics.")
        return {}

# New endpoint to compute and return metrics
@app.get('/evaluate_models')
def evaluate_models():
    class_metrics = evaluate_classification(df_test)
    cluster_metrics = evaluate_clustering(df_test)
    return {"classification": class_metrics, "clustering": cluster_metrics}

@app.get('/get_ranked_complaints')
def get_ranked_complaints(metric: str = 'overall'):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute('SELECT user_id, name, complaint_text, timestamp, city, state, country, latitude, longitude, support_count FROM grievances ORDER BY timestamp DESC;')
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return []


    df = pd.DataFrame(rows, columns=['user_id', 'name', 'text', 'timestamp', 'city', 'state', 'country', 'lat', 'lon', 'support_count'])


    # Re-run analysis for dynamic ranking
    for cat, kws in CATEGORIES.items():
        df[cat] = df['text'].apply(lambda t: compute_urgency(t, kws))
    df['Other'] = df.apply(lambda row: 5 if all(row[cat] == 0 for cat in CATEGORIES if cat != 'Other') else 0, axis=1)
    df['keyword_category'] = df.apply(lambda row: max(CATEGORIES, key=lambda cat: row[cat]), axis=1)
    bert_predictions = bert_classifier(df['text'].tolist())
    df['bert_category'] = [pred['label'] for pred in bert_predictions]
    df['bert_score'] = [pred['score'] * 10 for pred in bert_predictions]
    df['final_category'] = df.apply(lambda row: row['bert_category'] if row['bert_score'] > 7 else row['keyword_category'], axis=1)
    df['urgency'] = df.apply(lambda row: (row.get(row['final_category'], 0) + row['bert_score']) / 2, axis=1)


    # Impute missing values in categorical columns with 'Unknown'
    categorical_cols = ['city', 'state', 'country', 'final_category']  # Adjust as needed
    imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    df[categorical_cols] = pd.DataFrame(imputer.fit_transform(df[categorical_cols]), columns=categorical_cols)

    # One-hot encode the imputed categorical columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

    # Print the one-hot encoded data to console for all complaints
    print("One-Hot Encoded Categorical Data for Complaints:")
    print(encoded_df.to_string(index=False))
    print('-' * 80)  # Separator

    # Concatenate encoded features back to df (drop original categoricals if not needed, but keep for frontend)
    df = pd.concat([df, encoded_df], axis=1)  # Keep original cols for frontend display

    # Basic Feature Engineering
    print("[Feature Engineering] Starting basic feature extraction...")
    df['word_count'] = df['text'].apply(lambda t: len(t.split()))  # Word count
    df['char_count'] = df['text'].apply(len)  # Character count
    df['urgent_keyword_count'] = df['text'].apply(lambda t: sum(1 for kw in URGENCY_MAP if kw in t.lower()))  # Count urgent keywords
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour  # Hour from timestamp
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    print("[Feature Engineering] Added features: word_count, char_count, urgent_keyword_count, hour_of_day, day_of_week")
    print(df[['user_id', 'word_count', 'char_count', 'urgent_keyword_count', 'hour_of_day', 'day_of_week']].to_string(index=False))
    print("[Feature Engineering] Completed.")
    print('-' * 80)


    # Call evaluation and print metrics to console
    evaluate_models()

    # Sorting logic based on the selected metric from the frontend
    if metric == 'most_supported':
        ranked_df = df.sort_values(by='support_count', ascending=False)
    elif metric == 'location':
        # Since 'city' is still in df, sort by 'city' and 'urgency'
        ranked_df = df.sort_values(by=['city', 'urgency'], ascending=[True, False])
    elif metric == 'overall':
        ranked_df = df.sort_values(by='urgency', ascending=False)
    else:
        original_category_name = None
        for cat_name in CATEGORIES.keys():
            sanitized_name = cat_name.lower().replace(' ', '_').replace('_and_health_care', '_care')
            if sanitized_name == metric:
                original_category_name = cat_name
                break
        if original_category_name:
            filtered_df = df[df['final_category'] == original_category_name]
            ranked_df = filtered_df.sort_values(by='urgency', ascending=False)
        else:
            ranked_df = df.sort_values(by='urgency', ascending=False)
            
    return ranked_df.to_dict(orient='records')



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

#uvicorn main:app --reload
#streamlit run frontend.py