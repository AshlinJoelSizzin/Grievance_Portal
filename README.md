# FaceGuard: AI-Powered Grievance Intelligence Platform

FaceGuard is an AI-driven grievance analysis and ranking system that ingests public complaints, classifies them using transformer-based NLP, groups similar issues via hybrid clustering, and ranks them by urgency and community support. It is designed for transparent, data-centric decision-making in governance and public service contexts.

## Tech Stack

- **Frontend:** Streamlit (Python-based web UI for complaint submission, browsing, and support voting)  
- **Backend:** FastAPI (REST APIs for ingestion, analysis, and retrieval)  
- **NLP & ML:**
  - Sentence-BERT / BERT for text embeddings and classification  
  - Keyword-based heuristic classifier + hybrid fusion  
  - Robust Isolation Forest and Deep Isolation Forest for anomaly handling  
  - HDBSCAN for unsupervised clustering of complaints  
- **Database & Storage:**
  - PostgreSQL with **pgvector** extension for embedding storage and similarity search  
  - Tables for grievances, severity metrics, cluster IDs, and support counts  
- **Geospatial & External Services:**
  - Nominatim Geocoding API for latitude/longitude and address enrichment  
  - Plotly for maps/heatmaps in the UI

---

## System Architecture

The platform follows a three-tier architecture:

- **Frontend (Streamlit)** – submission form, complaint cards, support/“lend support” button, and metric dropdown to explore different ranking views.  
- **Backend (FastAPI)** – processes incoming complaints, runs NLP and ML pipelines, updates PostgreSQL/pgvector, and exposes JSON APIs (`/submit_complaint`, `/load_complaints`, etc.).  
- **Database Layer (PostgreSQL + pgvector)** – stores raw complaint text, metadata (city, state, country, timestamp), BERT embeddings, categories, severity and urgency scores, community support counts, and cluster IDs.

The architecture diagram in this repository illustrates:

- Data flow from **Streamlit** → **FastAPI** → **PostgreSQL + pgvector**  
- Background jobs for:
  - Loading models on startup (BERT classifier, embedding model)  
  - Nightly / periodic HDBSCAN clustering  
  - Recomputing urgency and severity when new data arrives  
- Integration with Nominatim (HTTP GET) for geocoding addresses.
<img width="774" height="1315" alt="image" src="https://github.com/user-attachments/assets/5ec2bcf4-99d3-4a1e-8128-c35a9d3ea189" />
