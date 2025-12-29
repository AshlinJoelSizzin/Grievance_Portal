# AI-Powered Grievance Portal

This repository contains an AI-driven grievance portal that ingests public complaints, classifies them using transformer-based NLP, groups similar issues through hybrid clustering, and ranks them by urgency and community support. The system is designed for transparent, data-centric decision-making in governance and public service environments. [file:1105]

## Tech Stack

- **Frontend:** Streamlit web UI for:
  - Complaint submission
  - Viewing complaint cards and metrics
  - Casting “Lend Support” votes on issues [file:1105]
- **Backend:** FastAPI for:
  - Complaint ingestion (`/submit_complaint`)
  - Loading and ranking complaints (`/load_complaints`)
  - Background analysis jobs (clustering, severity recomputation) [file:1105]
- **NLP & ML:**
  - BERT / Sentence-BERT for embeddings and text classification [file:1105]
  - Keyword-matching heuristic classifier
  - Hybrid fusion layer for final category selection
  - Robust Isolation Forest + Deep Isolation Forest for anomaly handling
  - HDBSCAN for unsupervised clustering and pattern detection [file:1105]
- **Database & Storage:**
  - PostgreSQL with **pgvector** extension for:
    - Storing complaint embeddings
    - Performing cosine-similarity search [file:1105]
  - Structured tables for grievances, severity metrics, support counts, and cluster IDs
- **Geospatial & External Services:**
  - Nominatim Geocoding API for latitude/longitude from city/state/country [file:1105]
  - Plotly maps for location-based visualizations (optional) [file:1105]

---

## System Architecture

The grievance portal follows a three-layer architecture: [file:1105]

- **Frontend (Streamlit):**
  - Submission form for name, complaint text, city, state, country
  - Complaint cards with category, urgency, and support button
  - Metric dropdown to view different ranking modes (e.g., urgency, severity, support) [file:1105]
- **Backend (FastAPI):**
  - Receives submissions as JSON and writes to PostgreSQL
  - Runs NLP, feature engineering, clustering, and severity calculation
  - Exposes APIs for loading ranked complaints and updating support counts [file:1105]
- **Database (PostgreSQL + pgvector):**
  - Stores raw complaint data, geolocation, embeddings, categories, urgency, severity scores, support counts, and cluster IDs [file:1105]

<img width="774" height="1315" alt="image" src="https://github.com/user-attachments/assets/d6d5fc29-c922-44ba-8c7d-5da66b1e888b" />

![System Architecture](Assets and Working)

1. Complaint Submission
 1. A user opens the Streamlit portal and fills out the submission form:
 - Name
 - Complaint text
 - City, State, Country [file:1105]
2. The frontend sends a POST request to /SUBMIT_COMPLAINT on the FastAPI backend with this data. [file:1105]
3. The backend:
 - Stores the complaint with a timestamp
 - Calls Nominatim to fetch latitude and longitude
 - Inserts a new row in the grievances table with text and metadata [file:1105]
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/5220c2e7-8b32-49aa-af2d-31a1d319ab7b" />
