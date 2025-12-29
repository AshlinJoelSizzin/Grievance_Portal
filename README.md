# AI-Powered Grievance Portal

This repository contains an AI-driven grievance portal that ingests public complaints, classifies them using transformer-based NLP, groups similar issues through hybrid clustering, and ranks them by urgency and community support. The system is designed for transparent, data-centric decision-making in governance and public service environments. 

## Tech Stack

- **Frontend:** Streamlit web UI for:
  - Complaint submission
  - Viewing complaint cards and metrics
  - Casting “Lend Support” votes on issues 
- **Backend:** FastAPI for:
  - Complaint ingestion (`/submit_complaint`)
  - Loading and ranking complaints (`/load_complaints`)
  - Background analysis jobs (clustering, severity recomputation) 
- **NLP & ML:**
  - BERT / Sentence-BERT for embeddings and text classification 
  - Keyword-matching heuristic classifier
  - Hybrid fusion layer for final category selection
  - Robust Isolation Forest + Deep Isolation Forest for anomaly handling
  - HDBSCAN for unsupervised clustering and pattern detection 
- **Database & Storage:**
  - PostgreSQL with **pgvector** extension for:
    - Storing complaint embeddings
    - Performing cosine-similarity search 
  - Structured tables for grievances, severity metrics, support counts, and cluster IDs
- **Geospatial & External Services:**
  - Nominatim Geocoding API for latitude/longitude from city/state/country
  - Plotly maps for location-based visualizations (optional)

---

## System Architecture

The grievance portal follows a three-layer architecture: 

- **Frontend (Streamlit):**
  - Submission form for name, complaint text, city, state, country
  - Complaint cards with category, urgency, and support button
  - Metric dropdown to view different ranking modes (e.g., urgency, severity, support) 
- **Backend (FastAPI):**
  - Receives submissions as JSON and writes to PostgreSQL
  - Runs NLP, feature engineering, clustering, and severity calculation
  - Exposes APIs for loading ranked complaints and updating support counts 
- **Database (PostgreSQL + pgvector):**
  - Stores raw complaint data, geolocation, embeddings, categories, urgency, severity scores, support counts, and cluster IDs 

<img width="774" height="1315" alt="image" src="https://github.com/user-attachments/assets/d6d5fc29-c922-44ba-8c7d-5da66b1e888b" />

- **Assets and Working:**
1. Complaint Submission
 1. A user opens the Streamlit portal and fills out the submission form:
    - Name
    - Complaint text
    - City, State, Country 
3. The frontend sends a POST request to /SUBMIT_COMPLAINT on the FastAPI backend with this data. 
4. The backend:
   - Stores the complaint with a timestamp
   - Calls Nominatim to fetch latitude and longitude
   - Inserts a new row in the grievances table with text and metadata 
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/5220c2e7-8b32-49aa-af2d-31a1d319ab7b" />

## Complaint Submission UI
After storage, the backend triggers the analysis pipeline: 

- **Embeddings:** A Sentence-BERT model converts complaint text into a 768‑dimensional embedding stored in the `embedding` (vector) column. 
- **Hybrid Classification:**
  - A keyword-matching engine suggests a category (e.g., Medical & Health Care, Environment, Public Safety, Crime, Cyber Crime, Road Transport).
  - A fine-tuned BERT classifier infers another category with probabilities.
  - A fusion logic resolves the final category and writes it to `final_category`.
- **Feature Engineering:**
  - Word count, character count, urgent keyword counts, time-based features
  - One-hot encoding of city, state, country, and final category to form a sparse, model-ready feature matrix 

<img width="978" height="651" alt="image" src="https://github.com/user-attachments/assets/6e5bf586-b648-45f9-86fc-3b733d3efc27" />
<img width="975" height="637" alt="image" src="https://github.com/user-attachments/assets/7573f770-bb39-4e94-a0da-47896dc73eeb" />
<img width="975" height="341" alt="image" src="https://github.com/user-attachments/assets/96e1d327-b080-4f0e-ac09-0ec95de44f47" />

## Feature Enigneering Output
To uncover patterns across heterogeneous complaints: 
1. Robust Isolation Forest: Filters outliers and noisy complaints in embedding space before clustering.
2. HDBSCAN Clustering: Operates on the filtered embeddings to discover variable-density clusters (e.g., regional traffic issues, recurring medical problems).
3. Deep Isolation Forest: Refines clusters by identifying subtle anomalies within each cluster, improving purity and robustness without discarding rare but critical complaints. 
Each complaint gets a cluster_id written back to PostgreSQL, and clustering jobs can be scheduled periodically (e.g., nightly). 
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/dd82064a-c122-48df-8a5f-3ad3777942c1" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/0e4d2ae2-8ee6-4241-9000-529accbe4fec" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/cfb759d9-2c1b-4339-891d-1b35953405c3" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/fdcfbbab-19f0-44bc-b09d-e481192c8dad" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/1687cf26-75e9-447a-9d50-a634983a1de5" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/f4d95124-e816-4e02-a0cb-8f09977ac8fe" />

## Embedding Visualization and Urgency Computation
For each complaint, the portal computes a composite **urgency/severity score** based on: 

- Classification confidence from the hybrid NLP model  
- Domain-specific severity indicators (e.g., “ambulance”, “flu”, “fire”, “theft”, “riot”)  
- Population and safety risk, economic, environmental, and accessibility impacts  
- Temporal dynamics (recency and resolution status)  
- Community validation via support count and support velocity (rate of new supports) 

These metrics are stored in columns such as `safety_risk`, `population_impact`, `economic_impact`, `environmental_impact`, `accessibility_impact`, `severity_level`, `overall_severity`, and `urgency`. 

<img width="931" height="275" alt="image" src="https://github.com/user-attachments/assets/64d76bfe-f171-4cae-8287-571430167db5" />

## Severity and Mertic Controls and Ranking
- Each complaint card in the Streamlit UI includes:
  - Complaint text and category
  - City/state/country
  - Urgency or severity score
  - A “Lend Support” button that sends a POST to update support_count. 
- Complaints are loaded via a GET endpoint that supports sorting by:
  - Overall urgency
  - Severity level
  - Community support
- The combination of urgency model + support count ensures:
  - Critical but low-visibility complaints can still rank high.
  - Highly supported, less severe complaints do not dominate unfairly. 

<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/262b9014-389a-4023-89c9-8cb9f228c04e" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/cff356a2-c12f-407c-92a3-e50354b434bc" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/a0b21c95-cea8-4e12-b3eb-7a374b664066" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/51a3a9e6-a3ba-44d3-8fd3-502c54560a38" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/aa6c736b-2c9e-40bc-a1ee-b71bfe2e9f53" />

## Ranked Complaints and Support
The `grievances` table in PostgreSQL stores: 

- `user_id` (SERIAL PK), `name`, `complaint_text`, `timestamp`  
- `city`, `state`, `country`, `latitude`, `longitude`  
- `embedding` (vector with pgvector), `bert_category`, `final_category`  
- `safety_risk`, `keyword_category`, `population_impact`, `economic_impact`, `environmental_impact`, `accessibility_impact`  
- `severity_level`, `overall_severity`, `support_count`, `cluster_id` 

Developers can inspect rows using pgAdmin (as in the execution screenshots) to verify data, embeddings, and geolocation values. 
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/10fbbc78-4eef-4ec3-987c-02a3cb1d7096" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/825412fb-77fd-4be8-80f5-c163eba3116a" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/af1efa35-973c-4ddf-b2e5-b9e2af9205db" />

## Embeddings and Geo Columns
- Real complaints from multiple countries (India, USA, China, Australia, Japan, South Africa, Sri Lanka) and categories (Medical & Health Care, Environment, Road Transport, Public Safety, Crime, Cyber Crime). 
- **Environment:** 4‑core server, PostgreSQL + pgvector, FastAPI backend, Streamlit frontend, Nominatim for geocoding. 
- **Validation:** Classification metrics (accuracy, precision, recall, F1), clustering metrics (silhouette score, Davies–Bouldin index), and empirical checks on ranking quality (e.g., critical medical complaints surfacing high even with low support).

## Getting Started

- git clone https://github.com/AshlinJoelSizzin/Grievance_Portal.git
- cd grievance-portal
- pip install -r requirements.txt
- Setup PostgreSQL with `pgvector` and create the `grievances` table (SQL schema provided in this repo).  
- Start the backend:
  - uvicorn main:app --host 0.0.0.0 --port 8000 --reload
- Start the Streamlit frontend:
  - streamlit run app.py
- Open the displayed URL (e.g., `http://localhost:8501`) to use the grievance portal.

## Repository Structure
├─ app.py # Streamlit frontend <br>
├─ main.py # FastAPI backend <br>
├─ clustering_system.py # Embedding, clustering, severity logic <br>
├─ data/ # Sample complaint data <br>
├─ assets/ # Architecture and execution screenshots <br>
│ ├─ architecture.png <br>
│ ├─ complaint-submission.png <br>
│ ├─ feature-engineering.png <br>
│ ├─ embeddings.png <br>
│ ├─ ranked-complaints.png <br>
│ ├─ severity-ui.png <br>
│ ├─ grievances-table.png <br>
│ ├─ embeddings-table.png <br>
│ └─ metrics-logs.png <br>
├─ requirements.txt <br>
└─ README.md
