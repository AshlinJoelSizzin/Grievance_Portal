# # frontend.py (Streamlit Dashboard - Run separately with `streamlit run frontend.py`)
# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# user_id = st.text_input("User ID")
# text = st.text_area("Complaint Text")
# timestamp = st.text_input("Timestamp (ISO)")
# lat = st.number_input("Latitude", value=0.0)
# lon = st.number_input("Longitude", value=0.0)

# if st.button("Submit"):
#     response = requests.post("http://localhost:8000/submit", json={
#         "user_id": user_id, "complaint_text": text, "timestamp": timestamp, "latitude": lat, "longitude": lon
#     })
#     st.write(response.json())

# # Run Clustering
# if st.button("Run Clustering"):
#     response = requests.post("http://localhost:8000/run_clustering")
#     st.write(response.json())

# # Issues Page
# st.header("Issues")
# state = st.text_input("Filter by State")
# country = st.text_input("Filter by Country")
# if st.button("Load Issues"):
#     params = {}
#     if state: params['state'] = state
#     if country: params['country'] = country
#     response = requests.get("http://localhost:8000/issues", params=params)
#     data = response.json()
#     df = pd.DataFrame(data)
    
#     # Visualization: Scatter Map
#     if not df.empty:
#         fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", size="final_score", color="final_score",
#                                 hover_name="complaint_text", hover_data=["score_danger", "score_population", "score_socioeconomic"],
#                                 mapbox_style="open-street-map", zoom=3)
#         st.plotly_chart(fig)
#     else:
#         st.write("No issues found")

# frontend.py - Streamlit frontend
# This file handles the user interface: form submission and complaint visualization.
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Complaints
# st.header("View Complaints")
# if st.button("Load Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_complaints")
#         response.raise_for_status()
#         complaints = response.json().get("complaints", [])
#         if complaints:
#             df = pd.DataFrame(complaints)
#             st.dataframe(df)  # Table view
            
#             # Map visualization
#             fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", 
#                                     hover_name="complaint_text", hover_data=["name", "city", "state", "country", "timestamp"],
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading complaints: {e}")

# frontend.py - Streamlit frontend
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Complaints
# st.header("View Complaints")
# if st.button("Load Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_complaints")
#         response.raise_for_status()
#         complaints = response.json().get("complaints", [])
#         if complaints:
#             df = pd.DataFrame(complaints)
#             st.dataframe(df)  # Table view
            
#             # Map visualization
#             fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", 
#                                     hover_name="complaint_text", hover_data=["name", "city", "state", "country", "timestamp"],
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading complaints: {e}")

# frontend.py - Streamlit frontend
# No changes needed here, as the form already matches the inputs

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Complaints
# st.header("View Complaints")
# if st.button("Load Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_complaints")
#         response.raise_for_status()
#         complaints = response.json().get("complaints", [])
#         if complaints:
#             df = pd.DataFrame(complaints)
#             st.dataframe(df)  # Table view
            
#             # Map visualization
#             fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", 
#                                     hover_name="complaint_text", hover_data=["name", "city", "state", "country", "timestamp"],
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading complaints: {e}")

# frontend.py - Streamlit frontend
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Complaints
# st.header("View Complaints")
# if st.button("Load Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_complaints")
#         response.raise_for_status()
#         complaints = response.json().get("complaints", [])
#         if complaints:
#             df = pd.DataFrame(complaints)
#             st.dataframe(df)  # Table view
            
#             # Map visualization
#             fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", 
#                                     hover_name="complaint_text", hover_data=["name", "city", "state", "country", "timestamp"],
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading complaints: {e}")

# frontend.py - Streamlit frontend with metric-based ranking
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Ranked Complaints
# st.header("View Ranked Complaints")
# metric = st.selectbox("Select Metric for Ranking", ["semantic", "location", "keyword", "temporal"])

# if st.button("Load Ranked Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric})
#         response.raise_for_status()
#         ranked = response.json()
#         if ranked:
#             df = pd.DataFrame(ranked)
#             st.dataframe(df[['id', 'text', 'score', 'timestamp', 'lat', 'lon']])  # Ranked table
            
#             # Map visualization (colored by score)
#             fig = px.scatter_mapbox(df, lat="lat", lon="lon", 
#                                     hover_name="text", hover_data=["score", "timestamp"],
#                                     color="score", size="score", color_continuous_scale="reds",
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading ranked complaints: {e}")

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Ranked Complaints
# st.header("View Ranked Complaints")
# metrics = ['Location', 'Medical and Health Care', 'Environment', 'Road Transport', 'Public Safety', 'Crime', 'Cyber Crime', 'Other', 'overall']
# metric = st.selectbox("Select Metric for Ranking", metrics)

# if st.button("Load Ranked Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric})
#         response.raise_for_status()
#         ranked = response.json()
#         if ranked:
#             df = pd.DataFrame(ranked)
#             score_col = metric if metric != 'overall' else 'overall'
#             st.dataframe(df[['name', 'text', score_col, 'city', 'state', 'country', 'timestamp']])  # Ranked table
            
#             # Map visualization (colored by urgency score)
#             fig = px.scatter_mapbox(df, lat="lat", lon="lon", 
#                                     hover_name="text", hover_data=["name", score_col, "timestamp"],
#                                     color=score_col, size=score_col, color_continuous_scale="reds",
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading ranked complaints: {e}")

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Ranked Complaints
# st.header("View Ranked Complaints")
# metrics = ['location', 'medical and health care', 'environment', 'road transport', 'public safety', 'crime', 'cyber crime', 'other', 'overall']
# metric = st.selectbox("Select Metric for Ranking", metrics)

# if st.button("Load Ranked Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric})
#         response.raise_for_status()
#         ranked = response.json()
#         if ranked:
#             df = pd.DataFrame(ranked)
#             st.dataframe(df[['name', 'text', 'urgency', 'category', 'city', 'state', 'country', 'timestamp']])  # Ranked table
            
#             # Map visualization (colored by urgency)
#             fig = px.scatter_mapbox(df, lat="lat", lon="lon", 
#                                     hover_name="text", hover_data=["name", "urgency", "category", "timestamp"],
#                                     color="urgency", size="urgency", color_continuous_scale="reds",
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found for this metric.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading ranked complaints: {e}")

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# Endpoint to get ranked complaints by metric or overall

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# frontend.py - Streamlit frontend with category-based urgency ranking
# Run with: streamlit run frontend.py

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Ranked Complaints
# st.header("View Ranked Complaints")

# # Define display labels and map to backend keys (lowercase with underscores)
# metrics_mapping = {
#     'Location': 'location',
#     'Medical and Health Care': 'medical_and_health_care',
#     'Environment': 'environment',
#     'Road Transport': 'road_transport',
#     'Public Safety': 'public_safety',
#     'Crime': 'crime',
#     'Cyber Crime': 'cyber_crime',
#     'Other': 'other',
#     'Overall': 'overall'
# }
# metrics = list(metrics_mapping.keys())
# selected_metric = st.selectbox("Select Metric for Ranking", metrics)

# # Get the sanitized backend key
# metric_key = metrics_mapping[selected_metric]

# if st.button("Load Ranked Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric_key})
#         response.raise_for_status()
#         ranked = response.json()
#         if ranked:
#             df = pd.DataFrame(ranked)
#             # Display table with classification and urgency details
#             display_cols = ['name', 'text', 'keyword_category', 'bert_category', 'final_category', 'urgency', 'city', 'state', 'country', 'timestamp']
#             st.dataframe(df[display_cols])  # Ranked table
            
#             # Map visualization with hover data
#             fig = px.scatter_mapbox(df, lat="lat", lon="lon", 
#                                     hover_name="text", hover_data=["name", "keyword_category", "bert_category", "final_category", "urgency", "timestamp"],
#                                     color="urgency", size="urgency", color_continuous_scale="reds",
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found for this metric.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading ranked complaints: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submit Complaint Form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             response = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             response.raise_for_status()
#             data = response.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # View Ranked Complaints
# st.header("View Ranked Complaints")

# # Define display labels and map to backend keys (lowercase with underscores)
# metrics_mapping = {
#     'Location': 'location',
#     'Medical and Health Care': 'medical_and_health_care',
#     'Environment': 'environment',
#     'Road Transport': 'road_transport',
#     'Public Safety': 'public_safety',
#     'Crime': 'crime',
#     'Cyber Crime': 'cyber_crime',
#     'Other': 'other',
#     'Overall': 'overall'
# }
# metrics = list(metrics_mapping.keys())
# selected_metric = st.selectbox("Select Metric for Ranking", metrics)

# # Get the sanitized backend key
# metric_key = metrics_mapping[selected_metric]

# if st.button("Load Ranked Complaints"):
#     try:
#         response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric_key})
#         response.raise_for_status()
#         ranked = response.json()
#         if ranked:
#             df = pd.DataFrame(ranked)
#             # Display table with classification and urgency details
#             display_cols = ['name', 'text', 'keyword_category', 'bert_category', 'final_category', 'urgency', 'city', 'state', 'country', 'timestamp']
#             st.dataframe(df[display_cols])  # Ranked table
            
#             # Map visualization with hover data
#             fig = px.scatter_mapbox(df, lat="lat", lon="lon", 
#                                     hover_name="text", hover_data=["name", "keyword_category", "bert_category", "final_category", "urgency", "timestamp"],
#                                     color="urgency", size="urgency", color_continuous_scale="reds",
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found for this metric.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading ranked complaints: {e}")

# frontend.py  ─ run with:  streamlit run frontend.py
# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # ─────────────────────────  Complaint form  ─────────────────────────
# st.header("Submit Complaint")
# name            = st.text_input("Name")
# complaint_text  = st.text_area("Complaint Text")
# city            = st.text_input("City")
# state           = st.text_input("State")
# country         = st.text_input("Country")

# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             r = requests.post(
#                 "http://localhost:8000/submit_complaint",
#                 json={
#                     "name": name,
#                     "complaint_text": complaint_text,
#                     "city": city,
#                     "state": state,
#                     "country": country,
#                 },
#             )
#             r.raise_for_status()
#             data = r.json()
#             st.success(f"Complaint submitted (ID {data['user_id']}) at {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Submission failed: {e}")

# # ────────────────────────  Ranked-complaint viewer  ────────────────────────
# st.header("View Ranked Complaints")

# # Dropdown labels → backend keys
# metric_map = {
#     "Location": "location",
#     "Medical and Health Care": "medical_and_health_care",
#     "Environment": "environment",
#     "Road Transport": "road_transport",
#     "Public Safety": "public_safety",
#     "Crime": "crime",
#     "Cyber Crime": "cyber_crime",
#     "Other": "other",
#     "Overall": "overall",
# }
# chosen_label = st.selectbox("Select Metric for Ranking", list(metric_map.keys()))
# metric      = metric_map[chosen_label]

# if st.button("Load Ranked Complaints"):
#     try:
#         r = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric})
#         r.raise_for_status()
#         records = r.json()
#         if not records:
#             st.info("No complaints found.")
#         else:
#             df = pd.DataFrame(records)
#             # Keep only the final category + urgency
#             df["category"] = df["final_category"]
#             cols = ["name", "text", "category", "urgency", "city", "state", "country", "timestamp"]
#             st.dataframe(df[cols])

#             # Map
#             if {"lat", "lon"}.issubset(df.columns):
#                 fig = px.scatter_mapbox(
#                     df,
#                     lat="lat",
#                     lon="lon",
#                     color="urgency",
#                     size="urgency",
#                     hover_name="text",
#                     hover_data=["category", "urgency", "timestamp"],
#                     zoom=3,
#                     height=500,
#                     mapbox_style="open-street-map",
#                 )
#                 st.plotly_chart(fig)
#     except requests.exceptions.RequestException as e:
#         st.error(f"Loading failed: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard")
# st.title("Grievance Dashboard")

# # Submission form
# st.header("Submit Complaint")
# name = st.text_input("Name")
# complaint_text = st.text_area("Complaint Text")
# city = st.text_input("City")
# state = st.text_input("State")
# country = st.text_input("Country")
# if st.button("Submit"):
#     if not all([name, complaint_text, city, state, country]):
#         st.error("Please fill all fields.")
#     else:
#         try:
#             r = requests.post("http://localhost:8000/submit_complaint", json={
#                 "name": name,
#                 "complaint_text": complaint_text,
#                 "city": city,
#                 "state": state,
#                 "country": country
#             })
#             r.raise_for_status()
#             data = r.json()
#             st.success(f"Complaint submitted! User ID: {data['user_id']}, Timestamp: {data['timestamp']}")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Error submitting complaint: {e}")

# # Ranked Complaints with Severity
# st.header("Ranked Complaints with Severity Analysis")
# metrics_mapping = {
#     'Location': 'location',
#     'Medical and Health Care': 'medical_and_health_care',
#     'Environment': 'environment',
#     'Road Transport': 'road_transport',
#     'Public Safety': 'public_safety',
#     'Crime': 'crime',
#     'Cyber Crime': 'cyber_crime',
#     'Other': 'other',
#     'Overall': 'overall'
# }
# metrics = list(metrics_mapping.keys())
# selected_metric = st.selectbox("Select Metric for Ranking", metrics)
# metric_key = metrics_mapping[selected_metric]

# if st.button("Load Ranked Complaints"):
#     try:
#         r = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": metric_key})
#         r.raise_for_status()
#         ranked = r.json()
#         if ranked:
#             df = pd.DataFrame(ranked)
#             # Display table with severity details
#             display_cols = ['name', 'text', 'final_category', 'urgency', 'severity_level', 'overall_severity', 'safety_risk', 'population_impact', 'community_validation', 'economic_impact', 'environmental_impact', 'accessibility_impact', 'city', 'state', 'country', 'timestamp']
#             st.dataframe(df[display_cols])  # Ranked table with severity
            
#             # Severity summary
#             st.subheader("Severity Level Summary")
#             severity_counts = df['severity_level'].value_counts().to_dict()
#             st.write(severity_counts)
            
#             # Map visualization colored by severity level
#             fig = px.scatter_mapbox(df, lat="lat", lon="lon", 
#                                     hover_name="text", hover_data=["final_category", "severity_level", "overall_severity", "urgency", "timestamp"],
#                                     color="severity_level", size="overall_severity", color_continuous_scale="reds",
#                                     zoom=3, height=500, mapbox_style="open-street-map")
#             st.plotly_chart(fig)
#         else:
#             st.info("No complaints found for this metric.")
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error loading ranked complaints: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard", layout="wide")
# st.title("Grievance Dashboard 🗺️")

# # Submit Complaint Form
# with st.expander("Submit a New Complaint", expanded=True):
#     with st.form("complaint_form"):
#         name = st.text_input("Name")
#         complaint_text = st.text_area("Complaint Description")
#         city = st.text_input("City")
#         state = st.text_input("State")
#         country = st.text_input("Country", "India")
#         submitted = st.form_submit_button("Submit Complaint")

#     if submitted:
#         if not all([name, complaint_text, city, state, country]):
#             st.error("Please fill all fields.")
#         else:
#             with st.spinner("Submitting and analyzing complaint..."):
#                 try:
#                     r = requests.post(
#                         "http://localhost:8000/submit_complaint",
#                         json={"name": name, "complaint_text": complaint_text, "city": city, "state": state, "country": country}
#                     )
#                     r.raise_for_status()
#                     data = r.json()
#                     st.success(f"Complaint submitted successfully! User ID: {data['user_id']}")
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Error connecting to backend: {e}")

# # View Ranked Complaints
# st.header("Public Grievance Dashboard")

# # --- MODIFIED: Reverted to original metrics and removed Environment/Accessibility ---
# metrics = ['Overall', 'Location', 'Medical and Health Care', 'Road Transport', 'Public Safety', 'Crime', 'Cyber Crime', 'Other']
# metric_map = {
#     'Overall': 'overall',
#     'Location': 'location',
#     'Medical and Health Care': 'medical_care',
#     'Road Transport': 'road_transport',
#     'Public Safety': 'public_safety',
#     'Crime': 'crime',
#     'Cyber Crime': 'cyber_crime',
#     'Other': 'other'
# }
# selected_metric_display = st.selectbox("Rank Complaints By Category", metrics)
# selected_metric_api = metric_map[selected_metric_display]

# if st.button("Load and Rank Complaints"):
#     with st.spinner("Loading and ranking complaints..."):
#         try:
#             response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": selected_metric_api})
#             response.raise_for_status()
#             complaints = response.json()
            
#             if complaints:
#                 df = pd.DataFrame(complaints)
                
#                 st.subheader(f"Complaints Ranked by: {selected_metric_display}")
#                 # --- MODIFIED: Removed impact columns from the display table ---
#                 display_cols = ["name", "text", "final_category", "urgency", "timestamp", "city", "state"]
#                 st.dataframe(df[display_cols])

#                 st.subheader("Geographical Hotspots by Urgency")
#                 # --- MODIFIED: Map now uses 'urgency' for color and size ---
#                 fig = px.scatter_mapbox(
#                     df, lat="lat", lon="lon",
#                     color="urgency",  # Changed from severity_level
#                     size="urgency",   # Changed from overall_severity
#                     hover_name="text",
#                     hover_data=["final_category", "urgency", "timestamp"],
#                     color_continuous_scale=px.colors.sequential.OrRd,
#                     zoom=4,
#                     height=600,
#                     mapbox_style="open-street-map"
#                 )
#                 fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No complaints found in the database.")
#         except Exception as e:
#             st.error(f"Error loading complaints: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# st.set_page_config(page_title="Grievance Dashboard", layout="wide")
# st.title("Grievance Dashboard 🗺️")

# # Submit Complaint Form
# with st.expander("Submit a New Complaint", expanded=True):
#     with st.form("complaint_form"):
#         name = st.text_input("Name")
#         complaint_text = st.text_area("Complaint Description")
#         city = st.text_input("City")
#         state = st.text_input("State")
#         country = st.text_input("Country", "India")
#         submitted = st.form_submit_button("Submit Complaint")

#     if submitted:
#         if not all([name, complaint_text, city, state, country]):
#             st.error("Please fill all fields.")
#         else:
#             with st.spinner("Submitting and analyzing complaint..."):
#                 try:
#                     r = requests.post(
#                         "http://localhost:8000/submit_complaint",
#                         json={"name": name, "complaint_text": complaint_text, "city": city, "state": state, "country": country}
#                     )
#                     r.raise_for_status()
#                     data = r.json()
#                     st.success(f"Complaint submitted successfully! User ID: {data['user_id']}")
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Error connecting to backend: {e}")

# # View Ranked Complaints
# st.header("Public Grievance Dashboard")

# # --- MODIFIED: Added 'Environment' back to the metrics list ---
# metrics = ['Overall', 'Location', 'Medical and Health Care', 'Environment', 'Road Transport', 'Public Safety', 'Crime', 'Cyber Crime', 'Other']
# metric_map = {
#     'Overall': 'overall',
#     'Location': 'location',
#     'Medical and Health Care': 'medical_care',
#     'Environment': 'environment', # <-- Added this line
#     'Road Transport': 'road_transport',
#     'Public Safety': 'public_safety',
#     'Crime': 'crime',
#     'Cyber Crime': 'cyber_crime',
#     'Other': 'other'
# }
# selected_metric_display = st.selectbox("Rank Complaints By Category", metrics)
# selected_metric_api = metric_map[selected_metric_display]

# if st.button("Load and Rank Complaints"):
#     with st.spinner("Loading and ranking complaints..."):
#         try:
#             response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": selected_metric_api})
#             response.raise_for_status()
#             complaints = response.json()
            
#             if complaints:
#                 df = pd.DataFrame(complaints)
                
#                 st.subheader(f"Complaints Ranked by: {selected_metric_display}")
#                 display_cols = ["name", "text", "final_category", "urgency", "timestamp", "city", "state"]
#                 st.dataframe(df[display_cols])

#                 st.subheader("Geographical Hotspots by Urgency")
#                 fig = px.scatter_mapbox(
#                     df, lat="lat", lon="lon",
#                     color="urgency",
#                     size="urgency",
#                     hover_name="text",
#                     hover_data=["final_category", "urgency", "timestamp"],
#                     color_continuous_scale=px.colors.sequential.Viridis,
#                     # color_continuous_scale=px.colors.sequential.Plasma,
#                     zoom=4,
#                     height=600,
#                     mapbox_style="open-street-map"
#                 )
#                 fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No complaints found in the database.")
#         except Exception as e:
#             st.error(f"Error loading complaints: {e}")

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# # Set page configuration
# st.set_page_config(page_title="Grievance Dashboard", layout="wide")
# st.title("Grievance Dashboard 🗺️")

# ## --- KEY CHANGE 1: Initialize Session State ---
# # This ensures our variable exists and persists across reruns.
# if 'complaints_visible' not in st.session_state:
#     st.session_state.complaints_visible = False

# # Expander for the complaint submission form
# with st.expander("Submit a New Complaint", expanded=True):
#     with st.form("complaint_form"):
#         name = st.text_input("Name")
#         complaint_text = st.text_area("Complaint Description")
#         city = st.text_input("City")
#         state = st.text_input("State")
#         country = st.text_input("Country", "India")
#         submitted = st.form_submit_button("Submit Complaint")

#     if submitted:
#         if not all([name, complaint_text, city, state, country]):
#             st.error("Please fill all fields.")
#         else:
#             with st.spinner("Submitting and analyzing complaint..."):
#                 try:
#                     r = requests.post(
#                         "http://localhost:8000/submit_complaint",
#                         json={"name": name, "complaint_text": complaint_text, "city": city, "state": state, "country": country}
#                     )
#                     r.raise_for_status()
#                     data = r.json()
#                     st.success(f"Complaint submitted successfully! User ID: {data['user_id']}")
#                     st.session_state.complaints_visible = True # Automatically show complaints after submitting a new one
#                 except requests.exceptions.RequestException as e:
#                     st.error(f"Error connecting to backend: {e}")

# # Main dashboard section for viewing complaints
# st.header("Public Grievance Dashboard")

# # Define metrics for the dropdown
# metrics = ['Overall', 'Most Supported', 'Location', 'Medical and Health Care', 'Environment', 'Road Transport', 'Public Safety', 'Crime', 'Cyber Crime', 'Other']
# metric_map = {
#     'Overall': 'overall',
#     'Most Supported': 'most_supported',
#     'Location': 'location',
#     'Medical and Health Care': 'medical_care',
#     'Environment': 'environment',
#     'Road Transport': 'road_transport',
#     'Public Safety': 'public_safety',
#     'Crime': 'crime',
#     'Cyber Crime': 'cyber_crime',
#     'Other': 'other'
# }
# # Use a session state key for the selectbox to remember the choice across reruns
# selected_metric_display = st.selectbox(
#     "Rank Complaints By Category",
#     metrics,
#     key='metric_select'
# )
# selected_metric_api = metric_map[selected_metric_display]

# ## --- KEY CHANGE 2: Button now only sets the session state ---
# # This button's only job is to turn on the visibility of the complaints section.
# if st.button("Load and Rank Complaints"):
#     st.session_state.complaints_visible = True

# ## --- KEY CHANGE 3: Display logic depends on session state, not the button ---
# # This block will now run every time the script reruns, as long as complaints_visible is True.
# if st.session_state.complaints_visible:
#     with st.spinner("Loading and ranking complaints..."):
#         try:
#             # Data is fetched FRESH on every rerun, ensuring we always have the latest support counts.
#             response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": selected_metric_api})
#             response.raise_for_status()
#             complaints = response.json()
            
#             if complaints:
#                 df = pd.DataFrame(complaints)
                
#                 st.subheader(f"Complaints Ranked by: {selected_metric_display}")

#                 # The display loop for each complaint remains the same
#                 for index, row in df.iterrows():
#                     with st.container(border=True):
#                         col1, col2 = st.columns([1, 4])
#                         with col1:
#                             st.metric("Supports", row.get('support_count', 0))
#                             if st.button("Lend Support 👍", key=f"support_{row['user_id']}"):
#                                 try:
#                                     support_res = requests.post(f"http://localhost:8000/support_complaint/{row['user_id']}")
#                                     support_res.raise_for_status()
#                                     st.success(f"Thank you for supporting complaint #{row['user_id']}!")
#                                     st.rerun() # This will now work correctly
#                                 except Exception as e:
#                                     st.error(f"Failed to lend support: {e}")
#                         with col2:
#                             # st.write(row['text'])
#                             # st.write(f"**{row['text']}**")
#                             st.markdown(f'<p style="font-size:18px; font-weight:bold;">{row["text"]}</p>', unsafe_allow_html=True)
#                             st.markdown(f"**Category:** `{row['final_category']}` | **Urgency:** `{row['urgency']:.2f}`")
#                             st.markdown(f"**Location:** `{row['city']}, {row['state']}`")
#                             st.caption(f"Complaint by {row['name']} on {pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                            
                
#                 # Geographical map visualization
#                 st.subheader("Geographical Hotspots by Urgency")
#                 fig = px.scatter_mapbox(
#                     df, lat="lat", lon="lon", color="urgency", size="urgency",
#                     hover_name="text", hover_data=["final_category", "urgency", "timestamp", "support_count"],
#                     color_continuous_scale=px.colors.sequential.Viridis, zoom=4, height=600,
#                     mapbox_style="open-street-map"
#                 )
#                 fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("No complaints found in the database.")
#         except Exception as e:
#             st.error(f"Error loading complaints: {e}")

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime


# Set page configuration
st.set_page_config(page_title="Grievance Dashboard", layout="wide")
st.title("Grievance Dashboard 🗺️")


## --- KEY CHANGE 1: Initialize Session State ---
# This ensures our variable exists and persists across reruns.
if 'complaints_visible' not in st.session_state:
    st.session_state.complaints_visible = False


# Expander for the complaint submission form
with st.expander("Submit a New Complaint", expanded=True):
    with st.form("complaint_form"):
        name = st.text_input("Name")
        complaint_text = st.text_area("Complaint Description")
        city = st.text_input("City")
        state = st.text_input("State")
        country = st.text_input("Country", "India")
        submitted = st.form_submit_button("Submit Complaint")


    if submitted:
        if not all([name, complaint_text, city, state, country]):
            st.error("Please fill all fields.")
        else:
            with st.spinner("Submitting and analyzing complaint..."):
                try:
                    r = requests.post(
                        "http://localhost:8000/submit_complaint",
                        json={"name": name, "complaint_text": complaint_text, "city": city, "state": state, "country": country}
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.success(f"Complaint submitted successfully! User ID: {data['user_id']}")
                    st.session_state.complaints_visible = True # Automatically show complaints after submitting a new one
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")


# Main dashboard section for viewing complaints
st.header("Public Grievance Dashboard")


# Define metrics for the dropdown
metrics = ['Overall', 'Most Supported', 'Location', 'Medical and Health Care', 'Environment', 'Road Transport', 'Public Safety', 'Crime', 'Cyber Crime', 'Other']
metric_map = {
    'Overall': 'overall',
    'Most Supported': 'most_supported',
    'Location': 'location',
    'Medical and Health Care': 'medical_care',
    'Environment': 'environment',
    'Road Transport': 'road_transport',
    'Public Safety': 'public_safety',
    'Crime': 'crime',
    'Cyber Crime': 'cyber_crime',
    'Other': 'other'
}
# Use a session state key for the selectbox to remember the choice across reruns
selected_metric_display = st.selectbox(
    "Rank Complaints By Category",
    metrics,
    key='metric_select'
)
selected_metric_api = metric_map[selected_metric_display]


## --- KEY CHANGE 2: Button now only sets the session state ---
# This button's only job is to turn on the visibility of the complaints section.
if st.button("Load and Rank Complaints"):
    st.session_state.complaints_visible = True


## --- KEY CHANGE 3: Display logic depends on session state, not the button ---
# This block will now run every time the script reruns, as long as complaints_visible is True.
if st.session_state.complaints_visible:
    with st.spinner("Loading and ranking complaints..."):
        try:
            # Data is fetched FRESH on every rerun, ensuring we always have the latest support counts.
            response = requests.get("http://localhost:8000/get_ranked_complaints", params={"metric": selected_metric_api})
            response.raise_for_status()
            complaints = response.json()
            
            if complaints:
                df = pd.DataFrame(complaints)
                
                # --- Add lapsed time calculation ---
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)  # Make timezone naive for subtraction
                current_time = datetime.utcnow()
                df['lapsed_time'] = df['timestamp'].apply(lambda ts: current_time - ts)
                
                # Format as human-readable string (e.g., '2d 3h 15m')
                def fmt_delta(td):
                    total_seconds = int(td.total_seconds())
                    days, rem = divmod(total_seconds, 86400)
                    hours, rem = divmod(rem, 3600)
                    minutes, seconds = divmod(rem, 60)
                    parts = []
                    if days > 0:
                        parts.append(f"{days}d")
                    if hours > 0:
                        parts.append(f"{hours}h")
                    if minutes > 0:
                        parts.append(f"{minutes}m")
                    if not parts:
                        parts.append(f"{seconds}s")
                    return " ".join(parts)
                
                df['lapsed_time'] = df['lapsed_time'].apply(fmt_delta)
                
                st.subheader(f"Complaints Ranked by: {selected_metric_display}")


                # The display loop for each complaint remains the same
                for index, row in df.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.metric("Supports", row.get('support_count', 0))
                            if st.button("Lend Support 👍", key=f"support_{row['user_id']}"):
                                try:
                                    support_res = requests.post(f"http://localhost:8000/support_complaint/{row['user_id']}")
                                    support_res.raise_for_status()
                                    st.success(f"Thank you for supporting complaint #{row['user_id']}!")
                                    st.rerun() # This will now work correctly
                                except Exception as e:
                                    st.error(f"Failed to lend support: {e}")
                        with col2:
                            # st.write(row['text'])
                            # st.write(f"**{row['text']}**")
                            st.markdown(f'<p style="font-size:18px; font-weight:bold;">{row["text"]}</p>', unsafe_allow_html=True)
                            st.markdown(f"**Category:** `{row['final_category']}` | **Urgency:** `{row['urgency']:.2f}`")
                            st.markdown(f"**Location:** `{row['city']}, {row['state']}`")
                            st.caption(f"Complaint by {row['name']} on {pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')} | Lapsed Time: {row['lapsed_time']}")
                            
                
                # Geographical map visualization
                st.subheader("Geographical Hotspots by Urgency")
                fig = px.scatter_mapbox(
                    df, lat="lat", lon="lon", color="urgency", size="urgency",
                    hover_name="text", hover_data=["final_category", "urgency", "timestamp", "support_count"],
                    color_continuous_scale=px.colors.sequential.Viridis, zoom=4, height=600,
                    mapbox_style="open-street-map"
                )
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No complaints found in the database.")
        except Exception as e:
            st.error(f"Error loading complaints: {e}")
