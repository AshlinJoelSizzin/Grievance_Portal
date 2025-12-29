# pylint: disable-all
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class ClusteringEmbeddings:
    def __init__(self, embedding_model='all-mpnet-base-v2', min_cluster_size=2, min_samples=1, alpha=0.5):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dangercluster_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, alpha=alpha, cluster_selection_method='leaf')
        self.populatecluster_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, alpha=alpha)
        
        self.embeddings = []
        self.cluster_labels = []
        
        self.final_scores = []
        
        self.db_login = {
            # All database login details here
        }


    def preprocess_texts(self, sentences):
        
        self.embeddings = self.embedding_model.encode(sentences)

        if self.embeddings.shape[-1] == 768:
            return self.embeddings


    def save_to_postgres(self):
        # TODO: Define method here to save self.embeddings to postgres if each is 768 numbers long
        
        return


    def load_from_postgres(self):
        # TODO: Define method here to load all embeddings from postgres into self.embeddings
        
        return


    def severity_clustering(self):
        try:
            if self.embeddings.shape[-1] == 768:
                self.dangercluster_model.fit(self.embeddings)
                
                self.cluster_labels = self.dangercluster_model.labels_

        except:
            raise TypeError(f"self.embeddings must be data type 'numpy.ndarray' but is {type(self.embeddings)}")
        
        counts = {}
        
        for label in set(self.cluster_labels):
            
            counts[int(label)] = list(self.cluster_labels).count(label)
        
        # Ordering clusters in descending order of density
        sorted_clusters = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        
        # TODO: See if dynamic threshold needs to be changed
        threshold_Low = sorted_clusters[next(iter(sorted_clusters))] // 3
        threshold_Medium = threshold_Low * 2
        
        danger_levels = []
        danger_scores = []
        
        for i in self.cluster_labels:
            if sorted_clusters[int(i)] >= threshold_Medium:
                danger_levels.append("High danger")
                danger_scores.append(3)
            elif sorted_clusters[int(i)] >= threshold_Low:
                danger_levels.append("Medium danger")
                danger_scores.append(2)
            else:
                danger_levels.append("Low danger")
                danger_scores.append(1)

        # Returning a list of danger levels respective to each complaint in original indexes and returning as integer values
        # Levels for testing and scores for functionality
        return danger_levels, danger_scores


    def population_clustering(self, coords):
        coords = np.array(coords)
        
        clustering = self.populatecluster_model.fit(coords)
        labels = clustering.labels_

        scores_mapping = {
            "High Density": 3,
            "Medium Density": 2,
            "Low Density": 1,
            "Noise (Possibly)": 0
        }

        density_map = {}
        density_scores = []
        
        for label in set(labels):
            if label == -1:
                # TODO: Although still clustered but not given a density map score, we must figure something more out for this
                continue
            
            counts = list(labels).count(label)
            # TODO: Change the threshold values or make them dynamic here
            if counts > 7:
                density_map[int(label)] = "High Density"
            elif counts > 3:
                density_map[int(label)] = "Medium Density"
            else:
                density_map[int(label)] = "Low Density"

        density_map = [density_map.get(l, "Noise (Possibly)") for l in labels]
        density_scores = [scores_mapping[element] for element in density_map]

        # Returning a list of density maps of the posted region respective to each complaint in original indexes
        # Levels for testing and scores for functionality
        return density_map, density_scores


    def socioeconomic_clustering(self, socioeconomic_metadata):
        # TODO: Check if any other metadata other than salary needed for this category
        results_levels = []
        results_scores = []
        
        for salary in socioeconomic_metadata:
            # TODO: Change the threshold values or make them dynamic here
            if salary >= 75000:
                results_levels.append("High Socio-Economic Area")
                results_scores.append(3)
            elif salary >= 30000:
                results_levels.append("Medium Socio-Economic Area")
                results_scores.append(2)
            else:
                results_levels.append("Low Socio-Economic Area")
                results_scores.append(1)

        # Returning a list of socioeconomic levels respective to each complaint in original indexes
        # Levels for testing and scores for functionality
        return results_levels, results_scores


    def finalize_scores(self, complaints, mode="default"):
        self.final_scores = []
        
        sentences = [i["content"] for i in complaints]
        coords = [[i["lat"], i["long"]] for i in complaints]
        salaries = [i["salary_level"] for i in complaints]
        
        weight_severity = 15
        weight_popn = 10
        weight_salary = 8
        bias_value = 1
        
        self.preprocess_texts(sentences)
        danger_scores = self.severity_clustering()[1]
        
        population_scores = self.population_clustering(coords)[1]
        
        socioeco_scores = self.socioeconomic_clustering(salaries)[1]
        
        # TODO: Discuss this final heuristic equation, if needing to be changed
        for i in range(len(complaints)):
            self.final_scores.append(weight_severity * danger_scores[i] + weight_popn * population_scores[i] + weight_salary * socioeco_scores[i] + bias_value)
        
        if mode == "testing":
            print(f"Severity scores: ", danger_scores)
            print(f"Population density scores: ", population_scores)
            print(f"Socio-economic scores: ", socioeco_scores)
        
        # Final heuristic scores
        return self.final_scores


    def plot_clusters(self, method="tsne"):
        # THIS WAS ONLY FOR PERSONAL TESTING; NOT NEARLY FINAL WAY FOR VISUALIZATION
        if method == "pca":
            transform = PCA(n_components=2)
            visualize = transform.fit_transform(self.embeddings)
        else:
            transform = TSNE(n_components=2, perplexity=3, random_state=42)
            visualize = transform.fit_transform(self.embeddings)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(visualize[:, 0], visualize[:, 1], c=self.cluster_labels, cmap="tab10", s=50, alpha=0.7)
        plt.title("Clustered Embeddings")
        plt.colorbar(scatter, label="Cluster")
        plt.show()
        
        return


if __name__ == '__main__':
    Cluster = ClusteringEmbeddings()
    
    complaints = [
        {"content": "Hello, my name is Dhruv!", "lat": -45.6789, "long": 123.4567, "salary_level": 45678},
        {"content": "No it is not...", "lat": 78.1234, "long": -56.7890, "salary_level": 78901},
        {"content": "Surya likes big, hairy men", "lat": 23.4567, "long": 178.2345, "salary_level": 23456},
        {"content": "Too many potholes on Arhaan avenue", "lat": -12.3456, "long": 45.6789, "salary_level": 56789},
        {"content": "Tanay street always congested with traffic", "lat": 67.8901, "long": -123.4567, "salary_level": 89012},
        {"content": "6 cars faced accident on Arnav Avenue", "lat": 34.5678, "long": 89.0123, "salary_level": 12345},
        {"content": "Fuck the cluster this belongs to", "lat": -78.9012, "long": -12.3456, "salary_level": 67890},
        {"content": "3 children run over at school crossing", "lat": 56.7890, "long": 167.8901, "salary_level": 34567},
        {"content": "No stop signs put up in front of Neel Elementary School", "lat": -23.4567, "long": -78.9012, "salary_level": 90123},
        {"content": "Which cluster will this belong to?", "lat": 45.6789, "long": 34.5678, "salary_level": 45678},
        {"content": "Do not look up Mother Horse Eyes", "lat": -67.8901, "long": 145.6789, "salary_level": 78901},
        {"content": "Too much graffiti on third street", "lat": 12.3456, "long": -89.0123, "salary_level": 23456},
        {"content": "Shops on Maiti boulevard have been ransacked and mugged", "lat": 89.0123, "long": 23.4567, "salary_level": 56789},
        {"content": "Broken streetlights on Vikas Road causing accidents", "lat": -34.5678, "long": -167.8901, "salary_level": 65432},
        {"content": "Flooding on Kiran Lane due to poor drainage", "lat": 78.9012, "long": 56.7890, "salary_level": 43210},
        {"content": "No pedestrian crossing near Ravi Market", "lat": -45.6789, "long": 123.4567, "salary_level": 87654},
        {"content": "Cracked sidewalks on Priya Street posing tripping hazard", "lat": 23.4567, "long": -34.5678, "salary_level": 34567},
        {"content": "Unmaintained speed bumps on Aryan Road damaging vehicles", "lat": -67.8901, "long": 89.0123, "salary_level": 78901},
        {"content": "Faded lane markings on Nisha Highway causing confusion", "lat": 12.3456, "long": -145.6789, "salary_level": 56789},
        {"content": "Overgrown trees blocking signs on Siddhant Avenue", "lat": 56.7890, "long": 23.4567, "salary_level": 23456},
        {"content": "No speed limit signs on Kunal Boulevard", "lat": -78.9012, "long": -56.7890, "salary_level": 90123},
        {"content": "Potholes on Meera Road causing flat tires", "lat": 34.5678, "long": 167.8901, "salary_level": 45678},
        {"content": "Inadequate road lighting on Tara Street at night", "lat": -23.4567, "long": -89.0123, "salary_level": 67890},
        {"content": "Construction debris left on Rohan Lane", "lat": 45.6789, "long": 12.3456, "salary_level": 12345},
        {"content": "No traffic signals at busy intersection of Anika Road", "lat": -12.3456, "long": 78.9012, "salary_level": 89012},
        {"content": "Damaged guardrails on Vihaan Bridge", "lat": 67.8901, "long": -123.4567, "salary_level": 34567},
        {"content": "Eroded road surface on Ishaan Street", "lat": -56.7890, "long": 34.5678, "salary_level": 56789},
        {"content": "Missing manhole covers on Aditi Avenue", "lat": 23.4567, "long": -167.8901, "salary_level": 78901}
    ]
    print(Cluster.finalize_scores(complaints))