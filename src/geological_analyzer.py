import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class GeologicalAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mwasifanwar = "mwasifanwar"
    
    def analyze_geological_suitability(self, geological_features, satellite_features):
        suitability_score = 0.0
        
        water_availability = geological_features[2]
        elevation_variance = geological_features[3]
        soil_quality = 1.0 - abs(geological_features[0] - 6.5) / 6.5
        
        structural_complexity = satellite_features.get('connected_structures', 0)
        flat_areas = satellite_features.get('flat_area_ratio', 0)
        
        suitability_score += water_availability * 0.3
        suitability_score += soil_quality * 0.25
        suitability_score += flat_areas * 0.2
        suitability_score += min(structural_complexity / 50, 1.0) * 0.15
        suitability_score += (1.0 - min(elevation_variance / 200, 1.0)) * 0.1
        
        return suitability_score
    
    def detect_resource_clusters(self, geological_data):
        if len(geological_data) == 0:
            return 0, 0.0
        
        X = geological_data[['lat', 'lon', 'mineral_diversity']].values
        X_scaled = self.scaler.fit_transform(X)
        
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(X_scaled)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_quality = np.mean([np.mean(X[labels == i, 2]) for i in range(n_clusters)]) if n_clusters > 0 else 0
        
        return n_clusters, cluster_quality
    
    def calculate_settlement_potential(self, geological_features, region_area_km2):
        base_potential = 0.0
        
        water_score = min(geological_features[2] * 10, 1.0)
        soil_score = 1.0 - abs(geological_features[0] - 7.0) / 7.0
        mineral_score = min(geological_features[1] / 5.0, 1.0)
        elevation_score = 1.0 - min(geological_features[3] / 300.0, 1.0)
        
        base_potential = (water_score * 0.35 + soil_score * 0.25 + 
                         mineral_score * 0.25 + elevation_score * 0.15)
        
        area_factor = min(region_area_km2 / 100.0, 1.0)
        
        return base_potential * area_factor
    
    def identify_geological_anomalies(self, geological_data):
        if len(geological_data) == 0:
            return 0.0
        
        features = ['soil_ph', 'mineral_diversity', 'water_availability', 'elevation']
        feature_matrix = geological_data[features].values
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        anomaly_scores = np.linalg.norm(scaled_features, axis=1)
        max_anomaly = np.max(anomaly_scores)
        
        return min(max_anomaly / 3.0, 1.0)

geo_analyzer = GeologicalAnalyzer()