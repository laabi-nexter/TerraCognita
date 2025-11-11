import numpy as np
import cv2
from skimage import feature, filters, measure

class SatelliteProcessor:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
    
    def extract_structural_features(self, satellite_data):
        visual_bands = satellite_data[:, :, :3]
        infrared_band = satellite_data[:, :, 3]
        topographic_band = satellite_data[:, :, 4]
        
        features = {}
        
        gray_image = np.mean(visual_bands, axis=2)
        
        edges = feature.canny(gray_image, sigma=2)
        features['edge_density'] = np.mean(edges)
        
        sobel_h = filters.sobel_h(gray_image)
        sobel_v = filters.sobel_v(gray_image)
        features['texture_variance'] = np.var(sobel_h) + np.var(sobel_v)
        
        labels = measure.label(edges)
        regions = measure.regionprops(labels)
        features['connected_structures'] = len(regions)
        
        infrared_variance = np.var(infrared_band)
        features['thermal_anomalies'] = infrared_variance
        
        topographic_features = self.analyze_topography(topographic_band)
        features.update(topographic_features)
        
        geometric_patterns = self.detect_geometric_patterns(visual_bands)
        features.update(geometric_patterns)
        
        return features
    
    def analyze_topography(self, topographic_data):
        features = {}
        
        features['elevation_range'] = np.max(topographic_data) - np.min(topographic_data)
        features['mean_slope'] = np.mean(np.abs(np.gradient(topographic_data)))
        
        flattened_areas = topographic_data < (np.mean(topographic_data) + np.std(topographic_data))
        features['flat_area_ratio'] = np.mean(flattened_areas)
        
        return features
    
    def detect_geometric_patterns(self, visual_bands):
        features = {}
        
        gray_image = np.mean(visual_bands, axis=2)
        
        circles = cv2.HoughCircles(
            (gray_image * 255).astype(np.uint8),
            cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        features['circular_structures'] = 0 if circles is None else len(circles[0])
        
        lines = cv2.HoughLinesP(
            (gray_image * 255).astype(np.uint8), 1, np.pi/180, threshold=50, 
            minLineLength=20, maxLineGap=10
        )
        
        features['linear_structures'] = 0 if lines is None else len(lines)
        
        return features
    
    def calculate_ndvi(self, satellite_data):
        visual_bands = satellite_data[:, :, :3]
        infrared_band = satellite_data[:, :, 3]
        
        red_band = visual_bands[:, :, 0]
        ndvi = (infrared_band - red_band) / (infrared_band + red_band + 1e-8)
        
        return np.nanmean(ndvi)
    
    def detect_anomalies(self, satellite_data):
        visual_bands = satellite_data[:, :, :3]
        infrared_band = satellite_data[:, :, 3]
        
        gray_image = np.mean(visual_bands, axis=2)
        
        local_std = filters.rank.standard_deviation(
            (gray_image * 255).astype(np.uint8), np.ones((5, 5))
        )
        
        anomaly_score = np.mean(local_std > np.percentile(local_std, 90))
        
        infrared_anomalies = np.mean(infrared_band > np.percentile(infrared_band, 95))
        
        return anomaly_score + infrared_anomalies

satellite_processor = SatelliteProcessor()