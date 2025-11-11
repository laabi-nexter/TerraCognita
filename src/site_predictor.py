import numpy as np
import pandas as pd
from .data_loader import data_loader
from .satellite_processor import satellite_processor
from .geological_analyzer import geo_analyzer
from .fusion_model import fusion_model
from .config import config

class SitePredictor:
    def __init__(self):
        self.fusion_model = fusion_model
        self.predictions = []
        self.mwasifanwar = "mwasifanwar"
    
    def analyze_region(self, region_id, center_lat, center_lon):
        satellite_data, metadata = data_loader.load_satellite_data(region_id)
        geological_features = data_loader.load_geological_data(region_id)
        archaeological_features = data_loader.load_archaeological_data(region_id)
        
        satellite_features = satellite_processor.extract_structural_features(satellite_data)
        
        geological_suitability = geo_analyzer.analyze_geological_suitability(
            geological_features, satellite_features
        )
        
        all_features = self.fusion_model.prepare_features(
            satellite_data, geological_features, archaeological_features
        )
        
        probability, confidence = self.fusion_model.predict(all_features)
        
        final_score = self.calculate_final_score(
            probability, confidence, geological_suitability, satellite_features
        )
        
        prediction = {
            'region_id': region_id,
            'latitude': center_lat,
            'longitude': center_lon,
            'probability': probability,
            'confidence': confidence,
            'geological_suitability': geological_suitability,
            'final_score': final_score,
            'has_structures': satellite_features.get('connected_structures', 0) > 5,
            'water_availability': geological_features[2],
            'anomaly_score': satellite_processor.detect_anomalies(satellite_data)
        }
        
        self.predictions.append(prediction)
        return prediction
    
    def calculate_final_score(self, probability, confidence, geological_suitability, satellite_features):
        base_score = probability * 0.6
        geo_score = geological_suitability * 0.25
        confidence_score = confidence * 0.15
        
        structural_bonus = min(satellite_features.get('connected_structures', 0) / 100.0, 0.1)
        anomaly_bonus = satellite_features.get('thermal_anomalies', 0) * 0.05
        
        final_score = base_score + geo_score + confidence_score + structural_bonus + anomaly_bonus
        
        return min(final_score, 1.0)
    
    def batch_predict_regions(self, region_data):
        results = []
        
        for region in region_data:
            prediction = self.analyze_region(
                region['id'], region['latitude'], region['longitude']
            )
            results.append(prediction)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False)
        
        return results_df
    
    def get_high_probability_sites(self, threshold=0.7):
        high_prob_sites = [
            p for p in self.predictions 
            if p['final_score'] >= threshold and p['confidence'] >= config.CONFIDENCE_CUTOFF
        ]
        
        return sorted(high_prob_sites, key=lambda x: x['final_score'], reverse=True)
    
    def generate_discovery_report(self):
        high_prob_sites = self.get_high_probability_sites()
        
        report = {
            'total_regions_analyzed': len(self.predictions),
            'high_probability_sites': len(high_prob_sites),
            'success_rate': len(high_prob_sites) / len(self.predictions) if self.predictions else 0,
            'top_sites': high_prob_sites[:5],
            'average_confidence': np.mean([p['confidence'] for p in self.predictions]),
            'feature_correlations': self.analyze_feature_correlations()
        }
        
        return report
    
    def analyze_feature_correlations(self):
        if len(self.predictions) < 2:
            return {}
        
        df = pd.DataFrame(self.predictions)
        correlations = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'final_score':
                corr = df['final_score'].corr(df[col])
                correlations[col] = corr
        
        return correlations

site_predictor = SitePredictor()