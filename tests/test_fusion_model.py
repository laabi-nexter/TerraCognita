import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fusion_model import MultiModalFusionModel
from src.data_loader import DataLoader
import numpy as np

def test_fusion_model():
    loader = DataLoader()
    model = MultiModalFusionModel()
    
    satellite_data, _ = loader.load_satellite_data(1)
    geological_features = loader.load_geological_data(1)
    archaeological_features = loader.load_archaeological_data(1)
    
    features = model.prepare_features(
        satellite_data, geological_features, archaeological_features
    )
    
    probability, confidence = model.predict(features)
    
    print(f"Prediction probability: {probability:.3f}")
    print(f"Prediction confidence: {confidence:.3f}")
    print(f"Feature vector shape: {features.shape}")
    
    print("Fusion model test completed successfully")

if __name__ == "__main__":
    test_fusion_model()