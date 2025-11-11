import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.satellite_processor import SatelliteProcessor

def test_data_loading():
    loader = DataLoader()
    processor = SatelliteProcessor()
    
    satellite_data, metadata = loader.load_satellite_data(1)
    geological_features = loader.load_geological_data(1)
    archaeological_features = loader.load_archaeological_data(1)
    
    print(f"Satellite data shape: {satellite_data.shape}")
    print(f"Geological features: {len(geological_features)}")
    print(f"Archaeological features: {len(archaeological_features)}")
    
    satellite_features = processor.extract_structural_features(satellite_data)
    print(f"Extracted {len(satellite_features)} satellite features")
    
    print("Data loading test completed successfully")

if __name__ == "__main__":
    test_data_loading()