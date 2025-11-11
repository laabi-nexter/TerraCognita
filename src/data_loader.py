import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import StandardScaler
from .config import config
from .utils.geospatial_tools import geoTools

class DataLoader:
    def __init__(self):
        self.satellite_scaler = StandardScaler()
        self.geological_scaler = StandardScaler()
        self.mwasifanwar = "mwasifanwar"
    
    def load_satellite_data(self, region_id):
        try:
            file_path = f"{config.DATA_PATHS['satellite']}region_{region_id}.tif"
            with rasterio.open(file_path) as src:
                satellite_data = src.read()
                metadata = src.meta
            
            processed_data = self.preprocess_satellite(satellite_data)
            return processed_data, metadata
        
        except Exception as e:
            return self.generate_simulated_satellite(region_id)
    
    def preprocess_satellite(self, satellite_data):
        normalized_data = satellite_data.astype(np.float32) / 255.0
        
        if len(normalized_data.shape) == 3:
            normalized_data = np.transpose(normalized_data, (1, 2, 0))
        
        target_size = config.SATELLITE_IMAGE_SIZE
        if normalized_data.shape[:2] != target_size:
            from skimage.transform import resize
            normalized_data = resize(normalized_data, target_size + (normalized_data.shape[2],))
        
        return normalized_data
    
    def load_geological_data(self, region_id):
        try:
            file_path = f"{config.DATA_PATHS['geological']}region_{region_id}.csv"
            geo_data = pd.read_csv(file_path)
            
            features = self.extract_geological_features(geo_data)
            return features
        
        except Exception as e:
            return self.generate_simulated_geological(region_id)
    
    def extract_geological_features(self, geo_data):
        features = np.zeros(config.GEOLOGICAL_FEATURES)
        
        if len(geo_data) > 0:
            features[0] = geo_data['soil_ph'].mean()
            features[1] = geo_data['mineral_diversity'].max()
            features[2] = geo_data['water_availability'].mean()
            features[3] = geo_data['elevation'].std()
            features[4] = geo_data['slope_angle'].mean()
            features[5] = len(geo_data['unique_rock_types'].unique())
            features[6] = geo_data['erosion_level'].mean()
            features[7] = geo_data['vegetation_density'].mean()
        
        return features
    
    def load_archaeological_data(self, region_id):
        try:
            file_path = f"{config.DATA_PATHS['archaeological']}region_{region_id}.csv"
            arch_data = pd.read_csv(file_path)
            
            features = self.extract_archaeological_features(arch_data)
            return features
        
        except Exception as e:
            return np.zeros(config.ARCHAEOLOGICAL_FEATURES)
    
    def extract_archaeological_features(self, arch_data):
        features = np.zeros(config.ARCHAEOLOGICAL_FEATURES)
        
        if len(arch_data) > 0:
            features[0] = len(arch_data)
            features[1] = arch_data['artifact_density'].mean()
            features[2] = arch_data['structure_complexity'].max()
            features[3] = arch_data['age_estimate'].mean()
            features[4] = arch_data['material_diversity'].max()
        
        return features
    
    def generate_simulated_satellite(self, region_id):
        np.random.seed(region_id)
        simulated_data = np.random.rand(*config.SATELLITE_IMAGE_SIZE, 5)
        simulated_data[:, :, 3] *= 2
        simulated_data[:, :, 4] = np.random.rand(*config.SATELLITE_IMAGE_SIZE) * 1000
        
        metadata = {'transform': None, 'crs': None}
        return simulated_data, metadata
    
    def generate_simulated_geological(self, region_id):
        np.random.seed(region_id)
        features = np.random.rand(config.GEOLOGICAL_FEATURES)
        features[1] *= 10
        features[3] *= 500
        return features

data_loader = DataLoader()