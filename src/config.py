import os

class Config:
    SATELLITE_IMAGE_SIZE = (256, 256)
    GEOLOGICAL_FEATURES = 15
    ARCHAEOLOGICAL_FEATURES = 8
    
    FUSION_HIDDEN_DIM = 512
    PREDICTION_THRESHOLD = 0.75
    CONFIDENCE_CUTOFF = 0.85
    
    DATA_PATHS = {
        'satellite': 'data/satellite/',
        'geological': 'data/geological/',
        'archaeological': 'data/archaeological/',
        'processed': 'data/processed/'
    }
    
    MODEL_PARAMS = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2
    }
    
    SATELLITE_BANDS = ['visual', 'infrared', 'topographic']
    GEOLOGICAL_LAYERS = ['soil_composition', 'mineral_deposits', 'water_sources']

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

config = DevelopmentConfig()