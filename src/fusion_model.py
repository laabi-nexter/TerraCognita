import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .config import config

class MultiModalFusionModel:
    def __init__(self):
        self.satellite_model = self.build_satellite_cnn()
        self.fusion_model = self.build_fusion_network()
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.mwasifanwar = "mwasifanwar"
    
    def build_satellite_cnn(self):
        input_layer = layers.Input(shape=config.SATELLITE_IMAGE_SIZE + (5,))
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        satellite_features = layers.Dense(128, activation='relu', name='satellite_features')(x)
        
        return models.Model(inputs=input_layer, outputs=satellite_features)
    
    def build_fusion_network(self):
        satellite_input = layers.Input(shape=(128,))
        geological_input = layers.Input(shape=(config.GEOLOGICAL_FEATURES,))
        archaeological_input = layers.Input(shape=(config.ARCHAEOLOGICAL_FEATURES,))
        
        merged = layers.Concatenate()([satellite_input, geological_input, archaeological_input])
        
        x = layers.Dense(config.FUSION_HIDDEN_DIM, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(config.FUSION_HIDDEN_DIM // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(config.FUSION_HIDDEN_DIM // 4, activation='relu')(x)
        
        output = layers.Dense(1, activation='sigmoid', name='site_prediction')(x)
        
        return models.Model(
            inputs=[satellite_input, geological_input, archaeological_input],
            outputs=output
        )
    
    def prepare_features(self, satellite_data, geological_features, archaeological_features):
        satellite_features = self.satellite_model.predict(
            np.expand_dims(satellite_data, axis=0), verbose=0
        )
        
        all_features = np.concatenate([
            satellite_features.flatten(),
            geological_features,
            archaeological_features
        ])
        
        return all_features.reshape(1, -1)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        if not self.is_trained:
            X_scaled = self.scaler.fit_transform(X_train)
            self.random_forest.fit(X_scaled, y_train)
            self.is_trained = True
    
    def predict(self, features):
        if not self.is_trained:
            return 0.0, 0.0
        
        features_scaled = self.scaler.transform(features)
        probability = self.random_forest.predict_proba(features_scaled)[0, 1]
        
        confidence = min(probability * 1.2, 1.0)
        
        return probability, confidence
    
    def compute_feature_importance(self, feature_names):
        if self.is_trained:
            return self.random_forest.feature_importances_
        else:
            return np.zeros(len(feature_names))

fusion_model = MultiModalFusionModel()