import numpy as np
import pandas as pd
from src.fusion_model import MultiModalFusionModel
from src.data_loader import DataLoader
from src.config import config

def generate_training_data(num_samples=1000):
    X_train = []
    y_train = []
    
    data_loader = DataLoader()
    
    for i in range(num_samples):
        satellite_data, _ = data_loader.load_satellite_data(i)
        geological_features = data_loader.load_geological_data(i)
        archaeological_features = data_loader.load_archaeological_data(i)
        
        fusion_model = MultiModalFusionModel()
        features = fusion_model.prepare_features(
            satellite_data, geological_features, archaeological_features
        )
        
        X_train.append(features.flatten())
        
        label = 1 if (np.random.random() > 0.7 and 
                     geological_features[2] > 0.5 and 
                     len(archaeological_features) > 0) else 0
        y_train.append(label)
    
    return np.array(X_train), np.array(y_train)

def main():
    print("Generating training data...")
    X_train, y_train = generate_training_data(800)
    
    X_val, y_val = generate_training_data(200)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    
    model = MultiModalFusionModel()
    
    print("Training fusion model...")
    model.train(X_train, y_train, X_val, y_val)
    
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    importance_scores = model.compute_feature_importance(feature_names)
    
    top_features = np.argsort(importance_scores)[-10:]
    print("Top 10 most important features:")
    for idx in top_features[::-1]:
        print(f"Feature {idx}: {importance_scores[idx]:.4f}")
    
    print("Training completed successfully")

if __name__ == "__main__":
    main()