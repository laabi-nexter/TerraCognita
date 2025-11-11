import matplotlib.pyplot as plt
import numpy as np
import folium

class VisualizationTools:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
    
    def plot_satellite_composite(self, satellite_data, title="Satellite Composite"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        visual_bands = satellite_data[:, :, :3]
        infrared = satellite_data[:, :, 3]
        topographic = satellite_data[:, :, 4]
        
        axes[0].imshow(visual_bands)
        axes[0].set_title('Visual Spectrum')
        
        axes[1].imshow(infrared, cmap='hot')
        axes[1].set_title('Infrared')
        
        axes[2].imshow(topographic, cmap='terrain')
        axes[2].set_title('Topography')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def create_interactive_map(self, predictions, center_lat=0, center_lon=0, zoom_start=6):
        site_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        
        for idx, pred in predictions.iterrows():
            if pred['confidence'] > 0.7:
                color = 'red' if pred['probability'] > 0.8 else 'orange'
                
                folium.CircleMarker(
                    location=[pred['latitude'], pred['longitude']],
                    radius=pred['probability'] * 20,
                    popup=f"Score: {pred['probability']:.2f}",
                    color=color,
                    fill=True
                ).add_to(site_map)
        
        return site_map
    
    def plot_feature_importance(self, feature_names, importance_scores):
        indices = np.argsort(importance_scores)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance for Site Prediction")
        plt.bar(range(len(feature_names)), importance_scores[indices])
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        return plt.gcf()

vizTools = VisualizationTools()