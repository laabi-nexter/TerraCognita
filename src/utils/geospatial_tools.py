import numpy as np
import rasterio
from pyproj import Transformer

class GeospatialTools:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
    
    def lat_lon_to_utm(self, latitude, longitude):
        zone = int((longitude + 180) / 6) + 1
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{zone}")
        return transformer.transform(latitude, longitude)
    
    def utm_to_lat_lon(self, easting, northing, zone):
        transformer = Transformer.from_crs(f"EPSG:326{zone}", "EPSG:4326")
        return transformer.transform(easting, northing)
    
    def calculate_slope(self, elevation_data):
        x, y = np.gradient(elevation_data)
        slope = np.sqrt(x**2 + y**2)
        return np.degrees(np.arctan(slope))
    
    def extract_region_features(self, geodata, center_lat, center_lon, radius_km):
        features = {}
        
        lat_range = radius_km / 111.0
        lon_range = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        mask = (
            (geodata['lat'] >= center_lat - lat_range) &
            (geodata['lat'] <= center_lat + lat_range) &
            (geodata['lon'] >= center_lon - lon_range) &
            (geodata['lon'] <= center_lon + lon_range)
        )
        
        region_data = geodata[mask]
        
        if len(region_data) > 0:
            features['elevation_variance'] = np.var(region_data['elevation'])
            features['water_proximity'] = np.min(region_data['distance_to_water'])
            features['soil_diversity'] = len(np.unique(region_data['soil_type']))
        
        return features
    
    def load_geotiff(self, filepath):
        with rasterio.open(filepath) as src:
            data = src.read()
            transform = src.transform
            crs = src.crs
        return data, transform, crs

geoTools = GeospatialTools()