import numpy as np
import pandas as pd
from src.site_predictor import SitePredictor
from src.utils.visualization import vizTools

def main():
    predictor = SitePredictor()
    
    region_data = [
        {'id': 1, 'latitude': 34.0522, 'longitude': -118.2437},
        {'id': 2, 'latitude': 40.7128, 'longitude': -74.0060},
        {'id': 3, 'latitude': 51.5074, 'longitude': -0.1278},
        {'id': 4, 'latitude': 35.6762, 'longitude': 139.6503},
        {'id': 5, 'latitude': 30.0444, 'longitude': 31.2357},
        {'id': 6, 'latitude': 19.4326, 'longitude': -99.1332},
        {'id': 7, 'latitude': 39.9042, 'longitude': 116.4074},
        {'id': 8, 'latitude': 28.6139, 'longitude': 77.2090},
        {'id': 9, 'latitude': -33.8688, 'longitude': 151.2093},
        {'id': 10, 'latitude': 41.9028, 'longitude': 12.4964}
    ]
    
    print("Analyzing regions for potential archaeological sites...")
    results = predictor.batch_predict_regions(region_data)
    
    print("\n=== TERRA COGNITA DISCOVERY REPORT ===")
    print(f"Regions analyzed: {len(results)}")
    
    high_prob_sites = results[results['final_score'] >= 0.7]
    print(f"High probability sites found: {len(high_prob_sites)}")
    
    if len(high_prob_sites) > 0:
        print("\nTop discovery sites:")
        for idx, site in high_prob_sites.head(3).iterrows():
            print(f"Region {site['region_id']}: Score {site['final_score']:.3f} "
                  f"(Confidence: {site['confidence']:.3f})")
            print(f"  Location: {site['latitude']:.4f}, {site['longitude']:.4f}")
            print(f"  Geological Suitability: {site['geological_suitability']:.3f}")
            print(f"  Structures Detected: {site['has_structures']}")
            print()
    
    report = predictor.generate_discovery_report()
    
    print("Feature correlations with discovery potential:")
    for feature, corr in report['feature_correlations'].items():
        print(f"  {feature}: {corr:.3f}")
    
    interactive_map = vizTools.create_interactive_map(
        results, center_lat=20, center_lon=0, zoom_start=2
    )
    interactive_map.save('discovery_map.html')
    print("\nInteractive map saved as 'discovery_map.html'")

if __name__ == "__main__":
    main()