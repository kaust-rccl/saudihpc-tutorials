import ee
import os
import pandas as pd

# 1. Initialize Earth Engine with your Google Cloud project ID
project_id = os.environ.get('EE_PROJECT', 'YOUR-GCP-PROJECT-ID')  # your Google Cloud project id
ee.Initialize(project=project_id)
print(f"Initialized Earth Engine with project: {project_id}")

# 2. Load the GFSAD1000 V1 image asset
image = ee.Image('USGS/GFSAD1000_V1').select('landcover')

# 3. Define a region of interest (e.g., an agricultural region in West Africa: [West, South, East, North])
roi = ee.Geometry.Rectangle([-17.5, 13.0, -16.5, 14.0])

# 4. Sample points from the image to build a tabular dataset (features + landcover target label)
# This extracts pixel values across a regular grid within your ROI
sample_points = image.sample(
    region=roi,
    scale=1000,          # 1km resolution matching GFSAD grid
    numPixels=2000,      # Number of sample points to generate
    geometries=True
)

# 5. Fetch the sampled data from Earth Engine servers to client memory
print("Fetching sample points from Earth Engine...")
feature_collection = sample_points.getInfo()

# 6. Parse the features into a clean Pandas DataFrame
rows = []
for feature in feature_collection['features']:
    props = feature['properties']
    coords = feature['geometry']['coordinates']

    # Flatten properties and append coordinate data
    row = {
        'longitude': coords[0],
        'latitude': coords[1],
        'landcover': props.get('landcover', 0)
    }
    rows.append(row)

df = pd.DataFrame(rows)

# Clean up missing or unclassified values (0 typically represents background/non-cropland)
df = df[df['landcover'] > 0].reset_index(drop=True)

# 7. Save the dataset directly into your current working directory (pwd)
output_filename = "gfsad1000_tutorial_input.csv"
local_path = os.path.join(os.getcwd(), output_filename)
df.to_csv(local_path, index=False)

print(f"Dataset successfully downloaded and saved to: {local_path}")
print(f"Total valid training samples: {len(df)}")
print(df.head())
