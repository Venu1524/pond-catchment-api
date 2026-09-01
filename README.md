# Pond Catchment Analysis Backend

Backend API for Assignment 1 - Phase 2.

## Features

- Accepts KML and KMZ contour maps.
- Extracts contour elevations from the uploaded input.
- Converts geographic coordinates to local metric coordinates.
- Builds an interpolated Digital Elevation Model (DEM).
- Calculates D8 flow directions.
- Identifies internal drainage basins.
- Selects a candidate pond location from the terrain.
- Estimates contributing catchment area.
- Returns structured JSON.
- No sample-map coordinates or results are hard-coded.

## API

### Health check

`GET /health`

### Analyze contour map

`POST /analyzeContour`

Upload a `.kml` or `.kmz` file using the form field:

`file`

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open Swagger documentation:

`http://127.0.0.1:8000/docs`

## Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/analyzeContour" \
  -F "file=@sample.kml"
```

## Deployment on Render

1. Push this project to GitHub.
2. Create a Render Web Service from the repository.
3. Render can use `render.yaml`.
4. The service URL will look like:

`https://your-service-name.onrender.com`

The assignment API route is:

`https://your-service-name.onrender.com/analyzeContour`

## Methodology

1. Parse KML/KMZ.
2. Extract contour coordinates and elevations.
3. Project longitude/latitude to local metres.
4. Interpolate contour points into a DEM.
5. Calculate D8 downhill flow.
6. Determine internal drainage basins.
7. Select a suitable low point associated with the largest internal basin.
8. Estimate catchment area from contributing raster cells.

## Important

The result is an engineering/algorithmic estimate intended for the assignment. It should not be treated as a field-survey or construction-grade hydrological analysis.
