# AI-based Village Pond Planning System

Full-stack Geospatial & Hydrological Planning System for **CSD Assignment 1 - Phase 3** (VIVA & Demo).

- **Student**: Katari Venu (Roll: 12341110)
- **Institution**: Department of Computer Science & Engineering, IIT Bhilai

---

## Live Links (IIT Bhilai Campus Network)

| Service | Internal Host Port | Public Campus URL |
| :--- | :--- | :--- |
| **Working Front-End** | `6000` | [http://10.1.75.79:6289](http://10.1.75.79:6289) |
| **Backend REST API** | `3000` | [http://10.1.75.79:3289](http://10.1.75.79:3289) |
| **Swagger Interactive Docs** | `3000` | [http://10.1.75.79:3289/docs](http://10.1.75.79:3289/docs) |
| **Public YouTube Demo Video** | - | [https://www.youtube.com/playlist?list=PLONuDrOmfdhY](https://www.youtube.com/playlist?list=PLONuDrOmfdhY) |

---

## Features

- **Interactive Satellite Map GIS**: High-resolution Esri World Imagery with vector street layer switcher and bounding box drawing.
- **On-Map Land Area Selection**: Select any custom bounding box or agricultural parcel on the map (`POST /api/analyze-area`).
- **Contour Map Ingestion**: Upload KML/KMZ elevation contours (`POST /analyzeContour`).
- **Hydrological D8 Flow Routing**: Computes steepest downhill gradients and topological upstream flow accumulation.
- **Valley Floor Hazard Exclusion**: Enforces morphological dilation buffers on low-lying river channels to prevent embankment flood breach.
- **Multi-Criteria Pond Site Optimization**: Ranks candidates based on contributing catchment area, terrain elevation, and local slope.
- **Historical Climatology Integration**: Queries Open-Meteo Historical Weather Archive with automated fallback to official IMD Climatological Normals for Durg/Bhilai.
- **Runoff Volume Modeling**: Implements the Rational Method ($Q = C \cdot P \cdot A$) for annual and monsoon harvestable water volume.
- **Civil Engineering Pond Sizing**: Computes depth, length, width, side slopes ($1.5:1$), and excavation volume conforming to **IS: 4987**.
- **Dynamic Map Overlays**: Visualizes delineated catchment polygons, river exclusion zones, pond pin with live telemetry popup, and Chart.js monthly runoff profiles.

---

## Project Structure

```text
pond-catchment-api/
|-- main.py                    # Complete FastAPI backend application
|-- requirements.txt           # Python dependencies (fastapi, uvicorn, scipy, numpy)
|-- render.yaml                # Cloud deployment configuration
|-- README.md                  # System documentation & live deployment URLs
|-- contours_1m.kml            # Primary 1m contour dataset (Shivnath River Basin)
|-- start_services.sh          # Background service daemon startup script
|-- frontend/
|   `-- index.html             # Standalone Leaflet.js Web GIS frontend
`-- report/
    |-- report.tex             # ACM manuscript LaTeX source code
    `-- ui-screenshot.png      # Application user interface screenshot
```

---

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Backend API
uvicorn main:app --host 0.0.0.0 --port 3000 --reload

# 3. Start Frontend Web Server
cd frontend
python -m http.server 6000
```

Access the frontend at `http://localhost:6000` and Swagger docs at `http://localhost:3000/docs`.

---

## Important

The hydrological results represent an algorithmic planning estimate intended for regional watershed conservation and site screening. Geotechnical core-drilling and civil ground surveying should be conducted prior to physical earthmoving.
