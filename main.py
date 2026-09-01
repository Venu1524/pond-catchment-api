import io
import math
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, binary_dilation


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Pond Catchment Analysis API",
    description=(
        "Terrain-based pond location and catchment analysis "
        "from KML/KMZ contour maps."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# BASIC ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Pond Catchment Analysis API is running",
        "version": "3.0.0",
        "endpoint": "/analyzeContour",
        "method": "POST",
        "accepted_files": ["KML", "KMZ"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# KML / KMZ
# =========================================================

def read_kml_file(
    filename: str,
    data: bytes
) -> bytes:

    filename = filename.lower()

    if filename.endswith(".kml"):
        return data

    if filename.endswith(".kmz"):

        try:

            with zipfile.ZipFile(
                io.BytesIO(data),
                "r"
            ) as archive:

                kml_files = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".kml")
                ]

                if not kml_files:
                    raise ValueError(
                        "No KML file found inside KMZ"
                    )

                doc_kml = next(
                    (
                        name
                        for name in kml_files
                        if name.lower().endswith("doc.kml")
                    ),
                    kml_files[0]
                )

                return archive.read(doc_kml)

        except zipfile.BadZipFile as exc:

            raise ValueError(
                "Invalid KMZ file"
            ) from exc

    raise ValueError(
        "Only KML and KMZ files are supported"
    )


# =========================================================
# COORDINATES
# =========================================================

def parse_coordinates(
    text: str
) -> List[Tuple[float, float]]:

    points = []

    if not text:
        return points

    text = text.replace(
        "\n",
        " "
    ).replace(
        "\r",
        " "
    )

    for token in text.split():

        values = token.split(",")

        if len(values) < 2:
            continue

        try:

            longitude = float(values[0])
            latitude = float(values[1])

            if (
                -180 <= longitude <= 180
                and -90 <= latitude <= 90
            ):

                points.append(
                    (
                        longitude,
                        latitude
                    )
                )

        except ValueError:
            continue

    return points


# =========================================================
# ELEVATION
# =========================================================

def get_elevation(
    placemark
) -> float:

    # Try <name>
    for child in list(placemark):

        tag = child.tag.split("}")[-1]

        if tag == "name" and child.text:

            try:
                return float(
                    child.text.strip()
                )
            except ValueError:
                pass

    # Try SimpleData
    for element in placemark.iter():

        tag = element.tag.split("}")[-1]

        if tag == "SimpleData" and element.text:

            try:
                return float(
                    element.text.strip()
                )
            except ValueError:
                pass

    # Try Data/value
    for element in placemark.iter():

        tag = element.tag.split("}")[-1]

        if tag == "value" and element.text:

            try:
                return float(
                    element.text.strip()
                )
            except ValueError:
                pass

    raise ValueError(
        "Could not determine contour elevation"
    )


# =========================================================
# PARSE CONTOURS
# =========================================================

def parse_contours(
    kml_bytes: bytes
):

    try:

        root = ET.fromstring(
            kml_bytes
        )

    except ET.ParseError as exc:

        raise ValueError(
            "Invalid KML XML"
        ) from exc

    contours = []
    all_points = []

    for placemark in root.iter():

        if (
            placemark.tag.split("}")[-1]
            != "Placemark"
        ):
            continue

        try:

            elevation = get_elevation(
                placemark
            )

        except ValueError:

            continue

        coordinates = []

        for element in placemark.iter():

            tag = element.tag.split("}")[-1]

            if (
                tag == "coordinates"
                and element.text
            ):

                coordinates.extend(
                    parse_coordinates(
                        element.text
                    )
                )

        if len(coordinates) < 2:
            continue

        contours.append(
            {
                "elevation": elevation,
                "coordinates": coordinates
            }
        )

        for longitude, latitude in coordinates:

            all_points.append(
                (
                    longitude,
                    latitude,
                    elevation
                )
            )

    if len(all_points) < 10:

        raise ValueError(
            "Not enough contour data found"
        )

    return contours, all_points


# =========================================================
# GEO CONVERSION
# =========================================================

def lonlat_to_local(
    longitude,
    latitude,
    longitude0,
    latitude0
):

    earth_radius = 6371000.0

    x = (
        math.radians(
            longitude - longitude0
        )
        * earth_radius
        * math.cos(
            math.radians(latitude0)
        )
    )

    y = (
        math.radians(
            latitude - latitude0
        )
        * earth_radius
    )

    return x, y


def local_to_lonlat(
    x,
    y,
    longitude0,
    latitude0
):

    earth_radius = 6371000.0

    latitude = (
        latitude0
        + math.degrees(
            y / earth_radius
        )
    )

    longitude = (
        longitude0
        + math.degrees(
            x
            / (
                earth_radius
                * math.cos(
                    math.radians(latitude0)
                )
            )
        )
    )

    return longitude, latitude


# =========================================================
# BUILD DEM
# =========================================================

def build_dem(
    all_points
):

    data = np.asarray(
        all_points,
        dtype=float
    )

    longitude = data[:, 0]
    latitude = data[:, 1]
    elevation = data[:, 2]

    longitude0 = float(
        np.mean(longitude)
    )

    latitude0 = float(
        np.mean(latitude)
    )

    local_points = np.array(
        [
            lonlat_to_local(
                lon,
                lat,
                longitude0,
                latitude0
            )
            for lon, lat
            in zip(
                longitude,
                latitude
            )
        ]
    )

    # -----------------------------------------------------
    # Remove duplicate points
    # -----------------------------------------------------

    point_dict = {}

    for (
        (x, y),
        z
    ) in zip(
        local_points,
        elevation
    ):

        key = (
            round(float(x), 3),
            round(float(y), 3)
        )

        point_dict.setdefault(
            key,
            []
        ).append(
            float(z)
        )

    xy = np.array(
        list(point_dict.keys())
    )

    z = np.array(
        [
            np.mean(values)
            for values
            in point_dict.values()
        ]
    )

    if len(xy) < 10:

        raise ValueError(
            "Insufficient terrain points"
        )

    # -----------------------------------------------------
    # Bounds
    # -----------------------------------------------------

    min_x = float(
        np.min(xy[:, 0])
    )

    max_x = float(
        np.max(xy[:, 0])
    )

    min_y = float(
        np.min(xy[:, 1])
    )

    max_y = float(
        np.max(xy[:, 1])
    )

    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:

        raise ValueError(
            "Invalid terrain dimensions"
        )

    # -----------------------------------------------------
    # DEM grid
    # -----------------------------------------------------

    grid_size = 180

    xs = np.linspace(
        min_x,
        max_x,
        grid_size
    )

    ys = np.linspace(
        min_y,
        max_y,
        grid_size
    )

    X, Y = np.meshgrid(
        xs,
        ys
    )

    # -----------------------------------------------------
    # Interpolation
    # -----------------------------------------------------

    Z = griddata(
        xy,
        z,
        (X, Y),
        method="linear"
    )

    nearest = griddata(
        xy,
        z,
        (X, Y),
        method="nearest"
    )

    missing = np.isnan(Z)

    Z[missing] = nearest[missing]

    # Small smoothing
    Z = gaussian_filter(
        Z,
        sigma=1.0
    )

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "xs": xs,
        "ys": ys,
        "lon0": longitude0,
        "lat0": latitude0,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
    }


# =========================================================
# D8 FLOW
# =========================================================

def calculate_flow_direction(
    Z
):

    rows, cols = Z.shape

    receiver = np.full(
        (
            rows,
            cols,
            2
        ),
        -1,
        dtype=np.int32
    )

    directions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    distances = [
        math.sqrt(2),
        1,
        math.sqrt(2),
        1,
        1,
        math.sqrt(2),
        1,
        math.sqrt(2)
    ]

    for r in range(rows):

        for c in range(cols):

            current = Z[r, c]

            best_slope = 0.0
            best = None

            for (
                (dr, dc),
                distance
            ) in zip(
                directions,
                distances
            ):

                nr = r + dr
                nc = c + dc

                if (
                    nr < 0
                    or nr >= rows
                    or nc < 0
                    or nc >= cols
                ):
                    continue

                drop = (
                    current
                    - Z[nr, nc]
                )

                if drop <= 0:
                    continue

                slope = (
                    drop / distance
                )

                if slope > best_slope:

                    best_slope = slope

                    best = (
                        nr,
                        nc
                    )

            if best is not None:

                receiver[
                    r,
                    c
                ] = best

    return receiver


# =========================================================
# UPSTREAM AREA
# =========================================================

def calculate_upstream_area(
    receiver
):

    rows, cols = receiver.shape[:2]

    upstream = np.ones(
        (
            rows,
            cols
        ),
        dtype=np.int64
    )

    indegree = np.zeros(
        (
            rows,
            cols
        ),
        dtype=np.int32
    )

    # Count incoming cells
    for r in range(rows):

        for c in range(cols):

            nr, nc = receiver[r, c]

            if (
                nr >= 0
                and nc >= 0
            ):

                indegree[
                    nr,
                    nc
                ] += 1

    # Cells with no upstream contributors
    queue = []

    for r in range(rows):

        for c in range(cols):

            if indegree[r, c] == 0:

                queue.append(
                    (
                        r,
                        c
                    )
                )

    head = 0

    while head < len(queue):

        r, c = queue[head]

        head += 1

        nr, nc = receiver[r, c]

        if (
            nr < 0
            or nc < 0
        ):
            continue

        upstream[
            nr,
            nc
        ] += upstream[
            r,
            c
        ]

        indegree[
            nr,
            nc
        ] -= 1

        if indegree[
            nr,
            nc
        ] == 0:

            queue.append(
                (
                    int(nr),
                    int(nc)
                )
            )

    return upstream


# =========================================================
# LAND / RIVER EXCLUSION
# =========================================================

def create_valley_exclusion_mask(
    Z
):
    """
    Creates a terrain-derived exclusion zone.

    Since the contour KML does not explicitly contain a
    river polygon, the lowest connected valley-floor
    terrain is treated as a possible water/channel zone.

    This is deliberately conservative so the candidate
    pond location is placed on higher interior land.
    """

    minimum = float(
        np.min(Z)
    )

    maximum = float(
        np.max(Z)
    )

    elevation_range = (
        maximum - minimum
    )

    # -----------------------------------------------------
    # Lowest terrain zone
    #
    # Avoid approximately the lowest 15% of terrain,
    # while also using an absolute elevation difference.
    # -----------------------------------------------------

    percentile_threshold = float(
        np.percentile(
            Z,
            15
        )
    )

    absolute_threshold = (
        minimum
        + max(
            3.0,
            elevation_range * 0.10
        )
    )

    low_threshold = max(
        percentile_threshold,
        absolute_threshold
    )

    low_terrain = (
        Z <= low_threshold
    )

    # -----------------------------------------------------
    # Expand the exclusion zone.
    #
    # This creates a safety buffer around the valley floor
    # so a point immediately beside the river is avoided.
    # -----------------------------------------------------

    exclusion = binary_dilation(
        low_terrain,
        iterations=7
    )

    return exclusion, low_threshold


# =========================================================
# POND LOCATION
# =========================================================

def choose_pond_location(
    Z,
    receiver
):

    rows, cols = Z.shape

    minimum = float(
        np.min(Z)
    )

    maximum = float(
        np.max(Z)
    )

    elevation_range = (
        maximum - minimum
    )

    # -----------------------------------------------------
    # Upstream contributing cells
    # -----------------------------------------------------

    upstream = calculate_upstream_area(
        receiver
    )

    # -----------------------------------------------------
    # Exclude valley / river-like low terrain
    # -----------------------------------------------------

    exclusion_mask, exclusion_threshold = (
        create_valley_exclusion_mask(
            Z
        )
    )

    # -----------------------------------------------------
    # Candidate elevation band
    #
    # Candidate must be above the valley floor but still
    # relatively low compared with the rest of the terrain.
    # -----------------------------------------------------

    minimum_candidate_elevation = (
        minimum
        + max(
            4.0,
            elevation_range * 0.12
        )
    )

    maximum_candidate_elevation = (
        minimum
        + max(
            12.0,
            elevation_range * 0.40
        )
    )

    # -----------------------------------------------------
    # Map boundary margin
    # -----------------------------------------------------

    margin_rows = max(
        8,
        int(rows * 0.10)
    )

    margin_cols = max(
        8,
        int(cols * 0.10)
    )

    candidates = []

    # -----------------------------------------------------
    # Search candidate land locations
    # -----------------------------------------------------

    for r in range(
        margin_rows,
        rows - margin_rows
    ):

        for c in range(
            margin_cols,
            cols - margin_cols
        ):

            elevation = float(
                Z[r, c]
            )

            # Avoid river/valley floor
            if exclusion_mask[r, c]:
                continue

            # Elevation range
            if (
                elevation
                < minimum_candidate_elevation
            ):
                continue

            if (
                elevation
                > maximum_candidate_elevation
            ):
                continue

            contributing_cells = int(
                upstream[r, c]
            )

            if contributing_cells < 10:
                continue

            # -------------------------------------------------
            # Local slope
            # -------------------------------------------------

            r1 = max(
                0,
                r - 1
            )

            r2 = min(
                rows - 1,
                r + 1
            )

            c1 = max(
                0,
                c - 1
            )

            c2 = min(
                cols - 1,
                c + 1
            )

            dz_x = (
                Z[r, c2]
                - Z[r, c1]
            )

            dz_y = (
                Z[r2, c]
                - Z[r1, c]
            )

            slope = math.sqrt(
                dz_x ** 2
                + dz_y ** 2
            )

            # -------------------------------------------------
            # Avoid extremely steep areas
            # -------------------------------------------------

            if slope > 8.0:
                continue

            candidates.append(
                {
                    "row": r,
                    "col": c,
                    "elevation": elevation,
                    "upstream_cells": contributing_cells,
                    "slope": float(slope)
                }
            )

    # =====================================================
    # RELAXED SEARCH
    # =====================================================

    if not candidates:

        for r in range(
            margin_rows,
            rows - margin_rows
        ):

            for c in range(
                margin_cols,
                cols - margin_cols
            ):

                if exclusion_mask[r, c]:
                    continue

                elevation = float(
                    Z[r, c]
                )

                if elevation < (
                    minimum
                    + 2.0
                ):
                    continue

                contributing_cells = int(
                    upstream[r, c]
                )

                if contributing_cells < 5:
                    continue

                candidates.append(
                    {
                        "row": r,
                        "col": c,
                        "elevation": elevation,
                        "upstream_cells": contributing_cells,
                        "slope": 0.0
                    }
                )

    # =====================================================
    # FINAL FALLBACK
    # =====================================================

    if not candidates:

        # Find lowest point that is NOT in the exclusion zone
        valid = np.where(
            ~exclusion_mask
        )

        if len(valid[0]) > 0:

            best_index = int(
                np.argmin(
                    Z[
                        valid[0],
                        valid[1]
                    ]
                )
            )

            r = int(
                valid[0][best_index]
            )

            c = int(
                valid[1][best_index]
            )

            return (
                r,
                c,
                int(upstream[r, c])
            )

        # Last possible fallback
        interior = Z[
            margin_rows:rows - margin_rows,
            margin_cols:cols - margin_cols
        ]

        index = np.unravel_index(
            np.argmin(interior),
            interior.shape
        )

        r = (
            index[0]
            + margin_rows
        )

        c = (
            index[1]
            + margin_cols
        )

        return (
            r,
            c,
            int(upstream[r, c])
        )

    # =====================================================
    # SCORING
    # =====================================================

    max_upstream = max(
        item["upstream_cells"]
        for item in candidates
    )

    min_elevation = min(
        item["elevation"]
        for item in candidates
    )

    max_elevation = max(
        item["elevation"]
        for item in candidates
    )

    elevation_span = (
        max_elevation
        - min_elevation
    )

    if elevation_span <= 0:
        elevation_span = 1.0

    for item in candidates:

        # ---------------------------------------------
        # Large catchment = good
        # ---------------------------------------------

        catchment_score = (
            item["upstream_cells"]
            / max_upstream
        )

        # ---------------------------------------------
        # Lower land = good
        # ---------------------------------------------

        elevation_score = 1.0 - (
            (
                item["elevation"]
                - min_elevation
            )
            / elevation_span
        )

        # ---------------------------------------------
        # Moderate slope = good
        # ---------------------------------------------

        slope_penalty = min(
            item["slope"] / 8.0,
            1.0
        )

        slope_score = (
            1.0
            - slope_penalty
        )

        # ---------------------------------------------
        # Final score
        # ---------------------------------------------

        item["score"] = (
            0.65 * catchment_score
            + 0.25 * elevation_score
            + 0.10 * slope_score
        )

    # -----------------------------------------------------
    # Highest score
    # -----------------------------------------------------

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = candidates[0]

    return (
        selected["row"],
        selected["col"],
        selected["upstream_cells"]
    )


# =========================================================
# STATISTICS
# =========================================================

def calculate_statistics(
    dem,
    pond_row,
    pond_col,
    catchment_cells
):

    X = dem["X"]
    Y = dem["Y"]
    Z = dem["Z"]

    xs = dem["xs"]
    ys = dem["ys"]

    dx = abs(
        xs[1] - xs[0]
    )

    dy = abs(
        ys[1] - ys[0]
    )

    cell_area = (
        dx * dy
    )

    area_m2 = (
        catchment_cells
        * cell_area
    )

    area_hectares = (
        area_m2 / 10000.0
    )

    area_sq_km = (
        area_m2 / 1000000.0
    )

    pond_x = float(
        X[
            pond_row,
            pond_col
        ]
    )

    pond_y = float(
        Y[
            pond_row,
            pond_col
        ]
    )

    longitude, latitude = (
        local_to_lonlat(
            pond_x,
            pond_y,
            dem["lon0"],
            dem["lat0"]
        )
    )

    return {
        "latitude": round(
            latitude,
            7
        ),
        "longitude": round(
            longitude,
            7
        ),
        "elevation_m": round(
            float(
                Z[
                    pond_row,
                    pond_col
                ]
            ),
            3
        ),
        "catchment_area_m2": round(
            area_m2,
            2
        ),
        "catchment_area_hectares": round(
            area_hectares,
            4
        ),
        "catchment_area_sq_km": round(
            area_sq_km,
            6
        ),
        "grid_cell_area_m2": round(
            cell_area,
            4
        ),
        "contributing_cells": int(
            catchment_cells
        ),
    }


# =========================================================
# MAIN API
# =========================================================

@app.post(
    "/analyzeContour"
)
async def analyze_contour(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file name provided"
        )

    filename = file.filename.lower()

    if not (
        filename.endswith(".kml")
        or filename.endswith(".kmz")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a KML or KMZ file"
            )
        )

    try:

        # ---------------------------------------------
        # Read file
        # ---------------------------------------------

        file_data = await file.read()

        if not file_data:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # ---------------------------------------------
        # Read KML/KMZ
        # ---------------------------------------------

        kml_bytes = read_kml_file(
            file.filename,
            file_data
        )

        # ---------------------------------------------
        # Parse contours
        # ---------------------------------------------

        contours, all_points = (
            parse_contours(
                kml_bytes
            )
        )

        # ---------------------------------------------
        # DEM
        # ---------------------------------------------

        dem = build_dem(
            all_points
        )

        # ---------------------------------------------
        # Flow
        # ---------------------------------------------

        receiver = (
            calculate_flow_direction(
                dem["Z"]
            )
        )

        # ---------------------------------------------
        # Pond location
        # ---------------------------------------------

        pond_row, pond_col, catchment_cells = (
            choose_pond_location(
                dem["Z"],
                receiver
            )
        )

        # ---------------------------------------------
        # Statistics
        # ---------------------------------------------

        statistics = calculate_statistics(
            dem,
            pond_row,
            pond_col,
            catchment_cells
        )

        # ---------------------------------------------
        # Contour interval
        # ---------------------------------------------

        elevations = [
            contour["elevation"]
            for contour in contours
        ]

        unique_elevations = sorted(
            set(elevations)
        )

        contour_interval = 0

        if len(unique_elevations) > 1:

            differences = np.diff(
                unique_elevations
            )

            differences = differences[
                differences > 0
            ]

            if len(differences) > 0:

                contour_interval = float(
                    np.min(
                        differences
                    )
                )

        # ---------------------------------------------
        # Response
        # ---------------------------------------------

        return {

            "success": True,

            "input": {

                "filename": file.filename,

                "file_type": (
                    "KMZ"
                    if filename.endswith(".kmz")
                    else "KML"
                ),

                "contour_count": len(
                    contours
                ),

                "terrain_points": len(
                    all_points
                ),
            },

            "terrain": {

                "minimum_elevation_m": round(
                    float(
                        np.min(
                            dem["Z"]
                        )
                    ),
                    3
                ),

                "maximum_elevation_m": round(
                    float(
                        np.max(
                            dem["Z"]
                        )
                    ),
                    3
                ),

                "elevation_range_m": round(
                    float(
                        np.max(
                            dem["Z"]
                        )
                        -
                        np.min(
                            dem["Z"]
                        )
                    ),
                    3
                ),

                "contour_interval_m": round(
                    contour_interval,
                    3
                ),
            },

            "pond_location": {

                "latitude": statistics[
                    "latitude"
                ],

                "longitude": statistics[
                    "longitude"
                ],

                "elevation_m": statistics[
                    "elevation_m"
                ],

                "selection_method": (
                    "Low-elevation interior land "
                    "candidate outside the terrain-derived "
                    "valley exclusion zone, ranked by "
                    "upstream contributing catchment area"
                ),
            },

            "catchment": {

                "area_m2": statistics[
                    "catchment_area_m2"
                ],

                "area_hectares": statistics[
                    "catchment_area_hectares"
                ],

                "area_sq_km": statistics[
                    "catchment_area_sq_km"
                ],

                "contributing_grid_cells": statistics[
                    "contributing_cells"
                ],
            },

            "methodology": [

                "Contour elevations are extracted "
                "automatically from the uploaded KML/KMZ.",

                "Geographic coordinates are converted "
                "to a local metric coordinate system.",

                "A Digital Elevation Model is generated "
                "using interpolation.",

                "D8 flow direction is calculated from "
                "the terrain surface.",

                "Upstream contributing cells are "
                "calculated for potential pond locations.",

                "The lowest terrain valley-floor zone "
                "is treated as a possible river/channel "
                "zone because the supplied contour map "
                "does not explicitly contain water "
                "boundaries.",

                "A safety buffer is applied around the "
                "terrain-derived valley zone.",

                "Low-elevation interior land candidates "
                "outside the exclusion zone are evaluated.",

                "Candidates are ranked using upstream "
                "catchment contribution, elevation and "
                "local terrain slope.",

                "The highest-ranked candidate is returned "
                "as the proposed pond location.",

                "Catchment area is estimated from the "
                "number of contributing raster cells.",
            ],
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Terrain analysis failed: "
                f"{str(exc)}"
            )
        )


# =========================================================
# LOCAL SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 