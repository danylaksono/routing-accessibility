import ast
import json
import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import shape


CHARGEPOINTS_PATH = "london_chargepoints.geojson"
BOUNDARY_PATH = "London_Boroughs.gpkg"
GRAPH_CACHE = "london_drive_graph.pkl"
OUTPUT_GEOJSON = "london_isochrones.geojson"
OUTPUT_MAP = "london_iso_map.png"
OUTPUT_FILTERED_CHARGEPOINTS = "london_chargepoints_filtered.geojson"
CRS = 27700
DOWNLOAD_BUFFER_METERS = 5000
GRID_SIZE_METERS = 25
INTERVAL_MINUTES = 2
GRID_ROW_BATCH_SIZE = 200
NEAREST_NODE_BATCH_SIZE = 200_000
FILTER_PUBLIC_CHARGERS_ONLY = True
FILTER_OPERATIONAL_CHARGERS_ONLY = True
FILTER_RAPID_CHARGERS_ONLY = True
PUBLIC_USAGE_TYPE_IDS = {1, 4, 5, 7}
OPERATIONAL_STATUS_TYPE_IDS = {10, 20, 50, 75}
RAPID_POWER_THRESHOLD_KW = 43
USAGE_TYPE_LABELS = {
    1: "Public",
    4: "Public - Membership Required",
    5: "Public - Pay At Location",
    7: "Public - Notice Required",
}
STATUS_TYPE_LABELS = {
    10: "Currently Available",
    20: "Currently In Use",
    50: "Operational",
    75: "Partly Operational",
    0: "Unknown",
    30: "Temporarily Unavailable",
    100: "Not Operational",
    150: "Planned",
    200: "Removed",
}


def selected_charger_label():
    descriptors = []
    if FILTER_PUBLIC_CHARGERS_ONLY:
        descriptors.append("public")
    if FILTER_OPERATIONAL_CHARGERS_ONLY:
        descriptors.append("operational")
    if FILTER_RAPID_CHARGERS_ONLY:
        descriptors.append("rapid")
    return " ".join(descriptors + ["chargepoint"]).strip()


def load_drive_graph(boundary_polygon_proj):
    buffered_wgs84 = (
        gpd.GeoSeries([boundary_polygon_proj.buffer(DOWNLOAD_BUFFER_METERS)], crs=CRS)
        .to_crs(4326)
        .iloc[0]
    )

    if os.path.exists(GRAPH_CACHE):
        print("Loading road network from cache...")
        with open(GRAPH_CACHE, "rb") as file_handle:
            graph = pickle.load(file_handle)
    else:
        print("Downloading road network (buffered, this might take a while)...")
        graph = ox.graph_from_polygon(buffered_wgs84, network_type="drive")
        print("Saving road network to cache...")
        with open(GRAPH_CACHE, "wb") as file_handle:
            pickle.dump(graph, file_handle)

    print("Projecting graph and adding travel times...")
    graph = ox.project_graph(graph, to_crs=CRS)
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def parse_connections(raw_value):
    text = "" if raw_value is None else str(raw_value).strip()
    if not text or text in {"[]", "[ ]"}:
        return []

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        try:
            items = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []

    return items if isinstance(items, list) else []


def extract_max_power_kw(raw_connections):
    powers = []
    for item in parse_connections(raw_connections):
        if isinstance(item, dict) and item.get("PowerKW") is not None:
            try:
                powers.append(float(item["PowerKW"]))
            except (TypeError, ValueError):
                continue
    return max(powers) if powers else np.nan


def prepare_chargepoints(chargepoints, london_boundary_proj):
    print("Preparing chargepoint filters...")
    chargepoints_proj = chargepoints.to_crs(CRS).copy()
    chargepoints_proj = chargepoints_proj[chargepoints_proj.within(london_boundary_proj)].copy()
    print(f"  Within London boundary: {len(chargepoints_proj):,}")

    chargepoints_proj["usage_type_label"] = (
        chargepoints_proj["UsageTypeID"].map(USAGE_TYPE_LABELS).fillna("Unknown")
    )
    chargepoints_proj["status_type_label"] = (
        chargepoints_proj["StatusTypeID"].map(STATUS_TYPE_LABELS).fillna("Unknown")
    )
    chargepoints_proj["max_power_kw"] = chargepoints_proj["Connections"].apply(extract_max_power_kw)
    chargepoints_proj["connection_count"] = chargepoints_proj["Connections"].apply(
        lambda value: len(parse_connections(value))
    )
    chargepoints_proj["is_public_access"] = chargepoints_proj["UsageTypeID"].isin(PUBLIC_USAGE_TYPE_IDS)
    chargepoints_proj["is_operational_status"] = chargepoints_proj["StatusTypeID"].isin(
        OPERATIONAL_STATUS_TYPE_IDS
    )
    chargepoints_proj["is_rapid"] = chargepoints_proj["max_power_kw"].fillna(-1) >= RAPID_POWER_THRESHOLD_KW

    if FILTER_PUBLIC_CHARGERS_ONLY:
        chargepoints_proj = chargepoints_proj[chargepoints_proj["is_public_access"]].copy()
        print(f"  Public access only: {len(chargepoints_proj):,}")

    if FILTER_OPERATIONAL_CHARGERS_ONLY:
        chargepoints_proj = chargepoints_proj[chargepoints_proj["is_operational_status"]].copy()
        print(f"  Operational status only: {len(chargepoints_proj):,}")

    if FILTER_RAPID_CHARGERS_ONLY:
        chargepoints_proj = chargepoints_proj[chargepoints_proj["is_rapid"]].copy()
        print(
            f"  Rapid only (>= {RAPID_POWER_THRESHOLD_KW} kW by max connection power): {len(chargepoints_proj):,}"
        )

    chargepoints_proj = chargepoints_proj.drop_duplicates(subset=chargepoints_proj.geometry.name).copy()
    print(f"  Unique filtered chargepoint locations: {len(chargepoints_proj):,}")

    if chargepoints_proj.empty:
        raise ValueError("All chargepoints were filtered out. Relax the filters and try again.")

    print(f"Saving filtered chargepoints to {OUTPUT_FILTERED_CHARGEPOINTS}...")
    chargepoints_proj.to_crs(4326).to_file(OUTPUT_FILTERED_CHARGEPOINTS, driver="GeoJSON")
    return chargepoints_proj


def build_grid(boundary_polygon_proj):
    print(f"Building {GRID_SIZE_METERS} m analysis grid...")
    min_x, min_y, max_x, max_y = boundary_polygon_proj.bounds
    width = int(np.ceil((max_x - min_x) / GRID_SIZE_METERS))
    height = int(np.ceil((max_y - min_y) / GRID_SIZE_METERS))
    transform = from_origin(min_x, max_y, GRID_SIZE_METERS, GRID_SIZE_METERS)

    column_indices = np.arange(width)
    x_centers = min_x + (column_indices + 0.5) * GRID_SIZE_METERS
    row_chunks = []

    for start_row in range(0, height, GRID_ROW_BATCH_SIZE):
        end_row = min(start_row + GRID_ROW_BATCH_SIZE, height)
        batch_rows = np.arange(start_row, end_row)
        y_centers = max_y - (batch_rows + 0.5) * GRID_SIZE_METERS
        xx, yy = np.meshgrid(x_centers, y_centers)
        points = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=CRS)
        inside_mask = points.within(boundary_polygon_proj).to_numpy()

        if not inside_mask.any():
            continue

        rows = np.repeat(batch_rows, width)[inside_mask]
        cols = np.tile(column_indices, len(batch_rows))[inside_mask]
        row_chunks.append(
            pd.DataFrame(
                {
                    "row": rows,
                    "col": cols,
                    "x": xx.ravel()[inside_mask],
                    "y": yy.ravel()[inside_mask],
                }
            )
        )

    grid = pd.concat(row_chunks, ignore_index=True)
    grid = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid["x"], grid["y"]), crs=CRS)
    print(f"  Grid cells inside boundary: {len(grid):,}")
    print(f"  Raster dimensions: {width:,} columns x {height:,} rows")
    return grid, width, height, transform


def assign_travel_times(grid, graph, node_distances_seconds):
    print("Assigning nearest-node travel times to grid cells...")
    grid = grid.copy()
    x_values = grid["x"].to_numpy()
    y_values = grid["y"].to_numpy()
    nearest_nodes = np.empty(len(grid), dtype=np.int64)

    for start_index in range(0, len(grid), NEAREST_NODE_BATCH_SIZE):
        end_index = min(start_index + NEAREST_NODE_BATCH_SIZE, len(grid))
        nearest_nodes[start_index:end_index] = ox.nearest_nodes(
            graph,
            x_values[start_index:end_index],
            y_values[start_index:end_index],
        )

    grid["nearest_node"] = nearest_nodes
    grid["travel_time_seconds"] = pd.Series(nearest_nodes).map(node_distances_seconds).to_numpy()

    if grid["travel_time_seconds"].isna().any():
        missing_mask = grid["travel_time_seconds"].isna()
        missing = int(missing_mask.sum())
        print(f"Resolving {missing} grid cells that snapped to disconnected road nodes...")

        reachable_nodes = ox.graph_to_gdfs(graph, edges=False).loc[list(node_distances_seconds.keys())].copy()
        reachable_nodes["travel_time_seconds"] = pd.Series(node_distances_seconds)

        missing_points = gpd.GeoDataFrame(
            grid.loc[missing_mask, ["nearest_node"]].copy(),
            geometry=grid.loc[missing_mask, "geometry"],
            crs=CRS,
        )
        nearest_reachable = gpd.sjoin_nearest(
            missing_points,
            reachable_nodes[["travel_time_seconds", "geometry"]],
            how="left",
        )
        grid.loc[missing_mask, "travel_time_seconds"] = nearest_reachable["travel_time_seconds"].to_numpy()

    if grid["travel_time_seconds"].isna().any():
        missing = int(grid["travel_time_seconds"].isna().sum())
        raise ValueError(f"Missing travel times for {missing} grid cells after fallback assignment.")

    grid["travel_time_minutes"] = grid["travel_time_seconds"] / 60.0
    grid["time_limit"] = (
        np.ceil(grid["travel_time_minutes"] / INTERVAL_MINUTES)
        .clip(lower=1)
        .astype(int)
        * INTERVAL_MINUTES
    )
    grid["band_start"] = grid["time_limit"] - INTERVAL_MINUTES
    grid["label"] = grid.apply(
        lambda row: f"{int(row['band_start'])}-{int(row['time_limit'])} mins",
        axis=1,
    )
    return grid


def raster_to_isochrones(grid, width, height, transform, boundary_polygon_proj):
    print("Polygonizing travel-time bands...")
    label_grid = np.zeros((height, width), dtype=np.int16)
    label_grid[grid["row"].to_numpy(), grid["col"].to_numpy()] = grid["time_limit"].to_numpy(dtype=np.int16)

    band_stats = (
        grid.groupby(["time_limit", "band_start", "label"], as_index=False)
        .agg(
            min_minutes=("travel_time_minutes", "min"),
            mean_minutes=("travel_time_minutes", "mean"),
            max_minutes=("travel_time_minutes", "max"),
            cell_count=("travel_time_seconds", "count"),
        )
        .sort_values("time_limit")
    )

    geometries_by_band = {int(time_limit): [] for time_limit in band_stats["time_limit"].tolist()}
    for geometry_mapping, value in features.shapes(
        label_grid,
        mask=label_grid > 0,
        transform=transform,
        connectivity=4,
    ):
        time_limit = int(value)
        if time_limit in geometries_by_band:
            geometries_by_band[time_limit].append(shape(geometry_mapping))

    records = []
    for band in band_stats.itertuples(index=False):
        band_geometries = geometries_by_band[int(band.time_limit)]
        geometry = gpd.GeoSeries(band_geometries, crs=CRS).union_all().intersection(boundary_polygon_proj)
        if geometry.is_empty:
            continue
        records.append(
            {
                "time_limit": int(band.time_limit),
                "band_start": int(band.band_start),
                "label": band.label,
                "min_minutes": float(band.min_minutes),
                "mean_minutes": float(band.mean_minutes),
                "max_minutes": float(band.max_minutes),
                "cell_count": int(band.cell_count),
                "geometry": geometry,
            }
        )

    isochrones = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS).sort_values("time_limit").reset_index(drop=True)

    uncovered_area = boundary_polygon_proj.difference(isochrones.union_all())
    if not uncovered_area.is_empty:
        print("Adding residual boundary slivers to the outermost time band...")
        outermost_index = isochrones["time_limit"].idxmax()
        isochrones.loc[outermost_index, "geometry"] = isochrones.loc[outermost_index, "geometry"].union(
            uncovered_area
        )

    return isochrones


def plot_isochrones(london_boroughs_proj, chargepoints_proj, isochrones_proj):
    print("Generating map...")
    fig, ax = plt.subplots(figsize=(24, 24))
    london_boroughs_proj.plot(ax=ax, color="#f4f1e8", edgecolor="#b9b29f", linewidth=0.6)
    isochrones_proj.plot(
        ax=ax,
        column="time_limit",
        legend=True,
        alpha=0.88,
        cmap="YlOrRd",
        edgecolor="none",
        legend_kwds={
            "label": f"Driving time to nearest {selected_charger_label()} (minutes)",
            "orientation": "horizontal",
            "shrink": 0.55,
            "pad": 0.03,
        },
    )
    chargepoints_proj.plot(ax=ax, color="#0b3c5d", markersize=3, alpha=0.35)
    ax.set_title(
        f"London driving time to nearest {selected_charger_label()}",
        fontsize=24,
        color="#1a1a1a",
        pad=16,
    )
    ax.axis("off")
    fig.patch.set_facecolor("#f7f3ea")
    plt.tight_layout()
    plt.savefig(OUTPUT_MAP, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)


def print_diagnostics(isochrones_proj, grid, filtered_chargepoints):
    print("Travel-time diagnostics within London boundary:")
    print(f"  Filtered chargepoints used: {len(filtered_chargepoints):,}")
    print(f"  Grid size: {GRID_SIZE_METERS} meters")
    print(f"  Grid cells: {len(grid):,}")
    print(f"  Interval size: {INTERVAL_MINUTES} minutes")
    print(f"  Bands produced: {len(isochrones_proj)}")
    print(f"  Minimum travel time: {grid['travel_time_minutes'].min():.2f} minutes")
    print(f"  Mean travel time: {grid['travel_time_minutes'].mean():.2f} minutes")
    print(f"  Maximum travel time: {grid['travel_time_minutes'].max():.2f} minutes")
    print(f"  Time bands: {', '.join(str(value) for value in isochrones_proj['time_limit'].tolist())}")


def main():
    print("Loading chargepoints...")
    chargepoints = gpd.read_file(CHARGEPOINTS_PATH)

    print("Loading London boundary...")
    london_boroughs = gpd.read_file(BOUNDARY_PATH)
    london_boroughs_proj = london_boroughs.to_crs(CRS)
    london_boundary_proj = london_boroughs_proj.union_all()

    graph = load_drive_graph(london_boundary_proj)
    filtered_chargepoints = prepare_chargepoints(chargepoints, london_boundary_proj)

    print("Snapping filtered chargepoints to the drive network...")
    chargepoint_node_ids = ox.nearest_nodes(
        graph,
        filtered_chargepoints.geometry.x.to_numpy(),
        filtered_chargepoints.geometry.y.to_numpy(),
    )
    chargepoint_node_ids = list(set(chargepoint_node_ids))

    print("Calculating multi-source travel times...")
    node_distances_seconds = nx.multi_source_dijkstra_path_length(
        graph,
        chargepoint_node_ids,
        weight="travel_time",
    )

    grid, width, height, transform = build_grid(london_boundary_proj)
    grid = assign_travel_times(grid, graph, node_distances_seconds)
    isochrones_proj = raster_to_isochrones(grid, width, height, transform, london_boundary_proj)
    print_diagnostics(isochrones_proj, grid, filtered_chargepoints)

    print(f"Saving GeoJSON output to {OUTPUT_GEOJSON}...")
    isochrones_proj.to_crs(4326).to_file(OUTPUT_GEOJSON, driver="GeoJSON")

    plot_isochrones(london_boroughs_proj, filtered_chargepoints, isochrones_proj)
    print("Done!")


if __name__ == "__main__":
    main()