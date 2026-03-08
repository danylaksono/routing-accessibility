"""Generic isochrone analysis tool

This script builds driving-time isochrones from a set of input point locations
and a boundary polygon. It uses OSMnx to download or load a cached drive
network, snaps the points to the network, computes a multi-source Dijkstra
travel-time tree, and then rasterises the resulting travel times to produce a
GeoJSON of isochrone bands and a map image.

Configuration is handled via command‑line arguments.  The tool is deliberately
agnostic about the meaning of the input points; they need not be charge
points and may come from any GeoJSON source.  The boundary may be supplied as
a file or derived automatically from either a user‑supplied place name or the
bounds of the points themselves.

The grid resolution, time interval, buffer size, CRS, and other parameters are
all configurable.  A local network cache can be used to avoid repeated
downloads.
"""

import argparse
import json
import os
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import shape


DEFAULT_CRS = 27700
DEFAULT_DOWNLOAD_BUFFER_METERS = 5000
DEFAULT_GRID_SIZE_METERS = 25
DEFAULT_INTERVAL_MINUTES = 2
DEFAULT_GRID_ROW_BATCH_SIZE = 200
DEFAULT_NEAREST_NODE_BATCH_SIZE = 200_000


def load_drive_graph(boundary_polygon_proj, cache_path=None, crs=DEFAULT_CRS,
                     buffer_meters=DEFAULT_DOWNLOAD_BUFFER_METERS):
    """Return a projected drive graph covering the given boundary.

    The polygon must already be projected to ``crs``; it will be buffered in
    that coordinate system, transformed back to WGS84 for the OSMnx
    download, then reprojected.  If ``cache_path`` exists it is used instead of
    downloading.
    """
    buffered_wgs84 = (
        gpd.GeoSeries([boundary_polygon_proj.buffer(buffer_meters)], crs=crs)
        .to_crs(4326)
        .iloc[0]
    )

    if cache_path and os.path.exists(cache_path):
        print(f"Loading road network from cache ({cache_path})...")
        with open(cache_path, "rb") as fh:
            graph = pickle.load(fh)
    else:
        print("Downloading road network (buffered, this might take a while)...")
        graph = ox.graph_from_polygon(buffered_wgs84, network_type="drive")
        if cache_path:
            print(f"Saving road network to cache ({cache_path})...")
            with open(cache_path, "wb") as fh:
                pickle.dump(graph, fh)

    print("Projecting graph and adding travel times...")
    graph = ox.project_graph(graph, to_crs=crs)
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def build_grid(boundary_polygon_proj, grid_size_meters=DEFAULT_GRID_SIZE_METERS,
               row_batch_size=DEFAULT_GRID_ROW_BATCH_SIZE, crs=None):
    """Construct a point grid inside ``boundary_polygon_proj``.

    ``boundary_polygon_proj`` may be a shapely polygon/multipolygon or a
    GeoSeries/GeoDataFrame.  The caller should supply ``crs`` if the boundary
    is not already a GeoPandas object with a ``.crs`` attribute.

    Returns a GeoDataFrame of centres plus the raster width/height and
    affine transform needed to rasterise.
    """
    print(f"Building {grid_size_meters} m analysis grid...")
    if hasattr(boundary_polygon_proj, "crs") and boundary_polygon_proj.crs is not None:
        boundary_crs = boundary_polygon_proj.crs
        if isinstance(boundary_polygon_proj, (gpd.GeoDataFrame, gpd.GeoSeries)):
            boundary_geom = boundary_polygon_proj.unary_union
        else:
            boundary_geom = boundary_polygon_proj
    else:
        if crs is None:
            raise ValueError("CRS must be provided when boundary is not a GeoPandas object")
        boundary_crs = crs
        boundary_geom = boundary_polygon_proj

    min_x, min_y, max_x, max_y = boundary_geom.bounds
    width = int(np.ceil((max_x - min_x) / grid_size_meters))
    height = int(np.ceil((max_y - min_y) / grid_size_meters))
    transform = from_origin(min_x, max_y, grid_size_meters, grid_size_meters)

    column_indices = np.arange(width)
    x_centers = min_x + (column_indices + 0.5) * grid_size_meters
    row_chunks = []

    for start_row in range(0, height, row_batch_size):
        end_row = min(start_row + row_batch_size, height)
        batch_rows = np.arange(start_row, end_row)
        y_centers = max_y - (batch_rows + 0.5) * grid_size_meters
        xx, yy = np.meshgrid(x_centers, y_centers)
        points = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=boundary_crs)
        inside_mask = points.within(boundary_geom).to_numpy()

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
    grid = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid["x"], grid["y"]), crs=boundary_crs)
    print(f"  Grid cells inside boundary: {len(grid):,}")
    print(f"  Raster dimensions: {width:,} columns x {height:,} rows")
    return grid, width, height, transform


def assign_travel_times(grid, graph, node_distances_seconds,
                        nearest_node_batch_size=DEFAULT_NEAREST_NODE_BATCH_SIZE,
                        interval_minutes=DEFAULT_INTERVAL_MINUTES):
    """Snap grid cells to the network and tag them with travel-time bands."""
    print("Assigning nearest-node travel times to grid cells...")
    grid = grid.copy()
    x_values = grid["x"].to_numpy()
    y_values = grid["y"].to_numpy()
    nearest_nodes = np.empty(len(grid), dtype=np.int64)

    for start_index in range(0, len(grid), nearest_node_batch_size):
        end_index = min(start_index + nearest_node_batch_size, len(grid))
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
            crs=grid.crs,
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
        np.ceil(grid["travel_time_minutes"] / interval_minutes)
        .clip(lower=1)
        .astype(int)
        * interval_minutes
    )
    grid["band_start"] = grid["time_limit"] - interval_minutes
    grid["label"] = grid.apply(
        lambda row: f"{int(row['band_start'])}-{int(row['time_limit'])} mins",
        axis=1,
    )
    return grid


def raster_to_isochrones(grid, width, height, transform, boundary_polygon_proj, boundary_crs=None):
    """Convert labelled grid to a GeoDataFrame of isochrone polygons.

    ``boundary_polygon_proj`` may be a shapely geometry; in that case
    ``boundary_crs`` must supply the coordinate reference system.
    """
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

    recs = []
    for band in band_stats.itertuples(index=False):
        band_geometries = geometries_by_band[int(band.time_limit)]
        geo_series = gpd.GeoSeries(band_geometries, crs=boundary_crs)
        geometry = geo_series.union_all().intersection(boundary_polygon_proj)
        if geometry.is_empty:
            continue
        recs.append(
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

    crs_used = boundary_crs
    if crs_used is None and hasattr(boundary_polygon_proj, "crs"):
        crs_used = boundary_polygon_proj.crs

    isochrones = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs_used).sort_values("time_limit").reset_index(drop=True)

    uncovered_area = boundary_polygon_proj.difference(isochrones.union_all())
    if not uncovered_area.is_empty:
        print("Adding residual boundary slivers to the outermost time band...")
        outermost_index = isochrones["time_limit"].idxmax()
        isochrones.loc[outermost_index, "geometry"] = isochrones.loc[outermost_index, "geometry"].union(
            uncovered_area
        )

    return isochrones


def plot_isochrones(boundary_proj, points_proj, isochrones_proj, title, output_map):
    """Save a map of the boundary, isochrones and points."""
    fig, ax = plt.subplots(figsize=(24, 24))
    boundary_proj.plot(ax=ax, color="#f4f1e8", edgecolor="#b9b29f", linewidth=0.6)
    isochrones_proj.plot(
        ax=ax,
        column="time_limit",
        legend=True,
        alpha=0.88,
        cmap="YlOrRd",
        edgecolor="none",
        legend_kwds={
            "label": "Driving time (minutes)",
            "orientation": "horizontal",
            "shrink": 0.55,
            "pad": 0.03,
        },
    )
    points_proj.plot(ax=ax, color="#0b3c5d", markersize=3, alpha=0.35)
    ax.set_title(title, fontsize=24, color="#1a1a1a", pad=16)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f3ea")
    plt.tight_layout()
    plt.savefig(output_map, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)


def main():
    parser = argparse.ArgumentParser(description="Generic drive-time isochrone generator")
    parser.add_argument("--points", required=True, help="GeoJSON file containing source points")
    parser.add_argument("--boundary", help="Boundary polygon file (any format supported by GeoPandas)")
    parser.add_argument("--place", help="Place name to geocode if no boundary file is provided")
    parser.add_argument("--cache", help="Path to cache network graph pickle")
    parser.add_argument("--crs", type=int, default=DEFAULT_CRS, help="Projected CRS for analysis")
    parser.add_argument("--buffer", type=float, default=DEFAULT_DOWNLOAD_BUFFER_METERS,
                        help="Buffer in metres around boundary for network download")
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE_METERS,
                        help="Grid resolution in metres")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_MINUTES,
                        help="Time interval (minutes) for isochrone bands")
    parser.add_argument("--output-geojson", default="isochrones.geojson",
                        help="GeoJSON output path")
    parser.add_argument("--output-map", default="iso_map.png", help="Map image output path")

    args = parser.parse_args()

    points = gpd.read_file(args.points)
    if args.boundary:
        boundary = gpd.read_file(args.boundary)
    elif args.place:
        print(f"Geocoding place '{args.place}' for boundary...")
        boundary = ox.geocode_to_gdf(args.place)
    else:
        # derive boundary from point extent plus small buffer in data CRS later
        boundary = points.unary_union.envelope
        boundary = gpd.GeoSeries([boundary], crs=points.crs)

    boundary_proj = boundary.to_crs(args.crs)
    # ensure we end up with a single polygon (or multipolygon) geometry
    if isinstance(boundary_proj, gpd.GeoDataFrame):
        boundary_proj = boundary_proj.unary_union
    elif isinstance(boundary_proj, gpd.GeoSeries):
        boundary_proj = boundary_proj.unary_union
    # at this point ``boundary_proj`` is a shapely geometry, but load_drive_graph
    # expects something with a .buffer() method; shapely geometries are fine.
    graph = load_drive_graph(boundary_proj, cache_path=args.cache, crs=args.crs,
                             buffer_meters=args.buffer)

    print("Snapping source points to the drive network...")
    points_proj = points.to_crs(args.crs).copy()
    point_node_ids = ox.nearest_nodes(
        graph,
        points_proj.geometry.x.to_numpy(),
        points_proj.geometry.y.to_numpy(),
    )
    point_node_ids = list(set(point_node_ids))

    print("Calculating multi-source travel times...")
    node_distances_seconds = nx.multi_source_dijkstra_path_length(
        graph,
        point_node_ids,
        weight="travel_time",
    )

    grid, width, height, transform = build_grid(boundary_proj,
                                               grid_size_meters=args.grid_size,
                                               crs=args.crs)
    grid = assign_travel_times(grid, graph, node_distances_seconds,
                                interval_minutes=args.interval)
    isochrones_proj = raster_to_isochrones(grid, width, height, transform,
                                           boundary_proj, boundary_crs=args.crs)

    print(f"Saving GeoJSON output to {args.output_geojson}...")
    isochrones_proj.to_crs(4326).to_file(args.output_geojson, driver="GeoJSON")

    title = f"Driving time to nearest source point ({args.interval}-minute bands)"
    plot_isochrones(boundary.to_crs(args.crs), points_proj, isochrones_proj, title, args.output_map)

    print("Done!")


if __name__ == "__main__":
    main()
