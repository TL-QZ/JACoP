import numpy as np
import torch
from typing import List, Tuple, Dict

from skimage import measure
import matplotlib.pyplot as plt
import cv2

import potrace
from matplotlib.path import Path
import matplotlib.patches as patches
from PIL import Image
import io
import base64

import shapely
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree # Requires Shapely >= 1.8, but vectorized functions need >= 2.0

# --- Important: Ensure Shapely version is >= 2.0 ---
try:
    shapely_major_version = int(shapely.__version__.split('.')[0])
    if shapely_major_version < 2:
        print(f"Warning: Shapely version {shapely.__version__} detected. "
              f"Version 2.0 or higher is required for optimal performance and "
              f"vectorized functions used in this class.")
except Exception:
    print("Warning: Could not determine Shapely version. "
          "Version 2.0 or higher is recommended.")
# ---
class DirectionalRangeMapComputer:
    def __init__(self, environment_polygons: List[Polygon]):
        """
        Initialize the range map computer with environment obstacles.
        Prepares obstacles and builds a spatial index (STRtree).

        :param environment_polygons: List of shapely Polygon objects representing obstacles.
                                     Non-Polygon or invalid geometries will be filtered out.
        """
        # Filter out invalid or empty geometries and ensure they are Polygons
        # valid_obstacles = [
        #     obs for obs in environment_polygons
        #     if isinstance(obs, Polygon) and obs is not None and not obs.is_empty
        # ]
        valid_obstacles = environment_polygons

        self.obstacles = valid_obstacles # Keep the list of valid polygons

        if not self.obstacles:
            # Handle case with no obstacles
            print("Warning: No valid obstacles provided. Range maps will show max_range.")
            self.tree = None
            # Store as object array for type consistency, even if empty
            self.obstacles_array = np.array([], dtype=object)
        else:
            # Store obstacles as a NumPy array for efficient indexing with query results
            self.obstacles_array = np.array(self.obstacles, dtype=object)
            # Build spatial index for efficient querying
            self.tree = STRtree(self.obstacles_array)
            print(f"Initialized with {len(self.obstacles_array)} obstacles and built STRtree.")

    def compute_directional_range_map_vectorized(self,
                                     agent_locations: np.ndarray, # Shape (T, 2)
                                     alpha_mins: np.ndarray,      # Shape (T,)
                                     alpha_maxs: np.ndarray,      # Shape (T,)
                                     angle_resolution: float = 1.0,
                                     max_range: float = 100.0) -> np.ndarray: # Output: Shape (T, N, 2)
        """
        Computes directional range maps for multiple agent poses using vectorized
        operations and spatial indexing, assuming a FIXED number of angle steps (N)
        per pose.

        Requires Shapely >= 2.0 for efficient vectorization.

        :param agent_locations: NumPy array of (x, y) coordinates, shape (T, 2).
        :param alpha_mins: NumPy array of minimum angles in degrees (0=East), shape (T,).
        :param alpha_maxs: NumPy array of maximum angles in degrees, shape (T,).
                           The range (alpha_max - alpha_min) MUST be consistent
                           across all T poses to ensure a fixed N.
        :param angle_resolution: Resolution of angles in degrees (must be positive).
        :param max_range: Maximum detection range (must be positive).
        :return: A single NumPy array with shape (T, N, 2).
                 N is the number of angles determined by the (constant) angular range
                 and resolution.
                 Axis 2 contains [angle (degrees, 0-360), distance].
        """
        # --- Input Validation ---
        if not isinstance(agent_locations, np.ndarray) or agent_locations.ndim != 2 or agent_locations.shape[1] != 2:
            raise ValueError("agent_locations must be a NumPy array with shape (T, 2)")
        T = agent_locations.shape[0]
        if not isinstance(alpha_mins, np.ndarray) or alpha_mins.shape != (T,):
            raise ValueError("alpha_mins must be a NumPy array with shape (T,)")
        if not isinstance(alpha_maxs, np.ndarray) or alpha_maxs.shape != (T,):
            raise ValueError("alpha_maxs must be a NumPy array with shape (T,)")
        if angle_resolution <= 0:
            raise ValueError("angle_resolution must be positive")
        if max_range <= 0:
            raise ValueError("max_range must be positive")

        # --- Calculate N and Validate Consistency ---
        # Adjust max angles to handle wrap-around (e.g., 350 to 10 degrees)
        alpha_maxs_adjusted = np.where(alpha_maxs < alpha_mins, alpha_maxs + 360.0, alpha_maxs)
        # Calculate angular range for each pose
        angular_ranges = alpha_maxs_adjusted - alpha_mins
        # Calculate N for the first pose (add epsilon for float precision)
        N_float = angular_ranges[0] / angle_resolution + 1e-9
        N = int(np.floor(N_float)) + 1

        # Verify that N is consistent for all poses
        if T > 1:
            expected_ranges = (N - 1) * angle_resolution
            # Use np.isclose for robust floating point comparison
            if not np.all(np.isclose(angular_ranges, expected_ranges, atol=1e-6)):
                 raise ValueError(f"Angular range (alpha_max - alpha_min) is not consistent "
                                  f"across all poses for the given angle_resolution. "
                                  f"Expected range approx {expected_ranges:.4f}, found ranges: {angular_ranges}")
        if N <= 0:
             # Handle case where the range/resolution results in zero steps
             print("Warning: Angular range and resolution result in zero angle steps (N=0).")
             return np.empty((T, 0, 2), dtype=float) # Return shape (T, 0, 2)


        # --- 1. Calculate all angles (Vectorized) ---
        # Shape: (T, N)
        # Create sequences from 0 to N-1
        k_values = np.arange(N, dtype=float) # Use float for multiplication safety
        # Use broadcasting: alpha_mins[:, None] -> (T, 1), k_values -> (N,) => result (T, N)
        angles_deg = alpha_mins[:, None] + k_values[None, :] * angle_resolution
        angles_rad = np.radians(angles_deg)
        angles_deg_wrapped = angles_deg % 360.0 # For final output

        # --- 2. Create Geometries (Points and Rays) ---
        # Flatten angles for ray generation: shape (T*N,)
        flat_angles_rad = angles_rad.ravel()
        total_rays = T * N

        # Repeat agent locations N times for each pose: shape (T*N, 2)
        origin_coords = np.repeat(agent_locations, N, axis=0)

        # Calculate end points for all rays: shape (T*N, 2)
        end_coords_x = origin_coords[:, 0] + max_range * np.cos(flat_angles_rad)
        end_coords_y = origin_coords[:, 1] + max_range * np.sin(flat_angles_rad)
        end_coords = np.stack([end_coords_x, end_coords_y], axis=-1)

        # Create vectorized Shapely objects (requires Shapely >= 2.0)
        origin_points = shapely.points(origin_coords) # Array of T*N Point objects

        # Create LineString coordinates array: shape (T*N, 2, 2)
        ray_coords = np.stack([origin_coords, end_coords], axis=1)
        rays = shapely.linestrings(ray_coords) # Array of T*N LineString objects

        # --- 3. Perform Ray Casting using STRtree and Vectorization ---
        # Initialize all distances to max_range: shape (T*N,)
        min_distances_flat = np.full(total_rays, max_range, dtype=float)

        if self.tree is not None and self.obstacles_array.size > 0:
            # Query the STRtree
            query_indices, obstacle_indices = self.tree.query(rays, predicate='intersects')

            if query_indices.size > 0:
                 # Get geometries involved in potential intersections
                 intersecting_rays = rays[query_indices]
                 intersecting_origins = origin_points[query_indices]
                 candidate_obstacles = self.obstacles_array[obstacle_indices]

                 # Perform vectorized intersection
                 intersections = shapely.intersection(intersecting_rays, candidate_obstacles)

                 # Filter out empty/invalid intersections
                 is_valid_intersection = ~shapely.is_empty(intersections) & (intersections != None)

                 if np.any(is_valid_intersection):
                    valid_indices = np.where(is_valid_intersection)[0]
                    valid_origins = intersecting_origins[valid_indices]
                    valid_geoms = intersections[valid_indices]
                    original_ray_indices = query_indices[valid_indices] # Indices into the flat arrays

                    # Calculate distances
                    distances = shapely.distance(valid_origins, valid_geoms)

                    # Update minimum distances atomically
                    np.minimum.at(min_distances_flat, original_ray_indices, distances)

        # --- 4. Format Output Array ---
        # Reshape distances from (T*N,) to (T, N)
        min_distances = min_distances_flat.reshape(T, N)

        # Stack angles and distances to get the final (T, N, 2) array
        # angles_deg_wrapped already has shape (T, N)
        output_array = np.stack([angles_deg_wrapped, min_distances], axis=-1)

        return output_array
    


    def visualize_directional_range_map(self,
                                    range_maps_array: np.ndarray, # Shape (T, N, 2)
                                    agent_locations: np.ndarray, # Shape (T, 2)
                                    pose_index: int = 0,         # Index 't' (0 to T-1) to visualize
                                    center_degree: float = 0.0): # Optional centering for polar plot
        """
        Visualize the directional range map for a specific agent pose.

        Displays the computed rays on a polar plot and overlays the rays
        onto the environment obstacles on a Cartesian plot for the selected pose.

        :param range_maps_array: The output from compute_directional_range_map_vectorized,
                                 shape (T, N, 2), where axis 2 is [angle, distance].
        :param agent_locations: The agent locations used to generate the range maps,
                                shape (T, 2).
        :param pose_index: The index (0 <= pose_index < T) of the pose to visualize.
        :param center_degree: Center degree for visualization orientation on the polar plot.
        :return: matplotlib.figure.Figure object containing the plots.
        """
        # --- Input Validation ---
        if not isinstance(range_maps_array, np.ndarray) or range_maps_array.ndim != 3 or range_maps_array.shape[2] != 2:
            raise ValueError("range_maps_array must be a NumPy array with shape (T, N, 2)")
        T = range_maps_array.shape[0]
        N = range_maps_array.shape[1]

        if not isinstance(agent_locations, np.ndarray) or agent_locations.ndim != 2 or agent_locations.shape[0] != T or agent_locations.shape[1] != 2:
            raise ValueError(f"agent_locations must be a NumPy array with shape ({T}, 2)")

        if not (0 <= pose_index < T):
            raise ValueError(f"pose_index ({pose_index}) must be between 0 and {T-1}")

        if N == 0:
            print(f"Warning: No angles (N=0) to visualize for pose {pose_index}.")
            # Optionally, create a simple plot indicating no data or return None
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            ax[0].set_title(f'Pose {pose_index}: No Range Data')
            ax[1].set_title(f'Pose {pose_index}: No Range Data')
            return fig # Return an empty figure or handle as needed

        # --- Data Extraction for Selected Pose ---
        selected_data = range_maps_array[pose_index, :, :] # Shape (N, 2)
        angles_deg = selected_data[:, 0]      # Shape (N,) - angles in degrees (0-360)
        distances = selected_data[:, 1]       # Shape (N,) - corresponding distances
        agent_location = agent_locations[pose_index] # Shape (2,) - (x, y) for this pose

        # --- Visualization ---
        fig = plt.figure(figsize=(15, 7))

        # 1. Polar plot for rays
        ax1 = fig.add_subplot(121, projection='polar')

        # Adjust angles for plotting based on center_degree and convert to radians
        angles_rad_plot = np.radians(angles_deg - center_degree)

        # Plot individual rays as lines from origin (optional, can be slow for large N)
        # for i in range(N):
        #     ax1.plot([0, angles_rad_plot[i]], [0, distances[i]], '-', color='blue', alpha=0.6)

        # Add scatter points at the end of each ray (more efficient for large N)
        ax1.scatter(angles_rad_plot, distances, c='red', s=10, zorder=3, label='Detected Points') # s adjusts size

        # Optionally plot lines connecting the points for a clearer lidar scan look
        # ax1.plot(angles_rad_plot, distances, color='red', linestyle='-', linewidth=0.8, label='Range Outline')
        # Sort by angle if plotting connecting lines to avoid zig-zags if angles aren't monotonic
        # sort_indices = np.argsort(angles_rad_plot)
        # ax1.plot(angles_rad_plot[sort_indices], distances[sort_indices], color='red', linestyle='-', linewidth=0.8, label='Range Outline')


        # Configure polar plot
        ax1.set_title(f'Directional Range Map (Pose {pose_index}) - Rays')
        ax1.set_theta_zero_location('E')  # 0 degrees is East
        ax1.set_theta_direction(-1)       # Clockwise angles
        ax1.grid(True)
        ax1.legend()

        # 2. Cartesian plot for environment and ray visualization
        ax2 = fig.add_subplot(122)

        # Plot obstacles (using the stored self.obstacles list)
        if hasattr(self, 'obstacles') and self.obstacles:
            for obstacle in self.obstacles:
                if isinstance(obstacle, Polygon):
                    try:
                        # Plot exterior
                        x_ext, y_ext = obstacle.exterior.xy
                        ax2.fill(x_ext, y_ext, color='black', alpha=0.8, label='_nolegend_') # Use fill for solid obstacles
                        # Plot interiors (holes) if they exist
                        for interior in obstacle.interiors:
                             x_int, y_int = interior.xy
                             ax2.fill(x_int, y_int, color='white', alpha=1.0, label='_nolegend_')
                    except Exception as e:
                        print(f"Warning: Could not plot obstacle part: {e}")
                elif isinstance(obstacle, LineString):
                    # Plot LineString as a line
                    x_line, y_line = obstacle.xy
                    ax2.plot(x_line, y_line, color='black', linewidth=1.5, label='_nolegend_')
                else:
                     print(f"Warning: Skipping non-Polygon geometry in obstacles list during plotting.")
            ax2.plot([], [], color='black', label='Obstacles') # Add single entry for legend

        # Plot agent location for the selected pose
        ax2.plot(agent_location[0], agent_location[1], 'bo', markersize=8, label=f'Agent Pose {pose_index}')

        # Plot rays in Cartesian coordinates
        angles_rad_cartesian = np.radians(angles_deg) # Use original angles (0-360)
        end_xs = agent_location[0] + distances * np.cos(angles_rad_cartesian)
        end_ys = agent_location[1] + distances * np.sin(angles_rad_cartesian)

        # Plot lines for rays (can be slow for large N, use low alpha)
        for i in range(N):
            ax2.plot([agent_location[0], end_xs[i]], [agent_location[1], end_ys[i]], 'r-', alpha=0.3, linewidth=0.5, label='_nolegend_')

        # Plot intersection points
        ax2.plot(end_xs, end_ys, 'r.', markersize=4, label='Detected Points')

        # Configure Cartesian plot
        ax2.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
        ax2.set_title(f'Environment with Rays (Pose {pose_index})')
        ax2.set_xlabel("X Coordinate")
        ax2.set_ylabel("Y Coordinate")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout() # Adjust layout to prevent overlap
        # plt.show() # Typically called outside the method
        return fig


class SceneMap:
    """
    A Geometric Map is a int tensor of shape [H, W]. The homography must transform a point in scene
    coordinates to the respective point in map coordinates.

    :param data: Numpy array of shape [H,W] with int values (0: nonnavigable, 1: navigable)
    :param w2m: Numpy array of shape [3, 3] (world to map)
    :param m2w: Numpy array of shape [3, 3] (map to world)
    """
    def __init__(self, data, w2m, m2w=None) -> None:
        self.data = torch.from_numpy(data).to(dtype=torch.int8)
        self.w2m_mat = torch.from_numpy(w2m).to(dtype=torch.float32)
        if m2w is None:
            self.m2w_mat = torch.linalg.inv(self.w2m_mat).to(dtype=torch.float32)
        else:
            self.m2w_mat = torch.from_numpy(m2w).to(dtype=torch.float32)


    def _transform(self, input_pts, homography_mat) -> torch.Tensor:
        """
        Transform points from scene coordinates to map coordinates.
        """
        org_shape = None
        if len(input_pts.shape) != 2:
            org_shape = input_pts.shape
            input_pts = input_pts.reshape((-1, 2))
        N, dims = input_pts.shape
        points_with_one = torch.ones((dims + 1, N), dtype=input_pts.dtype, device=input_pts.device)
        points_with_one[:dims] = input_pts.T
        output_pts = (homography_mat @ points_with_one).T
        output_pts = output_pts[:,:2] / output_pts[:,2, None]
        if org_shape is not None:
            output_pts = output_pts.reshape(org_shape)
        return output_pts

    def to_map_points(self, scene_pts) -> torch.Tensor:
        """
        Transform points from scene coordinates to map coordinates.
        Here the map is the one stored in self.data and the homography is self.homography.
        """
        map_points = self._transform(scene_pts, self.w2m_mat)
        return map_points

    def to_world_points(self, map_pts) -> torch.Tensor:
        """
        Transform points from map coordinates (not cutout!) to scene coordinates.
        """
        world_points = self._transform(map_pts, self.m2w_mat)
        return world_points

    def check_navigability_with_global(self, global_pts: torch.Tensor) -> torch.Tensor:
        """
        Check navigability of global points in the map.

        :param global_pts: Tensor of shape (..., 2) representing points in global coordinates.
        :return: Boolean tensor of shape (...) indicating navigability (False for nonnavigable or out of bounds).
        """
        map_pts = self.to_map_points(global_pts)
        map_pts_rounded = torch.round(map_pts).long()
        H, W = self.data.shape
        x_valid = (map_pts_rounded[..., 0] >= 0) & (map_pts_rounded[..., 0] < W)
        y_valid = (map_pts_rounded[..., 1] >= 0) & (map_pts_rounded[..., 1] < H)
        valid = x_valid & y_valid
        navigable = torch.zeros_like(valid, dtype=torch.bool)
        if valid.any():
            valid_map_pts = map_pts_rounded[valid]
            map_values = self.data[valid_map_pts[:, 1], valid_map_pts[:, 0]]
            navigable[valid] = (map_values == 1)
        return navigable
    
    def check_navigability_with_local(self, local_pts: torch.Tensor) -> torch.Tensor:
        """
        Check navigability of local points in the map.

        :param local_pts: Tensor of shape (..., 2) representing points in local map coordinates.
        :return: Boolean tensor of shape (...) indicating navigability (False for nonnavigable or out of bounds).
        """
        map_pts_rounded = torch.round(local_pts).long()
        H, W = self.data.shape
        x_valid = (map_pts_rounded[..., 0] >= 0) & (map_pts_rounded[..., 0] < W)
        y_valid = (map_pts_rounded[..., 1] >= 0) & (map_pts_rounded[..., 1] < H)
        valid = x_valid & y_valid
        navigable = torch.zeros_like(valid, dtype=torch.bool)
        if valid.any():
            valid_map_pts = map_pts_rounded[valid]
            map_values = self.data[valid_map_pts[:, 1], valid_map_pts[:, 0]]
            navigable[valid] = (map_values == 1)
        return navigable
        

# Raster to Linestrings Conversion
def is_straight_line(line1, line2, tolerance=10):
    """
    Checks if two LineStrings form a straight line (within a tolerance in pixel value).
    """
    coords1 = list(line1.coords)
    coords2 = list(line2.coords)
    if not coords1 or not coords2:
        return False
    p1 = coords1[0]
    p2 = coords1[-1]
    p3 = coords2[0]
    p4 = coords2[-1]
    if p2 != p3:
        return False
    if abs((p4[1] - p1[1]) * (p2[0] - p1[0]) - (p4[0] - p1[0]) * (p2[1] - p1[1])) < tolerance:
        return True
    else:
        return False

def merge_straight_linestrings(linestrings):
    """
    Merges adjacent LineStrings that form a straight line.
    """
    merged_linestrings = []
    i = 0
    while i < len(linestrings):
        current_line = linestrings[i]
        j = i + 1
        while j < len(linestrings) and is_straight_line(current_line, linestrings[j]):
            current_line = LineString(current_line.coords[:] + linestrings[j].coords[1:])
            j += 1
        merged_linestrings.append(current_line)
        i = j
    return merged_linestrings

def raster_to_lines(binary_map, simplify_tolerance=5):
    """
    Converts a binary raster map to a list of smoothed shapely LineStrings.

    Args:
        binary_map (numpy.ndarray): Binary raster map.
        simplify_tolerance (float): Tolerance for line simplification.

    Returns:
        list: List of smoothed shapely LineStrings.
    """

    contours = measure.find_contours(binary_map, 0.5)

    linestrings = []
    for contour in contours:
        for i in range(len(contour) - 1):
            line = [tuple(contour[i]), tuple(contour[i + 1])]
            linestring = LineString(line)
            linestrings.append(linestring)

    merged_linestrings = merge_straight_linestrings(linestrings)

    # Apply line simplification
    smoothed_linestrings = [line.simplify(simplify_tolerance) for line in merged_linestrings]

    return smoothed_linestrings

def set_edge_to_one(image):
    """
    Sets the pixels along the outer edge of a rectangular image to 1.
    """
    height, width = image.shape
    image[0, :] = 1
    image[height - 1, :] = 1
    image[:, 0] = 1
    image[:, width - 1] = 1
    return image

def local_to_global_linestrings(linestrings, homography_matrix):
    """
    Converts local linestrings to global coordinates using a homography matrix.
    
    :param linestrings: List of shapely LineString objects in local coordinates
    :param homography_matrix: 3x3 homography matrix for transformation
    :return: List of transformed LineString objects in global coordinates
    """
    transformed_lines = []
    for line in linestrings:
        coords_reflect = np.array(line.coords)
        coords = np.array([[coord[1], coord[0]] for coord in coords_reflect])
        coords_homogeneous = np.hstack((coords, np.ones((coords.shape[0], 1))))
        transformed_coords = (homography_matrix @ coords_homogeneous.T).T
        transformed_coords /= transformed_coords[:, 2][:, np.newaxis]
        transformed_lines.append(LineString(transformed_coords[:, :2]))
    
    return transformed_lines

def local_to_global_polygon(polygons, homography_matrix):
    """
    Converts local polygons to global coordinates using a homography matrix.
    
    :param polygons: List of shapely Polygon objects in local coordinates
    :param homography_matrix: 3x3 homography matrix for transformation
    :return: List of transformed Polygon objects in global coordinates
    """
    transformed_polygons = []
    for polygon in polygons:
        coords_reflect = np.array(polygon.exterior.coords)
        coords = np.array([[coord[1], coord[0]] for coord in coords_reflect])
        coords_homogeneous = np.hstack((coords, np.ones((coords.shape[0], 1))))
        transformed_coords = (homography_matrix @ coords_homogeneous.T).T
        transformed_coords /= transformed_coords[:, 2][:, np.newaxis]
        transformed_polygons.append(Polygon(transformed_coords[:, :2]))
    
    return transformed_polygons

# Raster to Polygon and vector image
def vectorize_binary_image(img, output_svg=None, threshold=127, simplify=True):
    """
    Vectorize a binary image using Potrace algorithm.
    
    Parameters:
    -----------
    image_path : str
        Path to the binary image file
    output_svg : str, optional
        Path to save the SVG output file. If None, no file is saved.
    threshold : int, optional
        Threshold value for binarization (0-255)
    simplify : bool, optional
        Whether to simplify the paths
        
    Returns:
    --------
    tuple: (paths, shapely_geometries)
        - paths: List of paths (vector contours)
        - shapely_geometries: List of Shapely geometries (Polygons/MultiPolygons)
    """
    
    # Ensure binary image (threshold if needed)
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Convert to format expected by potrace (binary image with 0s as foreground)
    binary = binary.astype(np.uint8)
    binary = ~binary  # Invert: 0 is foreground in potrace
    
    # Trace the bitmap
    bm = potrace.Bitmap(binary)
    paths = bm.trace(turdsize=2, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY, 
                    alphamax=1.0, opticurve=simplify, opttolerance=0.2)
    
    # Convert to Shapely geometries
    shapely_geometries = paths_to_shapely(paths)
    
    # Save as SVG if output_svg is specified
    if output_svg:
        with open(output_svg, 'w') as f:
            # SVG header
            width, height = img.shape[1], img.shape[0]
            f.write(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ')
            f.write(f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
            f.write('<g>\n')
            
            # Write paths
            for path in paths:
                f.write('<path d="')
                # Start point
                f.write(f'M{path.start_point.x},{path.start_point.y} ')
                
                # Curve segments
                for segment in path.segments:
                    if segment.is_corner:
                        f.write(f'L{segment.c.x},{segment.c.y} ')
                        f.write(f'L{segment.end_point.x},{segment.end_point.y} ')
                    else:
                        f.write(f'C{segment.c1.x},{segment.c1.y} ')
                        f.write(f'{segment.c2.x},{segment.c2.y} ')
                        f.write(f'{segment.end_point.x},{segment.end_point.y} ')
                
                f.write('Z" fill="black" />\n')
            
            f.write('</g>\n')
            f.write('</svg>\n')
    
    return paths, shapely_geometries

def paths_to_shapely(paths, linearize=True, density=2.0):
    """
    Convert potrace paths to Shapely geometries.
    
    Parameters:
    -----------
    paths : list
        List of potrace Path objects
    linearize : bool, optional
        Whether to linearize Bezier curves (True) or approximate them with straight line segments (False)
    density : float, optional
        Density of points when linearizing curves (higher values = more points)
        
    Returns:
    --------
    list
        List of Shapely Polygon or MultiPolygon objects
    """
    shapely_geometries = []
    
    for path in paths:
        coords = []
        
        # Add start point
        coords.append((path.start_point.x, path.start_point.y))
        
        # Process segments
        for segment in path.segments:
            if segment.is_corner:
                # For corner segments, add both corner point and end point
                coords.append((segment.c.x, segment.c.y))
                coords.append((segment.end_point.x, segment.end_point.y))
            else:
                if linearize:
                    # Linearize Bezier curve with multiple points
                    bezier_points = linearize_bezier(
                        (coords[-1][0], coords[-1][1]),
                        (segment.c1.x, segment.c1.y),
                        (segment.c2.x, segment.c2.y),
                        (segment.end_point.x, segment.end_point.y),
                        density
                    )
                    coords.extend(bezier_points[1:])  # Skip first point as it's already in coords
                else:
                    # Just use control points and end point
                    coords.append((segment.c1.x, segment.c1.y))
                    coords.append((segment.c2.x, segment.c2.y))
                    coords.append((segment.end_point.x, segment.end_point.y))
        
        # Create Shapely polygon (must close the ring by repeating the first point)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
            
        # Create a Polygon with the exterior ring
        try:
            poly = Polygon(coords)
            if poly.is_valid:
                shapely_geometries.append(poly)
            else:
                # Try to fix invalid polygon
                shapely_geometries.append(poly.buffer(0))
        except Exception as e:
            print(f"Warning: Could not create polygon: {e}")
            # Try creating a LineString instead
            try:
                shapely_geometries.append(LineString(coords))
            except:
                pass
    
    return shapely_geometries

def linearize_bezier(p0, p1, p2, p3, density=2.0):
    """
    Approximate a cubic Bezier curve with line segments.
    
    Parameters:
    -----------
    p0 : tuple
        Start point (x, y)
    p1, p2 : tuple
        Control points (x, y)
    p3 : tuple
        End point (x, y)
    density : float
        Controls number of points (higher = more points)
        
    Returns:
    --------
    list
        List of points approximating the curve
    """
    # Calculate the length of the curve (approximated by the polygon length)
    length = np.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
    length += np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    length += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    length += np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    
    # Number of points based on length and density
    n = max(3, int(length * density / 4))
    
    points = []
    for i in range(n):
        t = i / (n - 1)
        # Cubic Bezier formula
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    
    return points


