import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cdist

class DyTSolver:
    def __init__(self, routes: List[List[int]], distances: np.ndarray):
        """
        Initialize DyT solver
        
        Args:
            routes: Current routes (list of delivery point indices)
            distances: Distance matrix between all points
        """
        self.routes = routes
        self.distances = distances
        
    def calculate_dyt_distance(self, route1: List[int], route2: List[int]) -> float:
        """
        Calculate Dynamic Time Warping distance between two routes
        
        Args:
            route1: First route (list of delivery point indices)
            route2: Second route (list of delivery point indices)
            
        Returns:
            DTW distance between the routes
        """
        n, m = len(route1), len(route2)
        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0
        
        # Calculate DTW matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = self.distances[route1[i-1], route2[j-1]]
                dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                     dtw[i, j-1],      # deletion
                                     dtw[i-1, j-1])    # match
        
        return dtw[n, m]
    
    def find_best_insertion(self, 
                          new_point: int,
                          max_deviation: float = 0.2) -> Tuple[List[List[int]], float]:
        """
        Find the best insertion point for a new delivery point
        
        Args:
            new_point: Index of the new delivery point
            max_deviation: Maximum allowed deviation from original route
            
        Returns:
            Tuple of (updated_routes, total_deviation)
        """
        updated_routes = []
        total_deviation = 0
        
        for route in self.routes:
            if not route:
                continue
                
            best_insertion = None
            min_deviation = float('inf')
            
            # Try inserting at each position
            for i in range(len(route) + 1):
                new_route = route[:i] + [new_point] + route[i:]
                deviation = self.calculate_dyt_distance(route, new_route)
                
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_insertion = new_route
            
            if min_deviation <= max_deviation * self.calculate_route_length(route):
                updated_routes.append(best_insertion)
                total_deviation += min_deviation
            else:
                # If deviation is too high, keep original route
                updated_routes.append(route)
        
        return updated_routes, total_deviation
    
    def calculate_route_length(self, route: List[int]) -> float:
        """Calculate the total length of a route"""
        if not route:
            return 0
        length = self.distances[0, route[0]]  # From depot to first point
        for i in range(len(route)-1):
            length += self.distances[route[i], route[i+1]]
        length += self.distances[route[-1], 0]  # From last point to depot
        return length
    
    def optimize_routes(self, 
                       new_points: List[int],
                       max_deviation: float = 0.2) -> List[List[int]]:
        """
        Optimize routes with multiple new delivery points
        
        Args:
            new_points: List of new delivery point indices
            max_deviation: Maximum allowed deviation from original routes
            
        Returns:
            Updated routes
        """
        current_routes = self.routes.copy()
        
        for point in new_points:
            updated_routes, _ = self.find_best_insertion(point, max_deviation)
            current_routes = updated_routes
            
        return current_routes 