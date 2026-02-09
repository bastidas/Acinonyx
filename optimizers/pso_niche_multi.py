"""
PSO with Niching (Species Formation) for multi-solution optimization.

Uses pyswarms library with a custom niching topology to find multiple
distinct local optima. Niching PSO maintains multiple sub-swarms (species)
that explore different regions of the search space.

Uses Mechanism-based fast path (4-7x faster than legacy compute_trajectory).

Reference:
- PySwarms: https://pyswarms.readthedocs.io/en/latest/
- Niching methods: Li, X. (2010). Niching Without Niching Parameters

License: pyswarms is MIT licensed (permissive for commercial use)
"""
from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np
import pyswarms.backend as P
from pyswarms.backend.handlers import BoundaryHandler
from pyswarms.backend.handlers import VelocityHandler
from pyswarms.backend.topology.base import Topology
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN

from pylink_tools.optimization_types import MultiSolutionResult
from pylink_tools.optimization_types import Solution

if TYPE_CHECKING:
    from pylink_tools.mechanism import Mechanism
    from pylink_tools.optimization_types import DimensionBoundsSpec, TargetTrajectory

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PSONichingConfig:
    """
    Configuration for PSO with Niching (species formation).

    Niching PSO maintains multiple sub-swarms (species) that explore different
    regions of the search space. When particles get too close, they form species
    that preserve diversity and prevent convergence to a single optimum.

    Attributes:
        n_particles: Total number of particles across all species
        n_iterations: Number of PSO iterations
        species_radius: Distance threshold for forming species (normalized 0-1)
        min_species_size: Minimum particles per species (smaller species merge)
        max_species: Maximum number of species to maintain
        w: Inertia weight (particle momentum)
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (species best attraction)
        epsilon_threshold: Error threshold for "near-optimal" solutions
        speciation_frequency: How often to recompute species (iterations)
        speciation_method: Method for species formation:
            - 'dbscan': Density-based clustering (default)
            - 'fitness_sharing': Reduce fitness based on nearby particles
            - 'crowding': Replace most similar particle
            - 'clearing': Keep only species winner, reset others
        init_mode: Swarm initialization mode:
            - 'random': Uniform random within bounds
            - 'sobol': Sobol quasi-random sequence
            - 'viable': Pre-filter for mechanism validity
            - 'fitness': Best samples by fitness score
        seed: Random seed for reproducibility
    """
    n_particles: int = 100
    n_iterations: int = 200
    species_radius: float = 0.2
    min_species_size: int = 3
    max_species: int = 10
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    epsilon_threshold: float = 1.0
    speciation_frequency: int = 10
    speciation_method: Literal['dbscan', 'fitness_sharing', 'crowding', 'clearing'] = 'dbscan'
    init_mode: Literal['random', 'sobol', 'viable', 'fitness'] = 'sobol'
    seed: int | None = None


# =============================================================================
# SPECIES FEATURE EXTRACTION
# =============================================================================


class SpeciesFeatureExtractor(ABC):
    """
    Abstract base class for extracting features used to define species.

    Species can be defined by various properties of the linkage solutions.
    This extensible system allows different feature sets to be used for
    determining which solutions belong to the same species.
    """

    @abstractmethod
    def extract_features(
        self,
        positions: np.ndarray,
        dimension_bounds_spec: DimensionBoundsSpec,
        pylink_data: dict,
    ) -> np.ndarray:
        """
        Extract features from particle positions for species comparison.

        Args:
            positions: Particle positions array (n_particles, n_dims)
            dimension_bounds_spec: Dimension specification with names and mappings
            pylink_data: Base linkage configuration

        Returns:
            Feature array (n_particles, n_features) for distance computation
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return names of extracted features for debugging/analysis."""
        pass


class DimensionFeatureExtractor(SpeciesFeatureExtractor):
    """
    Default feature extractor using normalized dimension values.

    Simply normalizes dimension values to [0, 1] range for distance computation.
    """

    def __init__(self):
        self._feature_names = []

    def extract_features(
        self,
        positions: np.ndarray,
        dimension_bounds_spec: DimensionBoundsSpec,
        pylink_data: dict,
    ) -> np.ndarray:
        """Normalize positions to [0, 1] for uniform distance computation."""
        lower = np.array([b[0] for b in dimension_bounds_spec.bounds])
        upper = np.array([b[1] for b in dimension_bounds_spec.bounds])
        range_vals = upper - lower
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero

        self._feature_names = [f'{name}_norm' for name in dimension_bounds_spec.names]

        return (positions - lower) / range_vals

    def get_feature_names(self) -> list[str]:
        return self._feature_names


class LinkagePropertyFeatureExtractor(SpeciesFeatureExtractor):
    """
    Feature extractor based on linkage structural properties.

    Extracts features that characterize the linkage topology and geometry:
    - Number of links in solution
    - Static node positions (x, y)
    - Link lengths by name
    - Ratios between link lengths

    This allows species to form based on structural similarity rather than
    just raw dimension values.
    """

    def __init__(
        self,
        include_link_count: bool = True,
        include_static_positions: bool = True,
        include_link_lengths: bool = True,
        include_length_ratios: bool = False,
        custom_features: list[Callable[[dict, DimensionBoundsSpec, np.ndarray], float]] | None = None,
    ):
        """
        Configure which linkage properties to include as features.

        Args:
            include_link_count: Include number of links as a feature
            include_static_positions: Include x, y of static nodes
            include_link_lengths: Include individual link lengths
            include_length_ratios: Include ratios between links (many features!)
            custom_features: List of callables (pylink_data, dim_spec, dims) -> float
        """
        self.include_link_count = include_link_count
        self.include_static_positions = include_static_positions
        self.include_link_lengths = include_link_lengths
        self.include_length_ratios = include_length_ratios
        self.custom_features = custom_features or []
        self._feature_names: list[str] = []

    def extract_features(
        self,
        positions: np.ndarray,
        dimension_bounds_spec: DimensionBoundsSpec,
        pylink_data: dict,
    ) -> np.ndarray:
        """
        Extract linkage property features from each particle position.

        Returns normalized features for distance computation.
        """
        n_particles = len(positions)
        features_list = []
        self._feature_names = []

        linkage = pylink_data.get('linkage', {})
        nodes = linkage.get('nodes', {})
        edges = linkage.get('edges', {})

        # Get static nodes (ground/fixed nodes)
        static_nodes = {
            name: node for name, node in nodes.items()
            if node.get('type') in ('ground', 'static', 'fixed')
        }

        # Feature: Number of links (constant across particles, but included for completeness)
        if self.include_link_count:
            n_links = len(edges)
            link_count_feature = np.full((n_particles, 1), n_links, dtype=np.float64)
            features_list.append(link_count_feature)
            self._feature_names.append('n_links')

        # Feature: Static node positions
        if self.include_static_positions and static_nodes:
            static_features = []
            for node_name, node in sorted(static_nodes.items()):
                x = node.get('x', 0.0)
                y = node.get('y', 0.0)
                static_features.extend([x, y])
                self._feature_names.extend([f'{node_name}_x', f'{node_name}_y'])

            if static_features:
                # Replicate for all particles (static nodes don't change with dimensions)
                static_array = np.tile(static_features, (n_particles, 1))
                features_list.append(static_array)

        # Feature: Link lengths (from dimension values)
        if self.include_link_lengths:
            # Map dimension names to their indices
            dim_indices = {name: i for i, name in enumerate(dimension_bounds_spec.names)}

            # Extract link lengths for each particle
            for dim_name in dimension_bounds_spec.names:
                if 'distance' in dim_name.lower() or 'length' in dim_name.lower():
                    dim_idx = dim_indices[dim_name]
                    link_lengths = positions[:, dim_idx:dim_idx+1]
                    features_list.append(link_lengths)
                    self._feature_names.append(f'length_{dim_name}')

        # Feature: Length ratios (optional - creates many features)
        if self.include_length_ratios:
            length_dims = [
                (name, i) for i, name in enumerate(dimension_bounds_spec.names)
                if 'distance' in name.lower() or 'length' in name.lower()
            ]

            for i, (name1, idx1) in enumerate(length_dims):
                for name2, idx2 in length_dims[i+1:]:
                    # Ratio feature: length1 / length2
                    l1 = positions[:, idx1]
                    l2 = positions[:, idx2]
                    # Avoid division by zero
                    ratio = np.where(l2 > 1e-6, l1 / l2, 1.0)
                    features_list.append(ratio.reshape(-1, 1))
                    self._feature_names.append(f'ratio_{name1}/{name2}')

        # Custom features
        for i, custom_fn in enumerate(self.custom_features):
            custom_values = []
            for pos in positions:
                try:
                    value = custom_fn(pylink_data, dimension_bounds_spec, pos)
                    custom_values.append(value)
                except Exception:
                    custom_values.append(0.0)
            features_list.append(np.array(custom_values).reshape(-1, 1))
            self._feature_names.append(f'custom_{i}')

        # Combine all features
        if not features_list:
            # Fallback: use normalized positions
            return DimensionFeatureExtractor().extract_features(
                positions, dimension_bounds_spec, pylink_data,
            )

        all_features = np.hstack(features_list)

        # Normalize features to [0, 1] for uniform distance computation
        feat_min = all_features.min(axis=0)
        feat_max = all_features.max(axis=0)
        feat_range = feat_max - feat_min
        feat_range[feat_range == 0] = 1.0  # Avoid division by zero

        normalized = (all_features - feat_min) / feat_range

        return normalized

    def get_feature_names(self) -> list[str]:
        return self._feature_names


# =============================================================================
# SPECIATION METHODS
# =============================================================================


class SpeciationMethod(ABC):
    """Abstract base class for speciation methods."""

    @abstractmethod
    def form_species(
        self,
        features: np.ndarray,
        costs: np.ndarray,
        config: PSONichingConfig,
    ) -> np.ndarray:
        """
        Assign particles to species based on features and costs.

        Args:
            features: Feature array (n_particles, n_features)
            costs: Fitness costs for each particle
            config: Niching configuration

        Returns:
            Species labels array (n_particles,) with species ID per particle
        """
        pass


class DBSCANSpeciation(SpeciationMethod):
    """
    DBSCAN-based speciation using density clustering.

    Groups particles that are close together in feature space into species.
    Good for discovering arbitrary-shaped clusters.
    """

    def form_species(
        self,
        features: np.ndarray,
        costs: np.ndarray,
        config: PSONichingConfig,
    ) -> np.ndarray:
        n_particles = len(features)

        if n_particles <= 1:
            return np.zeros(n_particles, dtype=int)

        # Use DBSCAN for species formation
        clustering = DBSCAN(
            eps=config.species_radius,
            min_samples=config.min_species_size,
            metric='euclidean',
        )
        labels = clustering.fit_predict(features)

        # Handle noise points (-1 label): assign to nearest species
        noise_mask = labels == -1
        if np.any(noise_mask) and np.any(~noise_mask):
            for i in np.where(noise_mask)[0]:
                non_noise_features = features[~noise_mask]
                non_noise_labels = labels[~noise_mask]
                distances = np.linalg.norm(non_noise_features - features[i], axis=1)
                nearest_idx = np.argmin(distances)
                labels[i] = non_noise_labels[nearest_idx]
        elif np.all(noise_mask):
            # All particles are noise - treat as single species
            labels = np.zeros(n_particles, dtype=int)

        # Limit number of species
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) > config.max_species:
            # Keep top species by size
            species_sizes = [(s, np.sum(labels == s)) for s in unique_labels]
            species_sizes.sort(key=lambda x: -x[1])
            top_species = [s for s, _ in species_sizes[:config.max_species]]

            for i, label in enumerate(labels):
                if label not in top_species:
                    distances = []
                    for ts in top_species:
                        ts_features = features[labels == ts]
                        if len(ts_features) > 0:
                            dist = np.min(np.linalg.norm(ts_features - features[i], axis=1))
                            distances.append((ts, dist))
                    if distances:
                        labels[i] = min(distances, key=lambda x: x[1])[0]

        # Renumber labels to be consecutive
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])


class FitnessSharingSpeciation(SpeciationMethod):
    """
    Fitness sharing speciation.

    Reduces effective fitness of particles near others, encouraging diversity.
    Species form around peaks in the fitness landscape.

    """

    def form_species(
        self,
        features: np.ndarray,
        costs: np.ndarray,
        config: PSONichingConfig,
    ) -> np.ndarray:
        n_particles = len(features)

        if n_particles <= 1:
            return np.zeros(n_particles, dtype=int)

        # Compute pairwise distances
        distances = squareform(pdist(features, metric='euclidean'))

        # Compute sharing function
        sigma = config.species_radius
        alpha = 1.0  # Sharing function shape parameter

        sharing = np.zeros((n_particles, n_particles))
        for i in range(n_particles):
            for j in range(n_particles):
                if distances[i, j] < sigma:
                    sharing[i, j] = 1 - (distances[i, j] / sigma) ** alpha

        # Compute shared fitness (lower is better for costs)
        niche_count = sharing.sum(axis=1)
        niche_count[niche_count == 0] = 1.0

        # Assign species based on nearest best particle
        # Convert costs to fitness (higher is better)
        fitness = 1.0 / (costs + 1e-10)
        shared_fitness = fitness / niche_count

        # Find local peaks (particles with highest shared fitness in neighborhood)
        species_seeds = []
        remaining = set(range(n_particles))

        while remaining and len(species_seeds) < config.max_species:
            # Find particle with highest shared fitness among remaining
            remaining_list = list(remaining)
            best_idx = remaining_list[np.argmax(shared_fitness[remaining_list])]
            species_seeds.append(best_idx)

            # Remove particles within sigma of this seed
            for i in list(remaining):
                if distances[best_idx, i] < sigma:
                    remaining.discard(i)

        # Assign all particles to nearest seed
        labels = np.zeros(n_particles, dtype=int)
        for i in range(n_particles):
            if species_seeds:
                seed_distances = [distances[i, seed] for seed in species_seeds]
                labels[i] = np.argmin(seed_distances)

        return labels


class CrowdingSpeciation(SpeciationMethod):
    """
    Crowding-based speciation.

    New solutions replace the most similar existing solution.
    Species naturally form around optima.

    Reference: De Jong (1975)
    """

    def form_species(
        self,
        features: np.ndarray,
        costs: np.ndarray,
        config: PSONichingConfig,
    ) -> np.ndarray:
        # Crowding is more of an update rule than explicit species formation
        # For compatibility, we use k-means style clustering based on features
        n_particles = len(features)

        if n_particles <= 1:
            return np.zeros(n_particles, dtype=int)

        # Use k-means style assignment
        n_species = min(config.max_species, n_particles // config.min_species_size)
        n_species = max(1, n_species)

        # Initialize species seeds as particles with best fitness
        sorted_indices = np.argsort(costs)
        seeds_indices = sorted_indices[:n_species]
        seeds = features[seeds_indices]

        # Assign each particle to nearest seed
        labels = np.zeros(n_particles, dtype=int)
        for i in range(n_particles):
            distances = np.linalg.norm(seeds - features[i], axis=1)
            labels[i] = np.argmin(distances)

        return labels


class ClearingSpeciation(SpeciationMethod):
    """
    Clearing-based speciation.

    Within each niche, only the best particle (winner) keeps its fitness.
    Others are "cleared" to have zero fitness, preventing convergence.

    Reference: Petrowski (1996)
    """

    def form_species(
        self,
        features: np.ndarray,
        costs: np.ndarray,
        config: PSONichingConfig,
    ) -> np.ndarray:
        n_particles = len(features)

        if n_particles <= 1:
            return np.zeros(n_particles, dtype=int)

        # Compute pairwise distances
        distances = squareform(pdist(features, metric='euclidean'))
        sigma = config.species_radius

        # Find species by iteratively selecting winners
        assigned = np.full(n_particles, -1, dtype=int)
        species_id = 0

        # Sort by fitness (best first)
        sorted_indices = np.argsort(costs)

        for idx in sorted_indices:
            if assigned[idx] >= 0:
                continue

            # This particle becomes a species winner
            assigned[idx] = species_id

            # Clear (assign to same species) all particles within sigma
            for j in range(n_particles):
                if assigned[j] < 0 and distances[idx, j] < sigma:
                    assigned[j] = species_id

            species_id += 1
            if species_id >= config.max_species:
                break

        # Assign remaining unassigned to nearest species
        for i in range(n_particles):
            if assigned[i] < 0:
                # Find nearest assigned particle
                assigned_mask = assigned >= 0
                if np.any(assigned_mask):
                    assigned_distances = distances[i, assigned_mask]
                    nearest_assigned = np.where(assigned_mask)[0][np.argmin(assigned_distances)]
                    assigned[i] = assigned[nearest_assigned]
                else:
                    assigned[i] = 0

        return assigned


def get_speciation_method(method_name: str) -> SpeciationMethod:
    """Factory function to get speciation method by name."""
    methods = {
        'dbscan': DBSCANSpeciation(),
        'fitness_sharing': FitnessSharingSpeciation(),
        'crowding': CrowdingSpeciation(),
        'clearing': ClearingSpeciation(),
    }
    if method_name not in methods:
        raise ValueError(f'Unknown speciation method: {method_name}. Choose from: {list(methods.keys())}')
    return methods[method_name]


# =============================================================================
# NICHING TOPOLOGY
# =============================================================================


class NichingTopology(Topology):
    """
    Custom topology for niching/speciation in PSO.

    Instead of a single global best, this topology maintains multiple
    species-level bests. Particles are attracted to their species best
    rather than the overall global best, preserving diversity.
    """

    def __init__(
        self,
        config: PSONichingConfig,
        dimension_bounds_spec: DimensionBoundsSpec,
        pylink_data: dict,
        feature_extractor: SpeciesFeatureExtractor | None = None,
        static: bool = False,
    ):
        """
        Initialize the niching topology.

        Args:
            config: PSO niching configuration
            dimension_bounds_spec: Dimension specification for feature extraction
            pylink_data: Base linkage data for feature extraction
            feature_extractor: Custom feature extractor (default: LinkagePropertyFeatureExtractor)
            static: If True, species assignments don't change after initial formation
        """
        super().__init__(static=static)
        self.config = config
        self.dimension_bounds_spec = dimension_bounds_spec
        self.pylink_data = pylink_data

        # Feature extractor for species comparison
        self.feature_extractor = feature_extractor or LinkagePropertyFeatureExtractor(
            include_link_count=True,
            include_static_positions=True,
            include_link_lengths=True,
            include_length_ratios=False,
        )

        # Speciation method
        self.speciation_method = get_speciation_method(config.speciation_method)

        # Species tracking
        self.species_labels: np.ndarray | None = None
        self.species_bests: dict[int, tuple[np.ndarray, float]] = {}
        self.n_species = 0

    def compute_gbest(self, swarm, **kwargs):
        """
        Update species-level bests and return global best.

        This method:
        1. Extracts features from particle positions
        2. Forms/updates species based on features
        3. Computes best particle within each species
        4. Returns the overall best across all species
        """
        # Extract features for species comparison
        features = self.feature_extractor.extract_features(
            swarm.position,
            self.dimension_bounds_spec,
            self.pylink_data,
        )

        # Form species using configured method
        self.species_labels = self.speciation_method.form_species(
            features,
            swarm.pbest_cost,
            self.config,
        )
        self.n_species = len(np.unique(self.species_labels))

        # Update species-level bests
        for species_id in range(self.n_species):
            mask = self.species_labels == species_id
            if not np.any(mask):
                continue

            species_costs = swarm.pbest_cost[mask]
            species_positions = swarm.pbest_pos[mask]

            best_idx = np.argmin(species_costs)
            best_cost = species_costs[best_idx]
            best_pos = species_positions[best_idx]

            if species_id not in self.species_bests or best_cost < self.species_bests[species_id][1]:
                self.species_bests[species_id] = (best_pos.copy(), best_cost)

        # Return overall global best
        if self.species_bests:
            global_best_species = min(self.species_bests.items(), key=lambda x: x[1][1])
            return global_best_species[1][0], global_best_species[1][1]
        else:
            # Fallback
            best_idx = np.argmin(swarm.pbest_cost)
            return swarm.pbest_pos[best_idx], swarm.pbest_cost[best_idx]

    def compute_velocity(self, swarm, clamp=None, vh=None, bounds=None):
        """
        Compute velocity using species-level social component.

        Each particle is attracted to its species best, not the global best.
        """
        if vh is None:
            vh = VelocityHandler(strategy='unmodified')

        position = swarm.position
        velocity = swarm.velocity
        pbest_pos = swarm.pbest_pos
        options = swarm.options

        c1 = options.get('c1', 1.5)
        c2 = options.get('c2', 1.5)
        w = options.get('w', 0.7)

        r1 = np.random.random(position.shape)
        r2 = np.random.random(position.shape)

        # Cognitive component (toward personal best)
        cognitive = c1 * r1 * (pbest_pos - position)

        # Social component (toward species best)
        social = np.zeros_like(position)
        if self.species_labels is not None and self.species_bests:
            for i in range(len(position)):
                species_id = self.species_labels[i]
                if species_id in self.species_bests:
                    species_best_pos = self.species_bests[species_id][0]
                    social[i] = c2 * r2[i] * (species_best_pos - position[i])
        else:
            if swarm.best_pos is not None:
                social = c2 * r2 * (swarm.best_pos - position)

        new_velocity = w * velocity + cognitive + social
        new_velocity = vh(new_velocity, clamp, position=position, bounds=bounds)

        return new_velocity

    def compute_position(self, swarm, bounds=None, bh=None):
        """Update particle positions with boundary handling."""
        if bh is None:
            bh = BoundaryHandler(strategy='periodic')

        new_position = swarm.position + swarm.velocity

        if bounds is not None:
            new_position = bh(new_position, bounds)

        return new_position

    def get_species_info(self) -> dict:
        """Return current species information for analysis."""
        if self.species_labels is None:
            return {
                'n_species': 0,
                'species_sizes': [],
                'species_bests': {},
                'feature_names': self.feature_extractor.get_feature_names(),
            }

        unique_species = np.unique(self.species_labels)
        species_sizes = [int(np.sum(self.species_labels == s)) for s in unique_species]

        return {
            'n_species': self.n_species,
            'species_sizes': species_sizes,
            'species_bests': {
                int(k): {'position': v[0].tolist(), 'cost': float(v[1])}
                for k, v in self.species_bests.items()
            },
            'feature_names': self.feature_extractor.get_feature_names(),
        }


# =============================================================================
# SWARM INITIALIZATION
# =============================================================================


def initialize_swarm_positions(
    n_particles: int,
    dimension_bounds_spec: DimensionBoundsSpec,
    pylink_data: dict,
    target: TargetTrajectory | None = None,
    mode: str = 'sobol',
    seed: int | None = None,
) -> np.ndarray:
    """
    Initialize particle positions for the swarm using generate_valid_samples.

    Args:
        n_particles: Number of particles to initialize
        dimension_bounds_spec: Dimension specification with bounds
        pylink_data: Linkage data for viable/fitness sampling validation
        target: Target trajectory (required for 'fitness' mode)
        mode: Initialization mode:
            - 'random': Uniform random within bounds
            - 'sobol': Sobol quasi-random sequence with viability check
            - 'viable': Pre-filter for mechanism validity (sobol + viability)
            - 'fitness': Best samples by fitness score (requires target)
        seed: Random seed for reproducibility

    Returns:
        Initial positions array of shape (n_particles, n_dims)
    """
    from target_gen.sampling import generate_valid_samples

    n_dims = len(dimension_bounds_spec)
    lower = np.array([b[0] for b in dimension_bounds_spec.bounds])
    upper = np.array([b[1] for b in dimension_bounds_spec.bounds])

    if mode == 'random':
        # Simple uniform random initialization
        if seed is not None:
            np.random.seed(seed)
        positions = np.random.uniform(lower, upper, (n_particles, n_dims))
        # Ensure first particle starts at initial values
        positions[0] = np.array(dimension_bounds_spec.initial_values)
        return positions

    elif mode == 'sobol':
        # Sobol sequence with viability validation
        try:
            from demo.helpers import create_mechanism_from_dict
            mechanism = create_mechanism_from_dict(pylink_data)
            result = generate_valid_samples(
                mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,
                n_valid_requested=n_particles,
                max_attempts=n_particles,  # No extra attempts
                sampling_mode='sobol',
                seed=seed,
            )
            samples = result.samples
            if len(samples) < n_particles:
                # Fill remaining with random
                n_fill = n_particles - len(samples)
                if seed is not None:
                    np.random.seed(seed + 1000)
                fill = np.random.uniform(lower, upper, (n_fill, n_dims))
                samples = np.vstack([samples, fill])
            return samples[:n_particles]

        except Exception as e:
            logger.warning(f'Sobol initialization failed: {e}. Falling back to random.')
            if seed is not None:
                np.random.seed(seed)
            positions = np.random.uniform(lower, upper, (n_particles, n_dims))
            positions[0] = np.array(dimension_bounds_spec.initial_values)
            return positions

    elif mode == 'viable':
        # Sobol + viability filtering
        try:
            from demo.helpers import create_mechanism_from_dict
            mechanism = create_mechanism_from_dict(pylink_data)
            result = generate_valid_samples(
                mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,
                n_valid_requested=n_particles,
                max_attempts=n_particles * 10,  # More attempts for stricter validation
                sampling_mode='sobol',
                seed=seed,
            )
            samples = result.samples
            logger.info(f'Viable initialization: {len(samples)}/{n_particles} from {result.n_generated} attempts')

            if len(samples) < n_particles:
                n_fill = n_particles - len(samples)
                if seed is not None:
                    np.random.seed(seed + 1000)
                fill = np.random.uniform(lower, upper, (n_fill, n_dims))
                samples = np.vstack([samples, fill])

            return samples[:n_particles]

        except Exception as e:
            logger.warning(f'Viable initialization failed: {e}. Falling back to random.')
            if seed is not None:
                np.random.seed(seed)
            positions = np.random.uniform(lower, upper, (n_particles, n_dims))
            positions[0] = np.array(dimension_bounds_spec.initial_values)
            return positions

    elif mode == 'fitness':
        # Best samples by fitness score (use generate_good_samples with low epsilon)
        if target is None:
            logger.warning("'fitness' mode requires target. Falling back to 'sobol'.")
            return initialize_swarm_positions(
                n_particles, dimension_bounds_spec, pylink_data, target, 'sobol', seed,
            )

        try:
            from demo.helpers import create_mechanism_from_dict
            from target_gen.sampling import generate_good_samples
            mechanism = create_mechanism_from_dict(pylink_data)
            # Use generate_good_samples with a high epsilon to get best samples
            # We'll sort by score and take the best n_particles
            result = generate_good_samples(
                mechanism=mechanism,
                dimension_bounds_spec=dimension_bounds_spec,
                target=target,
                n_good_requested=n_particles * 2,  # Get more to select best
                epsilon=float('inf'),  # Accept all samples, we'll sort by score
                max_attempts=n_particles * 5,
                sampling_mode='sobol',
                seed=seed,
                phase_invariant=True,
                return_all=True,  # Get all samples to sort
            )
            # Sort by score and take best n_particles
            if result.scores is not None and len(result.scores) > 0:
                valid_mask = np.isfinite(result.scores)
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    sorted_indices = valid_indices[np.argsort(result.scores[valid_indices])[:n_particles]]
                    samples = result.samples[sorted_indices]
                    scores = result.scores[sorted_indices]
                    logger.info(f'Fitness initialization: best score = {scores[0]:.4f}')
                else:
                    samples = result.samples[:n_particles]
                    scores = None
            else:
                samples = result.samples[:n_particles]
                scores = None

            if len(samples) < n_particles:
                n_fill = n_particles - len(samples)
                if seed is not None:
                    np.random.seed(seed + 1000)
                fill = np.random.uniform(lower, upper, (n_fill, n_dims))
                samples = np.vstack([samples, fill])

            return samples[:n_particles]

        except Exception as e:
            logger.warning(f'Fitness initialization failed: {e}. Falling back to random.')
            if seed is not None:
                np.random.seed(seed)
            positions = np.random.uniform(lower, upper, (n_particles, n_dims))
            positions[0] = np.array(dimension_bounds_spec.initial_values)
            return positions

    else:
        raise ValueError(f'Unknown init mode: {mode}')


# =============================================================================
# MAIN OPTIMIZER
# =============================================================================


def run_pso_niching_multi(
    mechanism: Mechanism,
    target: TargetTrajectory,
    dimension_bounds_spec: DimensionBoundsSpec | None = None,
    config: PSONichingConfig | None = None,
    metric: str = 'mse',
    verbose: bool = True,
    phase_invariant: bool = True,
    phase_align_method: Literal['rotation', 'fft', 'frechet'] = 'rotation',
    feature_extractor: SpeciesFeatureExtractor | None = None,
    **kwargs,
) -> MultiSolutionResult:
    """
    Find multiple distinct solutions using PSO with Niching (species formation).

    Uses Mechanism-based fast path (4-7x faster than legacy).

    Args:
        mechanism: Mechanism object to optimize (will be modified in place)
        target: Target trajectory to match (joint name + positions)
        dimension_bounds_spec: Dimensions to optimize (extracted from mechanism if not provided)
        config: PSO niching configuration (uses defaults if not provided)
        metric: Error metric ('mse', 'rmse', 'total', 'max')
        verbose: Print progress information
        phase_invariant: Use phase-aligned scoring
        phase_align_method: Phase alignment algorithm
        feature_extractor: Custom feature extractor for species comparison
        **kwargs: Additional arguments (ignored, for interface compatibility)

    Returns:
        MultiSolutionResult with one solution per stable species
    """
    from pylink_tools.mechanism import Mechanism as MechanismType, create_mechanism_fitness

    # Validate mechanism type
    if not isinstance(mechanism, MechanismType):
        return MultiSolutionResult(
            solutions=[],
            best_solution=None,
            n_unique_clusters=0,
            epsilon_threshold=0.0,
            search_space_coverage=0.0,
            total_evaluations=0,
            success=False,
            method='pso_niching',
            method_config={},
            error_message=f'Expected Mechanism object, got {type(mechanism).__name__}. '
                          f'Convert pylink_data to Mechanism using create_mechanism_from_dict()',
        )

    if config is None:
        config = PSONichingConfig()

    if dimension_bounds_spec is None:
        dimension_bounds_spec = mechanism.get_dimension_bounds_spec()

    n_dims = len(dimension_bounds_spec)
    if n_dims == 0:
        return MultiSolutionResult(
            solutions=[],
            best_solution=None,
            n_unique_clusters=0,
            epsilon_threshold=config.epsilon_threshold,
            search_space_coverage=0.0,
            total_evaluations=0,
            success=False,
            method='pso_niching',
            method_config=config.__dict__,
            error_message='No dimensions to optimize',
        )

    if config.seed is not None:
        np.random.seed(config.seed)

    # Ensure mechanism has correct n_steps
    if mechanism.n_steps != target.n_steps:
        mechanism._n_steps = target.n_steps

    # Create fast fitness function using Mechanism
    fitness_func = create_mechanism_fitness(
        mechanism=mechanism,
        target=target,
        target_joint=target.joint_name,
        metric=metric,
        phase_invariant=phase_invariant,
        phase_align_method=phase_align_method,
    )

    # Bounds
    lower_bounds = np.array([b[0] for b in dimension_bounds_spec.bounds])
    upper_bounds = np.array([b[1] for b in dimension_bounds_spec.bounds])
    bounds = (lower_bounds, upper_bounds)

    # Initial error
    initial_values = np.array(dimension_bounds_spec.initial_values)
    initial_error = fitness_func(tuple(initial_values))

    if verbose:
        logger.info('Starting PSO Niching multi-solution optimization')
        logger.info(f'  Dimensions: {n_dims}')
        logger.info(f'  Particles: {config.n_particles}')
        logger.info(f'  Iterations: {config.n_iterations}')
        logger.info(f'  Species radius: {config.species_radius}')
        logger.info(f'  Speciation method: {config.speciation_method}')
        logger.info(f'  Init mode: {config.init_mode}')
        logger.info(f'  Initial error: {initial_error:.6f}')

    # Initialize swarm positions
    # TODO: Refactor initialize_swarm_positions to take Mechanism directly
    pylink_data_for_init = mechanism.to_dict()
    init_positions = initialize_swarm_positions(
        n_particles=config.n_particles,
        dimension_bounds_spec=dimension_bounds_spec,
        pylink_data=pylink_data_for_init,
        target=target,
        mode=config.init_mode,
        seed=config.seed,
    )

    # Create niching topology
    # TODO: Refactor NichingTopology to take Mechanism directly
    topology = NichingTopology(
        config=config,
        dimension_bounds_spec=dimension_bounds_spec,
        pylink_data=pylink_data_for_init,
        feature_extractor=feature_extractor,
        static=False,
    )

    # PSO options
    options = {'c1': config.c1, 'c2': config.c2, 'w': config.w}

    # Create swarm
    swarm = P.create_swarm(
        n_particles=config.n_particles,
        dimensions=n_dims,
        options=options,
        bounds=bounds,
        init_pos=init_positions,
    )

    vh = VelocityHandler(strategy='unmodified')
    bh = BoundaryHandler(strategy='periodic')

    total_evaluations = 0
    convergence_history = []

    def evaluate_swarm(positions: np.ndarray) -> np.ndarray:
        nonlocal total_evaluations
        costs = np.array([fitness_func(tuple(pos)) for pos in positions])
        total_evaluations += len(positions)
        return costs

    # Main PSO loop
    for iteration in range(config.n_iterations):
        swarm.current_cost = evaluate_swarm(swarm.position)
        swarm.pbest_cost = evaluate_swarm(swarm.pbest_pos)
        swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm)

        if iteration % config.speciation_frequency == 0 or iteration == 0:
            swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)

        convergence_history.append(float(swarm.best_cost))

        if verbose and (iteration + 1) % 20 == 0:
            info = topology.get_species_info()
            logger.info(
                f'  Iteration {iteration + 1}/{config.n_iterations}: '
                f'best={swarm.best_cost:.6f}, species={info["n_species"]}',
            )

        swarm.velocity = topology.compute_velocity(swarm, vh=vh, bounds=bounds)
        swarm.position = topology.compute_position(swarm, bounds=bounds, bh=bh)

    # Final species update
    topology.compute_gbest(swarm)

    if verbose:
        info = topology.get_species_info()
        logger.info('PSO Niching completed')
        logger.info(f'  Total evaluations: {total_evaluations}')
        logger.info(f'  Final species: {info["n_species"]}')
        logger.info(f'  Best error: {swarm.best_cost:.6f}')

    # Extract solutions
    solutions = []
    species_info = topology.get_species_info()

    for species_id, sinfo in species_info['species_bests'].items():
        species_pos = np.array(sinfo['position'])
        species_cost = sinfo['cost']

        optimized_dims = dict(zip(dimension_bounds_spec.names, species_pos))

        # Update mechanism with optimized dimensions and return copy
        mechanism.set_dimensions(species_pos)
        optimized_mechanism = mechanism.copy()

        sol = Solution(
            success=True,
            optimized_dimensions=optimized_dims,
            optimized_mechanism=optimized_mechanism,
            initial_error=initial_error,
            final_error=species_cost,
            iterations=config.n_iterations,
            cluster_id=species_id,
            distance_to_best=0.0,
            uniqueness_score=1.0,
        )
        solutions.append(sol)

    solutions.sort(key=lambda s: s.final_error)

    if not solutions:
        return MultiSolutionResult(
            solutions=[],
            best_solution=None,
            n_unique_clusters=0,
            epsilon_threshold=config.epsilon_threshold,
            search_space_coverage=0.0,
            total_evaluations=total_evaluations,
            success=False,
            method='pso_niching',
            method_config=config.__dict__,
            error_message='No solutions found',
        )

    # Compute distances and uniqueness
    best_pos = np.array(list(solutions[0].optimized_dimensions.values()))
    all_positions = np.array([list(s.optimized_dimensions.values()) for s in solutions])

    for i, sol in enumerate(solutions):
        sol_pos = np.array(list(sol.optimized_dimensions.values()))
        sol.distance_to_best = float(np.linalg.norm(sol_pos - best_pos))

    if len(solutions) > 1:
        distances = squareform(pdist(all_positions, metric='euclidean'))
        for i, sol in enumerate(solutions):
            other_distances = np.delete(distances[i], i)
            if len(other_distances) > 0:
                min_dist = np.min(other_distances)
                max_dist = np.max(other_distances) if np.max(other_distances) > 0 else 1.0
                sol.uniqueness_score = float(min_dist / max_dist)

    # Coverage estimation
    if len(all_positions) > 1:
        try:
            from scipy.spatial import ConvexHull
            normalized = (all_positions - lower_bounds) / (upper_bounds - lower_bounds)
            if len(normalized) > n_dims:
                hull = ConvexHull(normalized)
                coverage = min(hull.volume, 1.0)
            else:
                coverage = 0.1
        except Exception:
            coverage = 0.1
    else:
        coverage = 0.0

    return MultiSolutionResult(
        solutions=solutions,
        best_solution=solutions[0],
        n_unique_clusters=len(solutions),
        epsilon_threshold=config.epsilon_threshold,
        search_space_coverage=coverage,
        total_evaluations=total_evaluations,
        success=True,
        method='pso_niching',
        method_config=config.__dict__,
    )
