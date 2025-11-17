
import numpy as np
from functools import partial
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from scipy.optimize import fsolve, least_squares
from scipy.optimize import least_squares
from functools import partial
from itertools import cycle
from scipy.optimize import least_squares

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Annotated

class Node(BaseModel):
    #id: str = Field(description="Unique identifier for the node")
    name: str = Field(default="node", description="Name for the node, must be unique.")
    #pos: Tuple[float, float] = Field(description="Position coordinates (x, y)")
    fixed: bool = Field(default=False, description="Whether the node is fixed in position")
    fixed_loc: Optional[Tuple[float, float]] = Field(default=None, description="Fixed location coordinates (x, y) if fixed")

    target_loc: Optional[Tuple[float, float]] = Field(default=None, description="Target location coordinates (x, y) if applicable")
    target_radius: Optional[float] = Field(default=None, description="Target radius if applicable")
    target_cost_func: Optional[str] = Field(default=None, description="Target cost function if applicable")
    n_iterations: Annotated[int, Field(ge=1, le=1000, description="Number of iterations for simulation")]
    
    init_pos: Optional[Tuple[float, float]] = Field(description="Initial position coordinates (x, y)")
    #unhasables:
    pos: Optional[np.ndarray] = None  #unhashables/mutables:

    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
        # Validate on assignment
         "validate_assignment": True,
        "extra": "forbid", # Allow extra attributes like pos1, pos2
    }

    def model_post_init(self, __context: Any) -> None:
        # Set computed numpy arrays as instance attributes (not model fields)
        if self.pos is None:
            self.pos = np.empty((self.n_iterations, 2))
            self.pos[:] = np.nan
            self.pos[0,:] = self.init_pos
        elif self.init_pos is not None:
            #assert np.shape(self.pos) == (self.n_iterations, 2), "pos must have shape (n_iterations, 2)"
            assert self.pos[0,:][0] == self.init_pos[0] and self.pos[0,:][1] == self.init_pos[1], "pos first entry must match init_pos"
        assert np.shape(self.pos) == (self.n_iterations, 2), "pos must have shape (n_iterations, 2)"
        #self.pos2 = np.zeros((self.n_iterations, 2))
        
        if self.fixed:
            assert self.fixed_loc is not None, "fixed_loc must be set if has_fixed is True"
            #assert self.has_constraint is not None, "has_constraint must be set if has_fixed is True"
            #self.has_constraint = True
            #self.pos += self.fixed_loc


    def as_dict(self):
        """Convert to dictionary, similar to dataclass asdict()"""
        # Get the model dict and add the computed fields
        data = self.model_dump()
        if hasattr(self, 'pos'):
            data['pos'] = self.pos
        return data


class Link(BaseModel):
    """A link in the mechanical linkage system with validation."""
    #id: str = Field(description="Unique identifier for the link")
    name: str = Field(description="Name for the link, must be unique.")
    length: Annotated[float, Field(gt=0, le=100, description="Length of the link in inches")]
    
    target_length: Optional[float] = Field(default=None, description="Target length for the link if applicable")
    target_cost_func: Optional[str] = Field(default=None, description="Target cost function for the link if applicable")

    n_iterations: Annotated[int, Field(ge=1, le=1000, description="Number of iterations for simulation")]
    fixed_loc: Optional[Tuple[float, float]] = Field(default=None, description="Fixed location coordinates (x, y)")
    has_fixed: bool = Field(description="Whether the link has a fixed location")
    has_constraint: bool = Field(default=False, description="Whether the link has constraints")
    path: Optional[np.ndarray] = Field(default=None, description="Path array for the link")
    is_driven: bool = Field(default=False, description="Whether this is a driven link")
    flip: bool = Field(default=False, description="Whether to flip the link orientation")
    zlevel: int = Field(default=0, description="Z-level for rendering and physical building order." \
    " higher z-levels are on top of lower z-levels. When physically constructing an automaton two links that are connected" \
    " in the graph can NEVER be on the same z-level. Links that collide with each other should also be on different z-levels.")

     #unhashables/mutables:
    pos1: Optional[np.ndarray] = None  #unhashables/mutables:
    pos2: Optional[np.ndarray] = None

    #unhashables/mutables:
    #length: Annotated[float, Field(gt=0, le=100, description="Length of the link in inches")]

    # model_config = {
    #     "arbitrary_types_allowed": True,  # Allow numpy arrays
    #     #"validate_assignment": True,      # Validate on assignment
    #     "extra": "allow",                 # Allow extra attributes like pos1, pos2
    #     #"extra": "forbid", # Allow extra attributes like pos1, pos2
    # }
    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
        # Validate on assignment
        "validate_assignment": True,
        "extra": "forbid", # Allow extra attributes like pos1, pos2
    }
    
    @field_validator('fixed_loc')
    @classmethod
    def validate_fixed_loc(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("fixed_loc must be a tuple of exactly 2 numbers")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("fixed_loc coordinates must be numbers")
        return v
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            raise ValueError("path must be a numpy array")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize computed fields after validation"""
        # Set computed numpy arrays as instance attributes (not model fields)
        if self.pos1 is None:
            self.pos1 = np.zeros((self.n_iterations, 2))
        else:
            assert np.shape(self.pos1) == (self.n_iterations, 2), "pos1 must have shape (n_iterations, 2)"
        if self.pos2 is None:
            self.pos2 = np.zeros((self.n_iterations, 2))
        else:
            assert np.shape(self.pos2) == (self.n_iterations, 2), "pos2 must have shape (n_iterations, 2)"
        
        if self.has_fixed:
            assert self.fixed_loc is not None, "fixed_loc must be set if has_fixed is True"
            assert np.shape(self.fixed_loc) == (2,), "fixed_loc must be a tuple of 2 numbers"
            #self.has_fixed = True
            self.has_constraint = True
            # NOTE pos1 must be the fixed one!!!
            assert np.all(self.pos1 == 0), "pos1 must be all zeros before adding fixed_loc"
            self.pos1 += self.fixed_loc
    
    def as_dict(self):
        """Convert to dictionary, similar to dataclass asdict()"""
        # Get the model dict and add the computed fields
        data = self.model_dump()
        if hasattr(self, 'pos1'):
            data['pos1'] = self.pos1
        if hasattr(self, 'pos2'):
            data['pos2'] = self.pos2
        return data
    
    # def get_required_fields(model: type[BaseModel]) -> set[str]:
    #     required_fields = set()
    #     for field_name, field_info in model.model_fields.items():
    #         if field_info.default is Ellipsis or field_info.default is None and not hasattr(field_info, 'default_factory'):
    #             required_fields.add(field_name)
    #     return required_fields


class DriveGear(BaseModel):
    """A drive gear in the mechanical system with validation."""
    
    radius: Annotated[float, Field(gt=0, description="Radius of the drive gear")]
    fixed_loc: Optional[Tuple[float, float]] = Field(default=None, description="Fixed location coordinates (x, y)")
    
    model_config = {
        "validate_assignment": True,
        "extra": "allow",  # Allow extra attributes like has_fixed
    }
    
    @field_validator('fixed_loc')
    @classmethod
    def validate_fixed_loc(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("fixed_loc must be a tuple of exactly 2 numbers")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("fixed_loc coordinates must be numbers")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize computed fields after validation"""
        self.has_fixed = True
    
    def as_dict(self):
        """Convert to dictionary for compatibility"""
        data = self.model_dump()
        if hasattr(self, 'has_fixed'):
            data['has_fixed'] = self.has_fixed
        return data


class AcxGraph:
    def __init__(self, links: Optional[List[Link]] = None, nodes: Optional[List[Any]] = None, name: Optional[str] = None):
        self.links = links or []
        self.nodes = nodes or []
        self.name = name