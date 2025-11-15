
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


class Link(BaseModel):
    """A link in the mechanical linkage system with validation."""
    
    length: Annotated[float, Field(gt=0, description="Length of the link in meters")]
    name: Optional[str] = Field(default=None, description="Optional name for the link")
    n_iterations: Annotated[int, Field(ge=1, le=10000, description="Number of iterations for simulation")] = 100
    fixed_loc: Optional[Tuple[float, float]] = Field(default=None, description="Fixed location coordinates (x, y)")
    has_fixed: bool = Field(default=False, description="Whether the link has a fixed location")
    has_constraint: bool = Field(default=False, description="Whether the link has constraints")
    path: Optional[np.ndarray] = Field(default=None, description="Path array for the link")
    is_driven: bool = Field(default=False, description="Whether this is a driven link")
    flip: bool = Field(default=False, description="Whether to flip the link orientation")
    
    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
        "validate_assignment": True,      # Validate on assignment
        "extra": "allow",                 # Allow extra attributes like pos1, pos2
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
        self.pos1 = np.zeros((self.n_iterations, 2))
        self.pos2 = np.zeros((self.n_iterations, 2))
        
        if self.fixed_loc is not None:
            self.has_fixed = True
            self.has_constraint = True
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
