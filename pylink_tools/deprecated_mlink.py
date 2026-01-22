# Demo: Multi-link mechanism with multiple attached links (strider-style)
from __future__ import annotations

import numpy as np
import pylinkage as pl

# Configuration parameters for the multi-link mechanism
DIM_NAMES = (
    'triangle', 'aperture', 'femur', 'rockerL', 'rockerS', 'phi', 'tibia', 'shank',
)

DIMENSIONS = (
    2.0,           # triangle: distance scale
    np.pi / 4,     # aperture: angle offset
    1.8,           # femur: upper leg length
    2.6,           # rockerL: longer rocker arm
    1.4,           # rockerS: shorter rocker arm
    np.pi + 0.2,   # phi: phase angle
    2.5,           # tibia: lower leg length
    1.8,           # shank: foot segment
)

BOUNDS = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (8, 2 * np.pi, 7.2, 10.4, 5.6, 2 * np.pi, 10, 7.6),
)


def create_multi_link_demo(constraints=DIMENSIONS):
    """
    Create a multi-link visualization with a working 4-bar base plus additional linked elements.
    Uses proven geometry from the demo with extended visual markers.
    """

    # Fixed body points
    linkage = {}

    # Origin - body center
    linkage['A'] = pl.Static(x=0, y=0, name='Origin')

    # Main motor crank - proven working configuration
    linkage['Crank'] = pl.Crank(
        x=0, y=1,
        joint0=linkage['A'],
        angle=0.31,
        distance=1.0,
        name='Crank',
    )

    # Main pin - proven working configuration (forms 4-bar with crank and ground)
    linkage['MainPin'] = pl.Revolute(
        x=3, y=2,
        joint0=linkage['Crank'],
        joint1=(3, 0),
        distance0=3,
        distance1=2,
        name='MainPin',
    )

    # Link1: Static reference marker (vertical)
    linkage['Link1'] = pl.Static(
        x=0, y=3,
        name='Link1_Vertical',
    )

    # Link2: Static reference marker (top right)
    linkage['Link2'] = pl.Static(
        x=2, y=3,
        name='Link2_TopRight',
    )

    # Link3: Static reference marker (right side)
    linkage['Link3'] = pl.Static(
        x=4, y=1.5,
        name='Link3_Right',
    )

    # Link4: Static reference marker (bottom)
    linkage['Link4'] = pl.Static(
        x=0.5, y=-1.5,
        name='Link4_Bottom',
    )

    # Create the complete linkage
    strider = pl.Linkage(
        joints=tuple(linkage.values()),
        name='Multi-link Mechanism',
    )

    return strider


if __name__ == '__main__':
    print('Creating multi-link mechanism...')
    linkage = create_multi_link_demo()

    print(f'Linkage created: {linkage.name}')
    print(f'Number of joints: {len(linkage.joints)}')

    # Show the animation
    print('Displaying animation...')
    pl.show_linkage(linkage)

    print('Demo complete.')
