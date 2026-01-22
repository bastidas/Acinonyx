"""
Test for pylink trajectory computation using acinonyx_api endpoints.

This test validates:
- Copying test file to pygraphs directory
- Loading saved pylink graphs using load_pylink_graph
- Computing trajectories using the compute_pylink_trajectory endpoint
- Verifying the trajectory output structure and data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.acinonyx_api import compute_pylink_trajectory
from backend.acinonyx_api import load_pylink_graph
from backend.acinonyx_api import save_pylink_graph

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path is set


def test_4bar_trajectory():
    """Test loading and computing trajectory for 4bar linkage"""

    # Load the 4bar test file
    test_file = Path(__file__).parent.parent / 'demo' / 'test_graphs' / '4bar.json'

    with open(test_file) as f:
        pylink_data = json.load(f)

    # Step 1: Save the graph using save_pylink_graph
    save_result = save_pylink_graph(pylink_data)

    assert save_result['status'] == 'success', f"Save failed: {save_result.get('message')}"

    # Step 2: Load it back using load_pylink_graph
    load_result = load_pylink_graph(filename=save_result['filename'])

    assert load_result['status'] == 'success', f"Load failed: {load_result.get('message')}"
    loaded_data = load_result['data']

    # Step 3: Compute trajectory
    print('\nStep 3: Computing trajectory...')
    result = compute_pylink_trajectory(loaded_data)

    # Verify result
    assert result['status'] == 'success', f"Computation failed: {result.get('message')}"

    print(f"✓ Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print(f"  Execution time: {result['execution_time_ms']:.2f}ms")
    print(f"  Steps computed: {result['n_steps']}")

    # Verify trajectories
    trajectories = result['trajectories']
    joint_types = result['joint_types']

    for joint_name, positions in trajectories.items():
        joint_type = joint_types.get(joint_name, 'Unknown')

        # Show first and last position
        if positions:
            first_pos = positions[0]
            last_pos = positions[-1]

    # Validate trajectory structure
    assert 'trajectories' in result, 'Missing trajectories in result'
    assert len(trajectories) > 0, 'No trajectories computed'

    # Check that all joints have the correct number of steps
    n_steps = result['n_steps']
    for joint_name, positions in trajectories.items():
        assert len(positions) == n_steps, \
            f'Joint {joint_name} has {len(positions)} positions, expected {n_steps}'

    # Verify static joints remain static
    for joint_name, joint_type in joint_types.items():
        if joint_type == 'Static':
            positions = trajectories[joint_name]
            first_pos = positions[0]
            # Check all positions are the same for static joints
            for pos in positions:
                assert pos == first_pos, \
                    f'Static joint {joint_name} position changed: {first_pos} -> {pos}'


if __name__ == '__main__':
    # Run tests
    try:
        test_4bar_trajectory()

    except AssertionError as e:
        print(f'\n❌ TEST FAILED: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
