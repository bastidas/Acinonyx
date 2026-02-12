#!/usr/bin/env python3
"""
Reflection combinations demo – plot each viable reflection configuration.

Loads a mechanism, gets viable reflectable axes from get_reflectable_edges(),
and produces one plot per configuration: identity (original) plus one plot
for each reflection over a viable axis (crank angle unchanged).

Run:
    python demo/reflection_combinations_demo.py

Output: user/demo/reflection_combinations/*.png
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from configs.appconfig import USER_DIR
from demo.helpers import load_mechanism
from demo.helpers import print_section
from pylink_tools.mechanism import Mechanism
from viz_tools.demo_viz import variation_plot
from viz_tools.viz_styling import STYLE

logger = logging.getLogger(__name__)

# Try complex first; fall back to simple (4-bar) if no viable reflections yet
MECHANISM = 'complex'
FALLBACK_MECHANISM = 'simple'
N_STEPS = 64
OUTPUT_DIR = USER_DIR / 'demo' / 'reflection_combinations'
# Set to True to log why each candidate axis is rejected (to debug missing edges)
REFLECT_DEBUG = False


def _safe_filename(label: str, index: int) -> str:
    """Turn a config label into a safe filename stem."""
    safe = re.sub(r'[^\w\-]', '_', label)
    safe = safe.strip('_')[:40] or 'config'
    return f"{index:02d}_{safe}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / 'reflection_combinations.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode='w'),
        ],
    )
    logger.info('Reflection combinations demo: %s (fallback: %s)', MECHANISM, FALLBACK_MECHANISM)
    print_section('REFLECTION COMBINATIONS DEMO')
    print(f'Mechanism: {MECHANISM} (fallback: {FALLBACK_MECHANISM})')
    print(f'Output: {OUTPUT_DIR}')
    print(f'Log: {log_path}')

    if REFLECT_DEBUG:
        logging.getLogger('pylink_tools.mechanism').setLevel(logging.DEBUG)

    print('\nLoading mechanism...')
    mechanism, target_joint, description = load_mechanism(MECHANISM, n_steps=N_STEPS)
    reflectable_edges = mechanism.get_reflectable_edges(debug=REFLECT_DEBUG)

    if not reflectable_edges and MECHANISM != FALLBACK_MECHANISM:
        print(f'  No viable reflectable edges for "{MECHANISM}"; using "{FALLBACK_MECHANISM}".')
        mechanism, target_joint, description = load_mechanism(FALLBACK_MECHANISM, n_steps=N_STEPS)
        reflectable_edges = mechanism.get_reflectable_edges(debug=REFLECT_DEBUG)

    print(f'  {description}')
    print(f'  Target joint: {target_joint}')
    print(f'  Viable reflectable axes: {len(reflectable_edges)}')

    # Print and log the reflectable edges found
    print_section('Reflectable edges')
    for i, r in enumerate(reflectable_edges):
        label = f"{r.axis_node_1} – {r.axis_node_2}"
        print(f'  [{i+1}] {label}')
        logger.info('reflectable_edge %s %s %s', r.axis_node_1, r.axis_node_2, label)
    if not reflectable_edges:
        print('  (none)')
        logger.info('reflectable_edges none')

    # Configurations: identity + one per viable reflection
    configs: list[tuple[str, Mechanism]] = [('Identity', mechanism)]
    for r in reflectable_edges:
        label = f"Reflect {r.axis_node_1}–{r.axis_node_2}"
        reflected = mechanism.reflect_over_axis(r)
        configs.append((label, reflected))

    print_section('Plotting each configuration')
    for i, (label, mech) in enumerate(configs):
        stem = _safe_filename(label, i)
        out_path = OUTPUT_DIR / f"{stem}.png"
        print(f'  [{i+1}/{len(configs)}] {label} -> {out_path.name}')
        variation_plot(
            target_joint=target_joint,
            out_path=out_path,
            base_mechanism=mech,
            variation_mechanisms=[],
            title=label,
            subtitle=f'Configuration {i+1} of {len(configs)}',
            show_linkages=True,
            style=STYLE,
        )

    print_section('DONE')
    print('Outputs:')
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f'  {f.name}')
    print(f'  Log: {log_path}')


if __name__ == '__main__':
    main()
