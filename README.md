# Optimizing-Initial-Feature-Mapping-Variables-from-Given-Designs-via-Tracking
A modular and fully differentiable Python framework for reconstructing 
density-based topology optimization results using parametric capsule features.

The system provides analytical gradients and exact Hessians, supports 
smooth aggregation strategies, and enables staged nonlinear optimization 
with constraint handling and automatic feature refinement.


## Paper

Patrick Jung (2026).
Optimizing Initial Feature-Mapping Variables from Given Designs via Tracking.
arXiv:2602.13005  
https://arxiv.org/abs/2602.13005


## Architecture Overview

The framework is organized into modular components:

- **pill module**  
  Defines parametric capsule features (line segments with radii) and provides
  analytical signed distance functions, gradients, and Hessians.

- **optimization_definition module**  
  Formulates the nonlinear optimization problem with support for:
  - Least-squares tracking objectives
  - Reward-based initialization objectives
  - Exact Hessian or first-order solver interfaces
  - IPOPT and SciPy compatibility

- **staged optimization runner**  
  Executes multi-stage optimization pipelines (orientation → convergence),
  applies geometric constraints, and manages solver settings via JSON configs.

- **heuristics & refinement**
  - Automatic pruning of weak or redundant features
  - Feature merging based on geometric criteria
  - Greedy additive refinement to extend the design

The system transforms voxel-based density fields into compact,
constraint-safe feature representations.



## Installation

Create a virtual environment and install dependencies:

pip install -r requirements.txt


## Example Usage

python scripts/optimize.py \
    density/files/5bar_80.density.xml \
    opt_config/example_run.json \
    --output_dir ./example_run


## Workflow

1. Generate a target density field using CFS:
   - Create mesh
   - Define physical configuration
   - Run CFS solver
   - Obtain .density.xml file

2. Run the feature-based reconstruction:
   - Load density field
   - Initialize features
   - Execute staged optimization
   - Apply pruning / merging / refinement


## Technical Highlights

- Analytical signed distance functions
- Exact first and second derivatives
- Smooth aggregation (p-norm / softmax)
- Constraint-safe nonlinear optimization
- Exact Hessian support (IPOPT)
- Multi-stage reconstruction strategy
- Automated feature pruning and merging


## Contact

Patrick Jung  
paju_99@web.de


Note: The generation of density XML files requires an external CFS solver.
The reconstruction framework itself operates on existing density files.

