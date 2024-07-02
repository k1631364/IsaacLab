# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning meshes in the simulation.

NVIDIA Omniverse deals with meshes as USDGeomMesh prims. This sub-module provides various
configurations to spawn different types of meshes. Based on the configuration, the spawned prim can be:

* a visual mesh (no physics)
* a static collider (no rigid body)
* a deformable body (with deformable properties).
"""

from .meshes import spawn_mesh_capsule, spawn_mesh_cone, spawn_mesh_cuboid, spawn_mesh_cylinder, spawn_mesh_sphere
from .meshes_cfg import MeshCapsuleCfg, MeshCfg, MeshConeCfg, MeshCuboidCfg, MeshCylinderCfg, MeshSphereCfg