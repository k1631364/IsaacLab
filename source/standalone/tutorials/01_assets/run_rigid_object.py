# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # GroundPlaneCfg can be found in /workspace/isaaclab/source/extensions/omni.isaac.lab/omni/isaac/lab/sim/spawners/from_files/from_files_cfg.py
    # SpawnerCfg can be found in /workspace/isaaclab/source/extensions/omni.isaac.lab/omni/isaac/lab/sim/spawners/spawner_cfg.py
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    # module_path = prim_utils.__file__
    # print(f"The module 'omni.isaac.core.utils.prims' is being imported from: {module_path}")
    # Create prim can be found in /workspace/isaaclab/_isaac_sim/exts/omni.isaac.core/omni/isaac/core/utils/prims.py
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)

    # return the scene information
    scene_entities = {"cone": cone_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cone_object = entities["cone"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # To reset the simulation state of the spawned rigid object prims, we need to set their pose and velocity. Together they define the root state of the spawned rigid objects. 
            # https://isaac-sim.github.io/IsaacLab/source/tutorials/01_assets/run_rigid_object.html
            # reset root state
            root_state = cone_object.data.default_root_state.clone()
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
            )
            # write root state to simulation            
            cone_object.write_root_state_to_sim(root_state)
            # reset buffers
            cone_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        
        # If you want to change linear velocity instead of pos, 
        # root_state[:, 7:10] += root_state[:, 7:10]+0.5
        # Data representation is defined in /workspace/isaaclab/source/extensions/omni.isaac.lab/omni/isaac/lab/assets/rigid_object/rigid_object_data.py
        # root_state = cone_object.data.default_root_state.clone()
        original_vel = root_state[:, 7:]
        num_obj = original_vel.shape[0]
        vel_dim = original_vel.shape[1]
        new_linvel = torch.full((num_obj,3), 0.5)
        new_angvel = torch.full((num_obj,3), 0.0)
        new_linvel = new_linvel.to(cone_object.device)
        new_angvel = new_angvel.to(cone_object.device)
        # new_linvel = original_vel[:,:3]+0.0
        # new_angvel = original_vel[:,3:]+0.0
        # new_linvel[2,:] = original_vel[2,:3]+0.2
        new_linvel[:,0] = original_vel[:,0]+0.2
        # print(new_linvel.shape)
        # print(new_angvel.shape)
        # print("original vel")
        # print(original_vel)
        # print(new_linvel)
        
        cone_object.set_velocities(new_linvel, new_angvel)
        
        # apply sim data
        cone_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cone_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_state_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
