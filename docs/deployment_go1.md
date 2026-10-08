## Deployment (Real-World Go1)

- [What differs from the Go2](#what-differs-from-the-go2)
- [Hardware Setup](#hardware-setup)
- [Startup](#startup)
- [Running the Robot](#running-the-robot)
- [Shutdown](#shutdown)
- [Collecting Data and Finetuning](#collecting-data-and-finetuning)

The Go2 deployment stack in `go2_ros2_ws` is reused for the Unitree Go1. Only the parts that touch the robot or the lidar are swapped; the observer, world model + MPPI controller, state machine and data logger are unchanged. This setup is for **flat ground without a lidar**: the terrain height map that the Go2 builds from its lidar is replaced by a flat map at the measured body height.

### What differs from the Go2

| Go2 | Go1 |
|---|---|
| Robot talks `unitree_go` messages over DDS | `go1_bridge` node runs the `unitree_legged_sdk` UDP loop and publishes/consumes the same `unitree_go` messages on `/lowstate`, `/lowcmd`, `/wirelesscontroller` |
| `/clock` from the lidar IMU | `/clock` from the bridge (host clock) |
| Base velocity from Point-LIO or motion capture | `leg_odometry.py`: leg kinematics + IMU + foot contact |
| Height map from lidar elevation mapping | `flat_elevation.py`: flat map at the body height measured from the stance legs |
| `go2_description` | `go1_description` (URDF only; meshes optional) |
| `simdist_controller.yaml` (checkpoint `go2_2026-02-12`) | `simdist_controller_go1.yaml` (checkpoint `go1_paper`) |

Requirements: a **Go1 EDU** (low-level joint control is only available on the EDU), a control PC with an NVIDIA GPU (the MPPI planner runs at 50 Hz on the GPU) connected to the robot by Ethernet, and a pretrained Go1 world model in `checkpoints/models/` (see [Unitree Go1](pretraining_go2.md#unitree-go1)).

### Hardware Setup

On the control PC, install docker and the nvidia container toolkit if needed, then build the container (this also downloads the Go1 SDK):

```bash
sudo apt-get update
./go2_ros2_ws/setup/docker_install.sh && ./go2_ros2_ws/setup/nvidia-container-toolkit.sh
./go2_ros2_ws/docker/build.sh
```

Connect the PC to the Go1's Ethernet port and give the interface a static address on the robot's network, e.g. `192.168.123.100/24`. The bridge talks to the main control board at `192.168.123.10:8007` (configurable in [`go2_ros2_ws/src/config/config/go1.yaml`](../go2_ros2_ws/src/config/config/go1.yaml)). Set the interface name:

```bash
cp go2_ros2_ws/.env.example go2_ros2_ws/.env
# edit go2_ros2_ws/.env and set CYCLONEDDS_IFACE=<your_interface_name>
```

Optionally fetch the Go1 meshes for RViz (not needed to run):

```bash
./go2_ros2_ws/setup/fetch_go1_meshes.sh
```

Start the container and build the workspace:

```bash
./go2_ros2_ws/scripts/run.sh
colcon build
```

Configure the controller in [`simdist_controller_go1.yaml`](../go2_ros2_ws/src/config/config/simdist_controller_go1.yaml) (`model` and `logging` sections) and the control task in [`control.yaml`](../go2_ros2_ws/src/config/config/control.yaml), as for the Go2. Rebuild with `colcon build` after changing configs.

### Startup

1. Put the Go1 on the ground (or hang it in a harness for the first tests), turn it on and let it stand up.
2. Switch the robot to **low-level mode** with the remote: `L2+A`, `L2+A`, `L2+B`, then `L1+L2+Start`. The robot lies down and the internal controller releases the motors. **Do this before starting the bridge, every time — including receive-only tests.** The first low-level packet the control board receives switches it out of its normal balancing mode, so a robot that is still standing collapses. The bridge prints this warning and waits `go1.start_delay` (5 s) before its first packet so you can abort with `Ctrl+C`.
3. Start the container and the tmux layout:

```bash
./go2_ros2_ws/scripts/run.sh
tmuxp load tmuxp_go1.yaml
```

4. In the focused pane start the sensing side; the command is already typed:

```bash
ros2 launch bringup launch_go1.py
```

The bridge prints the connection settings; `ros2 topic hz /lowstate` should show about 500 Hz and `ros2 topic echo /elevation_vec --once` values around `-0.3` once the robot stands. If `/lowstate` is silent, the robot is not in low-level mode or the PC is not on `192.168.123.x`.

5. Start the controller in the next pane (command already typed). The world model is loaded and the controller initializes once observations arrive.

```bash
ros2 launch bringup launch_control.py controller_config:=simdist_controller_go1.yaml
```

### Running the Robot

The state machine (already running in its pane, started with `robot_config:=go1.yaml` so it uses the Go1 gains: stand Kp 30, walk Kp 25, Kd 1.0) is driven from the keyboard in its pane or the wireless remote, exactly as on the Go2; see [Running the Robot](deployment_go2.md#running-the-robot). The order is prone → stand → walk. The `go1_bridge` applies the SDK's joint limits and power protection (`go1.power_protect_level`, 5 = 50% by default; raise it once the robot walks reliably) and switches the motors to damping if `/lowcmd` stops for `go1.cmd_timeout` seconds.

Safety notes for the Go1:

- Keep a hand on the remote: prone (stop walking) is one key away, and killing the state machine or the bridge puts the motors in damping mode.
- The bridge treats |roll| or |pitch| above `go1.abort_roll_pitch` (0.7 rad) as a fall: the motors go to damping and stay there until the robot is upright **and** the state machine has been switched back to prone, so a righted robot never resumes a walking command by itself.
- The bridge watches the motor temperatures and battery reported in `/lowstate`: above `go1.motor_temp_warn` (60 °C) it logs a warning, above `go1.motor_temp_stop` (70 °C) it holds the motors in damping mode until they cool down, and it warns below `go1.battery_warn` (20 %). Watch them yourself with `ros2 topic echo /lowstate --field motor_state[1].temperature` and `--field bms_state.soc`.
- Use a harness for the first stand/walk tests. The Go1 is lighter than the Go2 and the world model was trained with Go1 motor limits (23.7 N·m), but the stand/prone joint targets in `state_machine.cpp` are shared with the Go2.
- Foot contact (for velocity and height estimation) uses the Go1 foot force sensors with threshold `contact_force_threshold` in `go1.yaml`; check `ros2 topic echo /lowstate --field foot_force` while the robot stands and lift a leg to confirm the threshold separates contact from swing.

### Shutdown

Stop walking (prone), stop the controller and the launch files with `Ctrl+C`, then exit the container. The bridge leaves the motors in damping mode; power-cycle the robot to return it to normal (sport) mode.

### Collecting Data and Finetuning

With `logging.enabled: true` in `simdist_controller_go1.yaml`, every walking episode is written to `datasets/real/go1_simdist_controller`. Process and finetune as in [Adaptation](adaptation.md), passing `system=go1`:

```bash
python scripts/aggregate_realworld_data.py dataset_name=go1_simdist_controller
python scripts/process_data.py dataset_name=go1_simdist_controller system=go1
python scripts/finetune_model.py data.dataset_name=go1_simdist_controller system=go1 checkpoint.resume_checkpoint=go1_paper run_name=go1_finetuned
```

Then point `model.checkpoint` in `simdist_controller_go1.yaml` at `go1_finetuned` and redeploy.
