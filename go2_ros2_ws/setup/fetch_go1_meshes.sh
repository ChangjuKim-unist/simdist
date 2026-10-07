#!/usr/bin/env bash
# Downloads the Go1 meshes (about 70 MB) into src/thirdparty/go1_description/meshes
# so the robot shows up in RViz. The URDF works without them.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${WS}/src/thirdparty/go1_description/meshes"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

git clone --depth 1 https://github.com/unitreerobotics/unitree_ros "${TMP}/unitree_ros"
rm -rf "${DEST}"
cp -r "${TMP}/unitree_ros/robots/go1_description/meshes" "${DEST}"
echo "Go1 meshes copied to ${DEST}. Rebuild with: colcon build --packages-select go1_description"
