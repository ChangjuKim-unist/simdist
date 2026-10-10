#!/bin/bash

source /opt/ros/humble/setup.bash
source "$ROS2_WS/install/setup.sh"
export TERM=screen-256color
export RCUTILS_COLORIZED_OUTPUT=1

if [ "$USE_SIM" = "true" ]; then
  echo "Running in simulation mode"
else
  echo "Running in real robot mode"
  if [ -z "${CYCLONEDDS_IFACE:-}" ]; then
    echo "Error: Could not determine network interface for CycloneDDS."
    echo "Configure go2_ros2_ws/.env with the appropriate interface name."
    return 1 2>/dev/null || exit 1
  fi
  echo "Using CycloneDDS interface: $CYCLONEDDS_IFACE"
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  if [ "$CYCLONEDDS_IFACE" = "lo" ]; then
    # All ROS 2 nodes run on this machine and the robot itself does not speak DDS
    # (Go1 over the unitree_legged_sdk bridge): keep DDS on the loopback interface
    # so the 500 Hz topics never touch the robot's network link. Loopback has no
    # multicast, so discovery is unicast over local participant indices.
    export CYCLONEDDS_URI="<CycloneDDS>
    <Domain>
      <General>
        <Interfaces>
          <NetworkInterface name=\"lo\" priority=\"default\" multicast=\"false\"/>
        </Interfaces>
        <AllowMulticast>false</AllowMulticast>
      </General>
      <Discovery>
        <ParticipantIndex>auto</ParticipantIndex>
        <MaxAutoParticipantIndex>120</MaxAutoParticipantIndex>
      </Discovery>
      <Internal>
        <SocketReceiveBufferSize min=\"10MB\"/>
        <SocketSendBufferSize min=\"2MB\"/>
      </Internal>
    </Domain>
  </CycloneDDS>"
  else
    export CYCLONEDDS_URI="<CycloneDDS>
    <Domain>
      <General>
        <Interfaces>
          <NetworkInterface name=\"${CYCLONEDDS_IFACE}\" priority=\"default\" multicast=\"default\"/>
        </Interfaces>
      </General>
    </Domain>
  </CycloneDDS>"
  fi
fi
