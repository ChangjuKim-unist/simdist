// Bridge between a Unitree Go1 and the unitree_go topics used by this workspace.
//
// The Go2 publishes unitree_go/LowState and consumes unitree_go/LowCmd natively
// over DDS. The Go1 instead speaks the unitree_legged_sdk UDP protocol, so this
// node runs the SDK loop and translates both directions:
//
//   SDK LowState  -> /lowstate (unitree_go/LowState), /wirelesscontroller
//   /lowcmd (unitree_go/LowCmd) -> SDK LowCmd
//
// It also publishes /clock from the host clock, since the Go2 setup derives
// /clock from the lidar IMU, which the Go1 does not have.
//
// The Go1 must be in low-level mode (remote: L2+A, L2+A, L2+B, L1+L2+Start)
// before this node is started, and the host must be on the robot's network
// (192.168.123.x).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <unitree_go/msg/wireless_controller.hpp>

#include "unitree_legged_sdk/go1_const.h"
#include "unitree_legged_sdk/joystick.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace sdk = UNITREE_LEGGED_SDK;

namespace {

constexpr int kNumJoints = 12;
constexpr uint8_t kServoMode = 0x0A;  // Go1 motor servo (position/torque) mode

// Joint limits from go1_const.h, by joint-within-leg index (hip, thigh, calf).
constexpr double kJointMin[3] = {sdk::go1_Hip_min, sdk::go1_Thigh_min, sdk::go1_Calf_min};
constexpr double kJointMax[3] = {sdk::go1_Hip_max, sdk::go1_Thigh_max, sdk::go1_Calf_max};

}  // namespace

class Go1Bridge : public rclcpp::Node {
public:
  Go1Bridge() : Node("go1_bridge") {
    declare_parameter<std::string>("topics.lowstate", "/lowstate");
    declare_parameter<std::string>("topics.lowcmd", "/lowcmd");
    declare_parameter<std::string>("topics.wirelesscontroller", "/wirelesscontroller");
    declare_parameter<std::string>("go1.robot_ip", "192.168.123.10");
    declare_parameter<int>("go1.robot_port", 8007);
    declare_parameter<int>("go1.local_port", 8090);
    declare_parameter<double>("go1.rate", 500.0);
    // Fraction of motor power allowed by the SDK's PowerProtect: 1 (10%) .. 10 (100%).
    declare_parameter<int>("go1.power_protect_level", 5);
    // Without a fresh /lowcmd for this long the motors are put in damping mode.
    declare_parameter<double>("go1.cmd_timeout", 0.5);
    declare_parameter<double>("go1.damping_kd", 2.0);
    declare_parameter<bool>("go1.publish_clock", true);

    const auto lowstate_topic = get_parameter("topics.lowstate").as_string();
    const auto lowcmd_topic = get_parameter("topics.lowcmd").as_string();
    const auto wireless_topic = get_parameter("topics.wirelesscontroller").as_string();
    const auto robot_ip = get_parameter("go1.robot_ip").as_string();
    const int robot_port = get_parameter("go1.robot_port").as_int();
    const int local_port = get_parameter("go1.local_port").as_int();
    rate_ = get_parameter("go1.rate").as_double();
    power_protect_level_ = get_parameter("go1.power_protect_level").as_int();
    cmd_timeout_ = get_parameter("go1.cmd_timeout").as_double();
    damping_kd_ = get_parameter("go1.damping_kd").as_double();
    publish_clock_ = get_parameter("go1.publish_clock").as_bool();

    lowstate_pub_ = create_publisher<unitree_go::msg::LowState>(lowstate_topic, 10);
    wireless_pub_ = create_publisher<unitree_go::msg::WirelessController>(wireless_topic, 10);
    if (publish_clock_) {
      clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    }
    lowcmd_sub_ = create_subscription<unitree_go::msg::LowCmd>(
        lowcmd_topic, 1, std::bind(&Go1Bridge::lowcmd_callback, this, std::placeholders::_1));

    sdk::InitEnvironment();
    safety_ = std::make_unique<sdk::Safety>(sdk::LeggedType::Go1);
    udp_ = std::make_unique<sdk::UDP>(sdk::LOWLEVEL, local_port, robot_ip.c_str(), robot_port);
    udp_->InitCmdData(sdk_cmd_);
    set_damping(sdk_cmd_);

    RCLCPP_INFO(get_logger(), "Go1 bridge: %s:%d <-> local %d at %.0f Hz, power level %d/10",
                robot_ip.c_str(), robot_port, local_port, rate_, power_protect_level_);
    running_ = true;
    loop_thread_ = std::thread(&Go1Bridge::run_loop, this);
  }

  ~Go1Bridge() override {
    running_ = false;
    if (loop_thread_.joinable()) {
      loop_thread_.join();
    }
  }

private:
  // ---- /lowcmd -> SDK ----------------------------------------------------
  void lowcmd_callback(const unitree_go::msg::LowCmd::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    for (int i = 0; i < kNumJoints; ++i) {
      const auto &m = msg->motor_cmd[i];
      auto &c = sdk_cmd_.motorCmd[i];
      c.mode = kServoMode;
      c.q = clamp_q(i, m.q);
      c.dq = m.dq;
      c.tau = m.tau;
      c.Kp = m.kp;
      c.Kd = m.kd;
    }
    last_cmd_time_ = std::chrono::steady_clock::now();
    have_cmd_ = true;
  }

  // PosStopF is the SDK sentinel for "no position target"; it must not be clamped.
  static float clamp_q(int joint, float q) {
    if (q >= sdk::PosStopF * 0.5) {
      return sdk::PosStopF;
    }
    const int j = joint % 3;
    return static_cast<float>(std::clamp(static_cast<double>(q), kJointMin[j], kJointMax[j]));
  }

  void set_damping(sdk::LowCmd &cmd) const {
    for (int i = 0; i < kNumJoints; ++i) {
      auto &c = cmd.motorCmd[i];
      c.mode = kServoMode;
      c.q = sdk::PosStopF;
      c.dq = sdk::VelStopF;
      c.tau = 0.0f;
      c.Kp = 0.0f;
      c.Kd = static_cast<float>(damping_kd_);
    }
  }

  // ---- SDK loop -----------------------------------------------------------
  void run_loop() {
    const auto period = std::chrono::duration<double>(1.0 / rate_);
    auto next = std::chrono::steady_clock::now();
    while (running_ && rclcpp::ok()) {
      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);

      udp_->Recv();
      udp_->GetRecv(sdk_state_);
      publish_state();

      sdk::LowCmd cmd;
      {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        const double age =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - last_cmd_time_).count();
        if (!have_cmd_ || age > cmd_timeout_) {
          if (have_cmd_ && !timed_out_) {
            RCLCPP_WARN(get_logger(), "No /lowcmd for %.2fs: damping mode", age);
          }
          timed_out_ = have_cmd_;
          set_damping(sdk_cmd_);
        } else {
          timed_out_ = false;
        }
        cmd = sdk_cmd_;
      }
      const int protect = safety_->PowerProtect(cmd, sdk_state_, power_protect_level_);
      if (protect < 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "PowerProtect triggered (level %d)", power_protect_level_);
      }
      udp_->SetSend(cmd);
      udp_->Send();

      std::this_thread::sleep_until(next);
    }
  }

  // ---- SDK -> /lowstate, /wirelesscontroller, /clock ----------------------
  void publish_state() {
    if (publish_clock_) {
      rosgraph_msgs::msg::Clock clk;
      clk.clock = rclcpp::Clock(RCL_SYSTEM_TIME).now();
      clock_pub_->publish(clk);
    }

    // tick is the robot controller's timestamp; unchanged means no new packet
    const auto &s = sdk_state_;
    if (s.tick == last_tick_) {
      if (!connected_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "No LowState from the Go1 yet (is it in low-level mode?)");
      }
      return;
    }
    last_tick_ = s.tick;
    if (!connected_) {
      RCLCPP_INFO(get_logger(), "Receiving LowState from the Go1");
      connected_ = true;
    }

    unitree_go::msg::LowState msg;
    msg.head = {s.head[0], s.head[1]};
    msg.level_flag = s.levelFlag;
    msg.frame_reserve = s.frameReserve;
    msg.sn = {s.SN[0], s.SN[1]};
    msg.version = {s.version[0], s.version[1]};
    msg.bandwidth = s.bandWidth;

    for (int i = 0; i < 4; ++i) {
      msg.imu_state.quaternion[i] = s.imu.quaternion[i];  // (w, x, y, z), same as the Go2
    }
    for (int i = 0; i < 3; ++i) {
      msg.imu_state.gyroscope[i] = s.imu.gyroscope[i];
      msg.imu_state.accelerometer[i] = s.imu.accelerometer[i];
      msg.imu_state.rpy[i] = s.imu.rpy[i];
    }
    msg.imu_state.temperature = s.imu.temperature;

    for (int i = 0; i < 20; ++i) {
      const auto &m = s.motorState[i];
      auto &o = msg.motor_state[i];
      o.mode = m.mode;
      o.q = m.q;
      o.dq = m.dq;
      o.ddq = m.ddq;
      o.tau_est = m.tauEst;
      o.q_raw = m.q_raw;
      o.dq_raw = m.dq_raw;
      o.ddq_raw = m.ddq_raw;
      o.temperature = m.temperature;
    }

    msg.bms_state.version_high = s.bms.version_h;
    msg.bms_state.version_low = s.bms.version_l;
    msg.bms_state.status = s.bms.bms_status;
    msg.bms_state.soc = s.bms.SOC;
    msg.bms_state.current = s.bms.current;
    msg.bms_state.cycle = s.bms.cycle;
    for (int i = 0; i < 2; ++i) {
      msg.bms_state.bq_ntc[i] = s.bms.BQ_NTC[i];
      msg.bms_state.mcu_ntc[i] = s.bms.MCU_NTC[i];
    }
    for (int i = 0; i < 10; ++i) {
      msg.bms_state.cell_vol[i] = s.bms.cell_vol[i];
    }

    for (int i = 0; i < 4; ++i) {
      msg.foot_force[i] = s.footForce[i];
      msg.foot_force_est[i] = s.footForceEst[i];
    }
    msg.tick = s.tick;
    std::copy(s.wirelessRemote.begin(), s.wirelessRemote.end(), msg.wireless_remote.begin());
    msg.crc = s.crc;
    lowstate_pub_->publish(msg);

    // The 40-byte wireless remote block has the same layout and key bit order
    // as the Go2's WirelessController message.
    xRockerBtnDataStruct remote;
    static_assert(sizeof(remote) <= 40, "wireless remote block");
    std::memcpy(&remote, s.wirelessRemote.data(), sizeof(remote));
    unitree_go::msg::WirelessController wc;
    wc.lx = remote.lx;
    wc.ly = remote.ly;
    wc.rx = remote.rx;
    wc.ry = remote.ry;
    wc.keys = remote.btn.value;
    wireless_pub_->publish(wc);
  }

  rclcpp::Publisher<unitree_go::msg::LowState>::SharedPtr lowstate_pub_;
  rclcpp::Publisher<unitree_go::msg::WirelessController>::SharedPtr wireless_pub_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Subscription<unitree_go::msg::LowCmd>::SharedPtr lowcmd_sub_;

  std::unique_ptr<sdk::UDP> udp_;
  std::unique_ptr<sdk::Safety> safety_;
  sdk::LowCmd sdk_cmd_ = {};
  sdk::LowState sdk_state_ = {};

  std::mutex cmd_mutex_;
  std::chrono::steady_clock::time_point last_cmd_time_;
  bool have_cmd_ = false;
  bool timed_out_ = false;
  uint32_t last_tick_ = 0;
  bool connected_ = false;

  double rate_;
  int power_protect_level_;
  double cmd_timeout_;
  double damping_kd_;
  bool publish_clock_;

  std::atomic<bool> running_{false};
  std::thread loop_thread_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go1Bridge>());
  rclcpp::shutdown();
  return 0;
}
