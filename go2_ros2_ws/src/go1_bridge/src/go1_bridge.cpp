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
#include <std_msgs/msg/string.hpp>
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
    // false: receive-only, nothing is sent to the motors (first bench test)
    declare_parameter<bool>("go1.send_commands", true);
    // motor temperature [C]: warn above the first, hold the motors in damping above the second
    declare_parameter<double>("go1.motor_temp_warn", 60.0);
    declare_parameter<double>("go1.motor_temp_stop", 70.0);
    // battery state of charge [%] below which a warning is logged
    declare_parameter<double>("go1.battery_warn", 20.0);
    // |roll| or |pitch| [rad] above which the robot is considered fallen: motors go
    // to damping and stay there until the state machine is back in OFF/PRONE
    declare_parameter<double>("go1.abort_roll_pitch", 0.7);
    declare_parameter<std::string>("topics.robot_state", "/robot_state");

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
    send_commands_ = get_parameter("go1.send_commands").as_bool();
    motor_temp_warn_ = get_parameter("go1.motor_temp_warn").as_double();
    motor_temp_stop_ = get_parameter("go1.motor_temp_stop").as_double();
    battery_warn_ = get_parameter("go1.battery_warn").as_double();
    abort_roll_pitch_ = get_parameter("go1.abort_roll_pitch").as_double();
    robot_state_sub_ = create_subscription<std_msgs::msg::String>(
        get_parameter("topics.robot_state").as_string(), 1,
        [this](const std_msgs::msg::String::SharedPtr msg) {
          std::lock_guard<std::mutex> lock(cmd_mutex_);
          robot_state_ = msg->data;
        });
    if (!send_commands_) {
      RCLCPP_WARN(get_logger(), "go1.send_commands=false: receive-only, motors are not commanded");
    }

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
    // The control board only answers packets it receives, so even receive-only
    // mode must send something: the SDK's initial command (no position target,
    // zero gains, zero torque), which leaves the motors passive.
    passive_cmd_ = sdk_cmd_;
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

      const int received = udp_->Recv();
      if (received > 0) {
        ++packets_received_;
      }
      udp_->GetRecv(sdk_state_);
      publish_state();
      const bool overheated = check_health();

      if (!send_commands_) {
        sdk::LowCmd passive = passive_cmd_;
        udp_->SetSend(passive);
        udp_->Send();
        std::this_thread::sleep_until(next);
        continue;
      }

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
      if (overheated) {
        set_damping(cmd);
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

  // ---- motor temperature and battery ---------------------------------------
  // Returns true while any motor is above the stop temperature; the caller then
  // overrides the command with damping until the motor has cooled below the
  // warn temperature.
  bool check_health() {
    if (!connected_) {
      return false;
    }
    int hottest = -1;
    int hottest_temp = -128;
    for (int i = 0; i < kNumJoints; ++i) {
      const int t = sdk_state_.motorState[i].temperature;
      if (t > hottest_temp) {
        hottest_temp = t;
        hottest = i;
      }
    }
    if (hottest_temp >= motor_temp_stop_) {
      if (!overheated_) {
        RCLCPP_ERROR(get_logger(), "Motor %d at %d C (stop %.0f C): holding motors in damping mode",
                     hottest, hottest_temp, motor_temp_stop_);
      }
      overheated_ = true;
    } else if (overheated_ && hottest_temp < motor_temp_warn_) {
      RCLCPP_WARN(get_logger(), "Motors cooled to %d C: commands accepted again", hottest_temp);
      overheated_ = false;
    } else if (hottest_temp >= motor_temp_warn_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "Motor %d at %d C (warn %.0f C)",
                           hottest, hottest_temp, motor_temp_warn_);
    }

    const int soc = sdk_state_.bms.SOC;
    if (soc > 0 && soc <= battery_warn_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 30000, "Battery at %d%%", soc);
    }

    // fall detection: latch damping until the operator has put the state machine
    // back in OFF/PRONE, so a righted robot does not resume a walking command
    const double roll = sdk_state_.imu.rpy[0];
    const double pitch = sdk_state_.imu.rpy[1];
    const bool tilted = std::abs(roll) > abort_roll_pitch_ || std::abs(pitch) > abort_roll_pitch_;
    if (tilted && !fallen_) {
      RCLCPP_ERROR(get_logger(), "Roll %.2f / pitch %.2f rad beyond %.2f: damping mode until the state machine is OFF or PRONE",
                   roll, pitch, abort_roll_pitch_);
      fallen_ = true;
    } else if (fallen_ && !tilted) {
      std::string state;
      {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        state = robot_state_;
      }
      if (state == "OFF" || state == "PRONE" || state == "PRONING") {
        RCLCPP_WARN(get_logger(), "Upright again and state machine in %s: commands accepted again", state.c_str());
        fallen_ = false;
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Upright again; switch the state machine to prone to resume (now %s)", state.c_str());
      }
    }
    return overheated_ || fallen_;
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
                             "No LowState from the Go1 yet (is it in low-level mode?) "
                             "UDP packets received so far: %ld",
                             static_cast<long>(packets_received_));
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
  sdk::LowCmd passive_cmd_ = {};
  sdk::LowState sdk_state_ = {};
  long packets_received_ = 0;

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
  bool send_commands_;
  double motor_temp_warn_;
  double motor_temp_stop_;
  double battery_warn_;
  double abort_roll_pitch_;
  bool overheated_ = false;
  bool fallen_ = false;
  std::string robot_state_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_state_sub_;

  std::atomic<bool> running_{false};
  std::thread loop_thread_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go1Bridge>());
  rclcpp::shutdown();
  return 0;
}
