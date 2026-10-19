// Saved poses only: this executable never calls the registration backends.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <Eigen/Geometry>
#include <pcl/common/point_tests.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

using Clock = std::chrono::steady_clock;
using ColorCloud = pcl::PointCloud<pcl::PointXYZRGB>;
using MapCloud = pcl::PointCloud<pcl::PointXYZ>;

struct SavedFrame {
  int idx, original_idx;
  std::int64_t timestamp_ns;
  std::string file;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
};

class Terminal {
 public:
  Terminal() {
    active_ = isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &saved_) == 0;
    if (active_) {
      auto raw = saved_;
      raw.c_lflag &= ~(ICANON | ECHO);
      raw.c_cc[VMIN] = 0;
      raw.c_cc[VTIME] = 0;
      tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
  }
  ~Terminal() { if (active_) tcsetattr(STDIN_FILENO, TCSANOW, &saved_); }
  char Read() const {
    if (!active_) return 0;
    fd_set descriptors;
    FD_ZERO(&descriptors);
    FD_SET(STDIN_FILENO, &descriptors);
    timeval timeout{0, 0};
    char value = 0;
    if (select(STDIN_FILENO + 1, &descriptors, nullptr, nullptr, &timeout) > 0)
      if (read(STDIN_FILENO, &value, 1) != 1) return 0;
    return value;
  }
 private:
  bool active_ = false;
  termios saved_{};
};

class Replay : public rclcpp::Node {
 public:
  Replay() : Node("localization_replay") {
    const auto manifest = declare_parameter<std::string>("frame_manifest", "");
    const auto map_file = declare_parameter<std::string>("map_file", "");
    scale_ = declare_parameter<double>("ply_scale", 1.);
    input_leaf_ = declare_parameter<double>("input_voxel_leaf_size", .3);
    const auto map_leaf = declare_parameter<double>("map_voxel_leaf_size", 1.);
    half_xy_ = declare_parameter<double>("submap_half_size", 100.);
    half_z_ = declare_parameter<double>("submap_half_height", 30.);
    declare_parameter<double>("speed", 1.);
    declare_parameter<bool>("paused", false);
    declare_parameter<std::int64_t>("frame_index", 0);
    loop_ = declare_parameter<bool>("loop", false);
    exit_at_end_ = declare_parameter<bool>("exit_at_end", false);
    if (!std::isfinite(scale_) || scale_ <= 0 || input_leaf_ < 0 || map_leaf < 0 ||
        !std::isfinite(input_leaf_) || !std::isfinite(map_leaf) || half_xy_ <= 0 || half_z_ <= 0)
      throw std::runtime_error("Invalid replay scale/filter parameters");
    std::ifstream stream(manifest);
    std::string line;
    if (!std::getline(stream, line) || line != "idx\toriginal_idx\ttimestamp_ns\tply_file\tpose_3x4")
      throw std::runtime_error("Invalid replay manifest/header");
    while (std::getline(stream, line)) {
      std::istringstream row(line);
      std::string idx, original, stamp, matrix;
      SavedFrame frame;
      if (!std::getline(row, idx, '\t') || !std::getline(row, original, '\t') ||
          !std::getline(row, stamp, '\t') || !std::getline(row, frame.file, '\t') ||
          !std::getline(row, matrix)) throw std::runtime_error("Malformed replay row");
      frame.idx = std::stoi(idx);
      frame.original_idx = std::stoi(original);
      frame.timestamp_ns = std::stoll(stamp);
      std::istringstream values(matrix);
      for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c)
        if (!(values >> frame.pose(r, c))) throw std::runtime_error("Malformed saved pose");
      std::string extra;
      const Eigen::Matrix3d rotation = frame.pose.block<3, 3>(0, 0);
      if ((values >> extra) || !frame.pose.allFinite() ||
          (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-3 ||
          std::abs(rotation.determinant() - 1) > 1e-3 || frame.timestamp_ns <= 0 ||
          (!frames_.empty() && (frame.timestamp_ns <= frames_.back().timestamp_ns || frame.idx != frames_.back().idx + 1)))
        throw std::runtime_error("Invalid pose/order in replay manifest");
      frames_.push_back(frame);
    }
    if (frames_.empty()) throw std::runtime_error("No saved frames");
    if (get_parameter("frame_index").as_int() < 0 ||
        get_parameter("frame_index").as_int() >= static_cast<std::int64_t>(frames_.size()))
      throw std::runtime_error("frame_index outside selected replay window");
    const auto qos = rclcpp::QoS(1).reliable().transient_local();
    source_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/replay/source", qos);
    submap_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/replay/submap", qos);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/replay/pose", qos);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/replay/path", qos);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/replay/markers", qos);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    map_ = std::make_shared<MapCloud>();
    if (pcl::io::loadPCDFile(map_file, *map_) < 0 || map_->empty())
      throw std::runtime_error("Cannot load saved run's map");
    if (map_leaf > 0) {
      auto filtered = std::make_shared<MapCloud>();
      pcl::VoxelGrid<pcl::PointXYZ> voxel;
      voxel.setInputCloud(map_);
      voxel.setLeafSize(map_leaf, map_leaf, map_leaf);
      voxel.filter(*filtered);
      map_ = filtered;
    }
    callback_ = add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      for (const auto& p : parameters) {
        if (p.get_name() == "speed" && (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
            !std::isfinite(p.as_double()) || p.as_double() <= 0 || p.as_double() > 100)) {
          result.successful = false; result.reason = "speed must be in (0,100]";
        } else if (p.get_name() == "frame_index" && (p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER ||
            p.as_int() < 0 || p.as_int() >= static_cast<std::int64_t>(frames_.size()))) {
          result.successful = false; result.reason = "frame_index outside selected replay window";
        } else if (p.get_name() != "speed" && p.get_name() != "frame_index" && p.get_name() != "paused") {
          result.successful = false; result.reason = "Only speed/paused/frame_index can change during replay";
        }
      }
      return result;
    });
    if (!std::isfinite(get_parameter("speed").as_double()) || get_parameter("speed").as_double() <= 0 ||
        get_parameter("speed").as_double() > 100) throw std::runtime_error("Invalid replay speed");
    RCLCPP_INFO(get_logger(), "Saved-pose replay: %zu frames, %zu map points. No registration. Space=pause, n/b=step, +/-=speed, r=restart, q=quit.", frames_.size(), map_->size());
    next_ = Clock::now();
    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() { Tick(); });
  }

  bool failed() const { return failed_; }

 private:
  geometry_msgs::msg::PoseStamped Pose(std::size_t index, const rclcpp::Time& stamp) const {
    geometry_msgs::msg::PoseStamped message;
    message.header.frame_id = "map";
    message.header.stamp = stamp;
    const auto& matrix = frames_[index].pose;
    message.pose.position.x = matrix(0, 3);
    message.pose.position.y = matrix(1, 3);
    message.pose.position.z = matrix(2, 3);
    const Eigen::Quaterniond q = Eigen::Quaterniond(matrix.block<3, 3>(0, 0)).normalized();
    message.pose.orientation.x = q.x(); message.pose.orientation.y = q.y();
    message.pose.orientation.z = q.z(); message.pose.orientation.w = q.w();
    return message;
  }

  void LoadCloud(std::size_t index) {
    auto raw = std::make_shared<ColorCloud>();
    if (pcl::io::loadPLYFile(frames_[index].file, *raw) < 0 || raw->empty())
      throw std::runtime_error("Cannot read saved source PLY: " + frames_[index].file);
    auto finite = std::make_shared<ColorCloud>();
    const float scale = static_cast<float>(scale_);
    for (auto point : *raw) {
      point.x *= scale; point.y *= scale; point.z *= scale;
      if (pcl::isFinite(point)) finite->push_back(point);
    }
    if (finite->empty()) throw std::runtime_error("PLY has no finite scaled points");
    auto filtered = std::make_shared<ColorCloud>();
    if (input_leaf_ > 0) {
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
      voxel.setInputCloud(finite);
      voxel.setLeafSize(input_leaf_, input_leaf_, input_leaf_);
      voxel.filter(*filtered);
    } else filtered = finite;
    // Bake the saved pose into this frame's geometry. RViz must not wait for
    // an independently delivered dynamic TF before it can display the cloud.
    // Keep double precision for R*p+t; only the PointXYZRGB output is float.
    const auto& pose = frames_[index].pose;
    for (auto& point : *filtered) {
      const Eigen::Vector3d world = pose.block<3, 3>(0, 0) *
        Eigen::Vector3d(point.x, point.y, point.z) + pose.block<3, 1>(0, 3);
      point.x = static_cast<float>(world.x());
      point.y = static_cast<float>(world.y());
      point.z = static_cast<float>(world.z());
    }
    pcl::toROSMsg(*filtered, source_);
    source_.header.frame_id = "map";

    const auto t = frames_[index].pose.block<3, 1>(0, 3).cast<float>();
    MapCloud submap;
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(map_);
    crop.setMin(Eigen::Vector4f(t.x() - half_xy_, t.y() - half_xy_, t.z() - half_z_, 1));
    crop.setMax(Eigen::Vector4f(t.x() + half_xy_, t.y() + half_xy_, t.z() + half_z_, 1));
    crop.filter(submap);
    pcl::toROSMsg(submap, submap_);
    submap_.header.frame_id = "map";
    RCLCPP_INFO(get_logger(), "frame=%d original_idx=%d timestamp_ns=%lld XYZ=(%.3f, %.3f, %.3f)",
                frames_[index].idx, frames_[index].original_idx, static_cast<long long>(frames_[index].timestamp_ns), t.x(), t.y(), t.z());
  }

  void Publish(std::size_t index) {
    // Fresh, common ROS timestamps keep TF usable while seeking, looping or paused.
    // Capture nanoseconds remain exact in the manifest, HUD and coordinate CSV.
    const auto stamp = now();
    auto pose = Pose(index, stamp);
    geometry_msgs::msg::TransformStamped transform;
    transform.header = pose.header; transform.child_frame_id = "replay_source";
    transform.transform.translation.x = pose.pose.position.x;
    transform.transform.translation.y = pose.pose.position.y;
    transform.transform.translation.z = pose.pose.position.z;
    transform.transform.rotation = pose.pose.orientation;
    tf_->sendTransform(transform);
    source_.header.stamp = stamp; submap_.header.stamp = stamp;
    source_pub_->publish(source_); submap_pub_->publish(submap_); pose_pub_->publish(pose);
    nav_msgs::msg::Path path;
    path.header = pose.header;
    // Rebuild the prefix on seek/back/loop: no duplicate or future poses.
    for (std::size_t i = 0; i <= index; ++i) path.poses.push_back(Pose(i, stamp));
    path_pub_->publish(path);
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker text;
    text.header = pose.header; text.ns = "saved_pose"; text.id = 0;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position = pose.pose.position; text.pose.position.z += 5;
    text.pose.orientation.w = 1.; text.scale.z = .85;
    text.color.r = 1; text.color.g = 1; text.color.b = 1; text.color.a = 1;
    const auto& m = frames_[index].pose;
    const double pitch = std::atan2(-m(2, 0), std::hypot(m(0, 0), m(1, 0)));
    const bool singular = std::hypot(m(0, 0), m(1, 0)) <= 1e-8;
    const double roll = singular ? 0 : std::atan2(m(2, 1), m(2, 2));
    const double yaw = singular ? std::atan2(-m(0, 1), m(1, 1)) : std::atan2(m(1, 0), m(0, 0));
    constexpr double degrees = 180 / 3.14159265358979323846;
    std::ostringstream info;
    info << "Source pose in map | frame " << frames_[index].idx << " / original " << frames_[index].original_idx
         << "\nts_ns " << frames_[index].timestamp_ns << std::fixed << std::setprecision(3)
         << "\nXYZ [m] " << m(0, 3) << ", " << m(1, 3) << ", " << m(2, 3)
         << std::setprecision(2) << "\nRPY [deg] " << roll * degrees << ", " << pitch * degrees << ", " << yaw * degrees
         << "\n" << (get_parameter("paused").as_bool() ? "PAUSED" : "PLAY") << "  " << get_parameter("speed").as_double() << "x"
         << " | axes: X red, Y green, Z blue";
    text.text = info.str(); markers.markers.push_back(text);
    auto heading = text;
    heading.id = 1; heading.type = visualization_msgs::msg::Marker::ARROW;
    heading.pose = pose.pose;
    heading.pose.orientation.x = 0; heading.pose.orientation.y = 0;
    heading.pose.orientation.z = std::sin(yaw / 2); heading.pose.orientation.w = std::cos(yaw / 2);
    heading.scale.x = 4; heading.scale.y = .3; heading.scale.z = .5;
    heading.color.r = 1; heading.color.g = .7; heading.color.b = 0;
    markers.markers.push_back(heading);
    marker_pub_->publish(markers);
    last_publish_ = Clock::now();
  }

  void Tick() {
    try {
      auto paused = get_parameter("paused").as_bool();
      const char key = terminal_.Read();
      if (key == 'q') { rclcpp::shutdown(); return; }
      if (key == ' ' || key == 'p') set_parameter(rclcpp::Parameter("paused", !paused));
      if (key == '+' || key == '=') set_parameter(rclcpp::Parameter("speed", std::min(100., get_parameter("speed").as_double() * 2)));
      if (key == '-') set_parameter(rclcpp::Parameter("speed", std::max(.03125, get_parameter("speed").as_double() / 2)));
      if (key == 'n' || key == 'b' || key == 'r') {
        set_parameter(rclcpp::Parameter("paused", true));
        const auto current = get_parameter("frame_index").as_int();
        const auto target = key == 'r' ? 0 : std::clamp<std::int64_t>(current + (key == 'n' ? 1 : -1), 0, frames_.size() - 1);
        set_parameter(rclcpp::Parameter("frame_index", target));
      }
      paused = get_parameter("paused").as_bool();
      const double speed = get_parameter("speed").as_double();
      auto index = static_cast<std::size_t>(get_parameter("frame_index").as_int());
      const auto clock = Clock::now();
      const bool moved = loaded_ != static_cast<std::int64_t>(index);
      if (moved) { LoadCloud(index); loaded_ = index; Publish(index); }
      if (moved || paused != was_paused_ || speed != last_speed_) {
        const auto dt = index + 1 < frames_.size() ? (frames_[index + 1].timestamp_ns - frames_[index].timestamp_ns) * 1e-9 : .1;
        next_ = clock + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(dt / speed));
      } else if (!paused && clock >= next_) {
        if (index + 1 < frames_.size()) {
          set_parameter(rclcpp::Parameter("frame_index", static_cast<std::int64_t>(index + 1)));
        } else if (loop_) {
          loaded_ = -1;
          set_parameter(rclcpp::Parameter("frame_index", static_cast<std::int64_t>(0)));
        } else if (exit_at_end_) { rclcpp::shutdown(); return; }
        else { set_parameter(rclcpp::Parameter("paused", true)); }
      }
      if (!moved && clock - last_publish_ >= std::chrono::milliseconds(200)) Publish(index);
      was_paused_ = paused; last_speed_ = speed;
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "%s", error.what());
      failed_ = true; rclcpp::shutdown();
    }
  }

  Terminal terminal_;
  std::vector<SavedFrame> frames_;
  MapCloud::Ptr map_;
  sensor_msgs::msg::PointCloud2 source_, submap_;
  double scale_, input_leaf_, half_xy_, half_z_, last_speed_ = -1;
  bool loop_, exit_at_end_, was_paused_ = false, failed_ = false;
  std::int64_t loaded_ = -1;
  Clock::time_point next_, last_publish_{};
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr source_pub_, submap_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr callback_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<Replay>();
    rclcpp::spin(node);
    const bool failed = node->failed();
    if (rclcpp::ok()) rclcpp::shutdown();
    return failed ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "Replay failed: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
