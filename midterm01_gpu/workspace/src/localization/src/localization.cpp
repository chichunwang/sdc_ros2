#include "localization/localization.h"
#include "localization/sensor_guided.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <stdexcept>
#include <limits>
#include <sstream>
#include <cuda_runtime_api.h>

#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace localization {

namespace {
using Clock = std::chrono::steady_clock;
double ElapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
}  // namespace

int LocalizationApp::Start(int argc, char** argv) {
    rclcpp::init(argc, argv);
    node_ = std::make_shared<rclcpp::Node>("image_point_cloud_localizer");

    if (!LoadParameters()) {
        Shutdown();
        return -1;
    }

    if (!SetupRegistration()) {
        Shutdown();
        return -1;
    }

    SetupRosInterfaces();

    const boost::filesystem::path output_path = boost::filesystem::path(save_path_) / save_file_;
    result_file_.open(output_path.string());
    result_file_.precision(9);
    if (!result_file_.is_open()) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to open result file: %s", output_path.string().c_str());
        Shutdown();
        return -1;
    }

    const auto map_start = Clock::now();
    if (!IndexPlyFiles(ply_path_) || !LoadMap()) {
        Shutdown();
        return -1;
    }
    RCLCPP_INFO(node_->get_logger(), "Startup map/index: %.3f s (includes PCD read and map voxel filtering).", ElapsedMs(map_start) / 1000.0);

    if (publish_global_map_) {
        PublishGlobalMap();
    }

    const bool success = RunLocalization();
    result_file_.close();

    if (success && keep_alive_ && rclcpp::ok()) {
        RCLCPP_INFO(node_->get_logger(), "Input processing finished. Keeping publishers alive for RViz (Ctrl-C to exit).");
        rclcpp::spin(node_);
    }

    Shutdown();
    return success ? 0 : -1;
}

// ** ================================================================== **
// ** ================ Data loading & Parameter setting ================ **
// ** ================================================================== **
bool LocalizationApp::LoadParameters() {
    node_->declare_parameter<std::string>("map_path", "");
    node_->declare_parameter<std::string>("map_file", "map.pcd");
    node_->declare_parameter<std::string>("map_cache_file", "");
    node_->declare_parameter<std::string>("bag_path", "");
    node_->declare_parameter<std::string>("ply_path", "");
    node_->declare_parameter<std::string>("init_guess_file", "");
    node_->declare_parameter<std::vector<double>>("initial_pose", {0.0, 0.0, 0.0, 0.0});
    node_->declare_parameter<std::string>("save_path", "");
    node_->declare_parameter<std::string>("save_file", "track_result.txt");

    node_->declare_parameter<double>("map_voxel_leaf_size", map_voxel_leaf_size_);
    node_->declare_parameter<double>("input_voxel_leaf_size", input_voxel_leaf_size_);
    node_->declare_parameter<double>("submap_half_size", submap_half_size_);
    node_->declare_parameter<double>("submap_half_height", submap_half_height_);
    node_->declare_parameter<int>("min_input_points", min_input_points_);
    

    node_->declare_parameter<double>("ndt_resolution", ndt_resolution_);
    node_->declare_parameter<double>("ndt_step_size", ndt_step_size_);
    node_->declare_parameter<double>("ndt_transformation_epsilon", ndt_transformation_epsilon_);
    node_->declare_parameter<int>("ndt_max_iterations", ndt_max_iterations_);
    node_->declare_parameter<std::string>("registration_backend", registration_backend_);
    node_->declare_parameter<std::string>("cuda_ndt_mode", cuda_ndt_mode_);
    node_->declare_parameter<double>("cuda_translation_epsilon", cuda_translation_epsilon_);
    node_->declare_parameter<double>("cuda_rotation_epsilon", cuda_rotation_epsilon_);
    node_->declare_parameter<double>("ply_scale", ply_scale_);

    node_->declare_parameter<bool>("keep_alive", keep_alive_);
    node_->declare_parameter<bool>("publish_global_map", publish_global_map_);
    node_->declare_parameter<int>("max_frames", max_frames_);
    node_->declare_parameter<std::string>("localization_profile", localization_profile_);
    node_->declare_parameter<std::string>("sensor_prior_file", sensor_prior_file_);
    node_->declare_parameter<double>("sensor_max_xy_correction", sensor_max_xy_correction_);
    node_->declare_parameter<double>("sensor_max_yaw_correction_deg", sensor_max_yaw_correction_deg_);
    node_->declare_parameter<double>("sensor_smoothing_alpha", sensor_smoothing_alpha_);
    node_->declare_parameter<std::string>("geometry_mode", geometry_mode_);
    node_->declare_parameter<double>("geometry_normal_gain", geometry_normal_gain_);
    node_->declare_parameter<double>("geometry_tangent_weight", geometry_tangent_weight_);


    map_path_ = node_->get_parameter("map_path").as_string();
    map_file_ = node_->get_parameter("map_file").as_string();
    map_cache_file_ = node_->get_parameter("map_cache_file").as_string();
    bag_path_ = node_->get_parameter("bag_path").as_string();
    ply_path_ = node_->get_parameter("ply_path").as_string();
    init_guess_file_ = node_->get_parameter("init_guess_file").as_string();
    initial_pose_ = node_->get_parameter("initial_pose").as_double_array();
    save_path_ = node_->get_parameter("save_path").as_string();
    save_file_ = node_->get_parameter("save_file").as_string();

    map_voxel_leaf_size_ = node_->get_parameter("map_voxel_leaf_size").as_double();
    input_voxel_leaf_size_ = node_->get_parameter("input_voxel_leaf_size").as_double();
    submap_half_size_ = node_->get_parameter("submap_half_size").as_double();
    submap_half_height_ = node_->get_parameter("submap_half_height").as_double();
    min_input_points_ = node_->get_parameter("min_input_points").as_int();
    ndt_max_iterations_ = node_->get_parameter("ndt_max_iterations").as_int();
    registration_backend_ = node_->get_parameter("registration_backend").as_string();
    cuda_ndt_mode_ = node_->get_parameter("cuda_ndt_mode").as_string();
    cuda_translation_epsilon_ = node_->get_parameter("cuda_translation_epsilon").as_double();
    cuda_rotation_epsilon_ = node_->get_parameter("cuda_rotation_epsilon").as_double();

    ndt_resolution_ = node_->get_parameter("ndt_resolution").as_double();
    ndt_step_size_ = node_->get_parameter("ndt_step_size").as_double();
    ndt_transformation_epsilon_ = node_->get_parameter("ndt_transformation_epsilon").as_double();
    ply_scale_ = node_->get_parameter("ply_scale").as_double();

    keep_alive_ = node_->get_parameter("keep_alive").as_bool();
    publish_global_map_ = node_->get_parameter("publish_global_map").as_bool();
    max_frames_ = node_->get_parameter("max_frames").as_int();
    localization_profile_ = node_->get_parameter("localization_profile").as_string();
    sensor_prior_file_ = node_->get_parameter("sensor_prior_file").as_string();
    sensor_max_xy_correction_ = node_->get_parameter("sensor_max_xy_correction").as_double();
    sensor_max_yaw_correction_deg_ = node_->get_parameter("sensor_max_yaw_correction_deg").as_double();
    sensor_smoothing_alpha_ = node_->get_parameter("sensor_smoothing_alpha").as_double();
    geometry_mode_ = node_->get_parameter("geometry_mode").as_string();
    geometry_normal_gain_ = node_->get_parameter("geometry_normal_gain").as_double();
    geometry_tangent_weight_ = node_->get_parameter("geometry_tangent_weight").as_double();
    if ((geometry_mode_ != "off" && geometry_mode_ != "diagnostic" && geometry_mode_ != "directional") ||
        (geometry_mode_ != "off" && (localization_profile_ != "sensor_guided" || registration_backend_ != "cuda_robust_icp")) ||
        !std::isfinite(geometry_normal_gain_) || geometry_normal_gain_ <= 0 || geometry_normal_gain_ > 10 ||
        !std::isfinite(geometry_tangent_weight_) || geometry_tangent_weight_ < 0 || geometry_tangent_weight_ > 1) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid geometry mode/weights; geometry requires sensor_guided/cuda_robust_icp.");
        return false;
    }
    if ((localization_profile_ != "legacy" && localization_profile_ != "sensor_guided") ||
        (localization_profile_ == "sensor_guided" && sensor_prior_file_.empty()) ||
        !std::isfinite(sensor_max_xy_correction_) || sensor_max_xy_correction_ <= 0 ||
        !std::isfinite(sensor_max_yaw_correction_deg_) || sensor_max_yaw_correction_deg_ <= 0 ||
        !std::isfinite(sensor_smoothing_alpha_) || sensor_smoothing_alpha_ <= 0 || sensor_smoothing_alpha_ > 1) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid localization_profile or sensor-guidance parameters.");
        return false;
    }


    if (ply_path_.empty() || map_path_.empty() || save_path_.empty() || save_file_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "PLY, map and output paths are required.");
        return false;
    }
    if (init_guess_file_.empty() &&
        (initial_pose_.size() != 4 || !std::all_of(initial_pose_.begin(), initial_pose_.end(), [](double value) { return std::isfinite(value); }))) {
        RCLCPP_ERROR(node_->get_logger(), "initial_pose must contain finite [x, y, z, yaw].");
        return false;
    }
    if ((registration_backend_ != "cuda_ndt" && registration_backend_ != "pcl_ndt" && registration_backend_ != "cuda_robust_icp") ||
        (cuda_ndt_mode_ != "P2D" && cuda_ndt_mode_ != "D2D")) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid registration_backend or cuda_ndt_mode.");
        return false;
    }
    if (registration_backend_ == "cuda_robust_icp" && localization_profile_ != "sensor_guided") {
        RCLCPP_ERROR(node_->get_logger(), "cuda_robust_icp requires localization_profile=sensor_guided.");
        return false;
    }
    const std::vector<double> positive = {ply_scale_, submap_half_size_, submap_half_height_, ndt_resolution_, ndt_step_size_, ndt_transformation_epsilon_, cuda_translation_epsilon_, cuda_rotation_epsilon_};
    if (!std::all_of(positive.begin(), positive.end(), [](double value) { return std::isfinite(value) && value > 0.0; }) ||
        !std::isfinite(map_voxel_leaf_size_) || map_voxel_leaf_size_ < 0.0 ||
        !std::isfinite(input_voxel_leaf_size_) || input_voxel_leaf_size_ < 0.0 ||
        max_frames_ < 0 || min_input_points_ < 1 || ndt_max_iterations_ < 1) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid filtering or NDT parameters.");
        return false;
    }
    boost::filesystem::create_directories(save_path_);
    return true;
}

void LocalizationApp::SetupRosInterfaces() {
    const auto cloud_qos =rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    input_cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/image_point_cloud", cloud_qos);
    submap_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/image_point_cloud_submap", cloud_qos);

    const auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    map_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/map_pc", map_qos);
    pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("/transformed_image_point_cloud_pose", rclcpp::QoS(1));
    path_pub_ = node_->create_publisher<nav_msgs::msg::Path>("/localization_path", rclcpp::QoS(1));
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
}

bool LocalizationApp::LoadMap() {
    const boost::filesystem::path map_directory(map_path_);
    const boost::filesystem::path source_path = map_directory / map_file_;
    boost::filesystem::path cache_path(map_cache_file_);
    if (!cache_path.empty() && !cache_path.is_absolute()) cache_path = map_directory / cache_path;
    const bool has_cache = !cache_path.empty() && boost::filesystem::exists(cache_path);
    const boost::filesystem::path selected_path = has_cache ? cache_path : source_path;

    map_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    RCLCPP_INFO(node_->get_logger(), "Loading map: %s", selected_path.string().c_str());
    if (pcl::io::loadPCDFile(selected_path.string(), *map_cloud_) == -1) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to read map: %s", selected_path.string().c_str());
        return false;
    }

    if (!has_cache && map_voxel_leaf_size_ > 0.0) {
        const std::size_t original_size = map_cloud_->size();
        auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(map_cloud_);
        const float leaf = static_cast<float>(map_voxel_leaf_size_);
        voxel.setLeafSize(leaf, leaf, leaf);
        voxel.filter(*filtered);
        map_cloud_ = filtered;
        RCLCPP_INFO(node_->get_logger(), "Downsampled map from %zu to %zu points (leaf %.2f m).", original_size, map_cloud_->size(), map_voxel_leaf_size_);

        if (!cache_path.empty()) {
            try {
                boost::filesystem::create_directories(cache_path.parent_path());
                if (pcl::io::savePCDFileBinary(cache_path.string(), *map_cloud_) != 0)
                    throw std::runtime_error("PCD writer returned a failure");
                RCLCPP_INFO(node_->get_logger(), "Saved map cache: %s", cache_path.string().c_str());
            } catch (const std::exception& error) {
                RCLCPP_ERROR(node_->get_logger(), "Cannot save requested map cache %s: %s", cache_path.string().c_str(), error.what());
                return false;
            }
        }
    } else {
        RCLCPP_INFO(node_->get_logger(), "Loaded %zu map points%s.", map_cloud_->size(), has_cache ? " from cache" : "");
    }
    return !map_cloud_->empty();
}

bool LocalizationApp::OpenDB3File() {
    bag_reader_.reset();
    next_db3_message_.reset();
    const std::string gnss_topic = "/localizer_node/gnss_pose";
    const std::string imu_topic = "/imu/data";
    if (!bag_path_.empty()) {
        bag_reader_ = std::make_unique<rosbag2_cpp::Reader>();
        try {
            rosbag2_storage::StorageOptions options;
            options.uri = bag_path_;
            options.storage_id = "sqlite3";
            rosbag2_cpp::ConverterOptions converter;
            converter.input_serialization_format = "cdr";
            converter.output_serialization_format = "cdr";
            bag_reader_->open(options, converter);
            const auto topics = bag_reader_->get_all_topics_and_types();
            const auto topic = std::find_if(topics.begin(), topics.end(), [&](const auto& item) { return item.name == gnss_topic; });
            if (topic == topics.end() || topic->type != "geometry_msgs/msg/PoseStamped") {
                RCLCPP_ERROR(node_->get_logger(), "GNSS topic '%s' must exist with type geometry_msgs/msg/PoseStamped.", gnss_topic.c_str());
                return false;
            }
            const auto imu = std::find_if(topics.begin(), topics.end(), [&](const auto& item) { return item.name == imu_topic; });
            if (imu == topics.end() || imu->type != "sensor_msgs/msg/Imu") {
                RCLCPP_ERROR(node_->get_logger(), "IMU topic '%s' must exist with type sensor_msgs/msg/Imu.", imu_topic.c_str());
                return false;
            }
            rosbag2_storage::StorageFilter filter;
            filter.topics = {gnss_topic, imu_topic};
            bag_reader_->set_filter(filter);
            if (bag_reader_->has_next()) next_db3_message_ = bag_reader_->read_next();
        } catch (const std::exception& error) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open sensor bag: %s", error.what());
            return false;
        }
    }

    return true;
}

bool LocalizationApp::LoadDB3Frame(std::int64_t timestamp_ns, DB3Frame& frame) {
    const std::string gnss_topic = "/localizer_node/gnss_pose";
    const std::string imu_topic = "/imu/data";
    rclcpp::Serialization<geometry_msgs::msg::PoseStamped> gnss_serialization;
    rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
    // Compare exact integer bag timestamps against the PLY filename ROS ns.
    // Consume both information streams up to this frame, without future measurements.
    try {
        while (next_db3_message_ && next_db3_message_->time_stamp <= timestamp_ns) {
            rclcpp::SerializedMessage serialized(*next_db3_message_->serialized_data);
            if (next_db3_message_->topic_name == gnss_topic) {
                gnss_serialization.deserialize_message(&serialized, &frame.gnss_pose);
                frame.gnss_bag_timestamp_ns = next_db3_message_->time_stamp;
                frame.has_gnss_pose = true;
            } else if (next_db3_message_->topic_name == imu_topic) {
                imu_serialization.deserialize_message(&serialized, &frame.imu_data);
                frame.imu_bag_timestamp_ns = next_db3_message_->time_stamp;
                frame.has_imu_data = true;
            }
            next_db3_message_ = bag_reader_->has_next() ? bag_reader_->read_next() : nullptr;
        }
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to read sensor message: %s", error.what());
        return false;
    }
    if (frame.has_gnss_pose) {
        RCLCPP_DEBUG(node_->get_logger(), "GNSS bag_time=%lld, frame_id=%s, position=(%.3f, %.3f, %.3f)",
                     static_cast<long long>(frame.gnss_bag_timestamp_ns), frame.gnss_pose.header.frame_id.c_str(),
                     frame.gnss_pose.pose.position.x, frame.gnss_pose.pose.position.y, frame.gnss_pose.pose.position.z);
    }
    if (frame.has_imu_data) {
        RCLCPP_DEBUG(node_->get_logger(), "IMU bag_time=%lld, angular_velocity=(%.6f, %.6f, %.6f), linear_acceleration=(%.6f, %.6f, %.6f)",
                     static_cast<long long>(frame.imu_bag_timestamp_ns),
                     frame.imu_data.angular_velocity.x, frame.imu_data.angular_velocity.y, frame.imu_data.angular_velocity.z,
                     frame.imu_data.linear_acceleration.x, frame.imu_data.linear_acceleration.y, frame.imu_data.linear_acceleration.z);
    }
    return true;
}

bool LocalizationApp::IndexPlyFiles(const std::string& directory_path) {
    ply_frames_.clear();
    const boost::filesystem::path directory(directory_path);
    if (!boost::filesystem::exists(directory) || !boost::filesystem::is_directory(directory)) {
        RCLCPP_ERROR(node_->get_logger(), "Image PLY directory does not exist: %s", directory.string().c_str());
        return false;
    }

    for (const auto& entry : boost::filesystem::directory_iterator(directory)) {
        if (!boost::filesystem::is_regular_file(entry.path())) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](const unsigned char value) 
                       {return static_cast<char>(std::tolower(value));});
        if (extension == ".ply") {
            std::int64_t timestamp_ns = 0;
            if (!TimestampNsFromPlyFilename(entry.path(), timestamp_ns)) {
                RCLCPP_WARN(node_->get_logger(), "Ignoring PLY without a valid _ts_<nanoseconds> timestamp: %s", entry.path().string().c_str());
                continue;
            }
            ply_frames_.push_back({timestamp_ns, entry.path().string()});
        }
    }
    std::sort(ply_frames_.begin(), ply_frames_.end(), [](const PlyFrameInfo& lhs, const PlyFrameInfo& rhs) 
              {return lhs.timestamp_ns < rhs.timestamp_ns;});

    if (ply_frames_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "No timestamped image point-cloud .ply files found in: %s", directory.string().c_str());
        return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Indexed %zu PLY filenames without loading point data (first=%lld, last=%lld).",
                ply_frames_.size(),
                static_cast<long long>(ply_frames_.front().timestamp_ns),
                static_cast<long long>(ply_frames_.back().timestamp_ns));
    return true;
}

bool LocalizationApp::LoadPlyFrame(const PlyFrameInfo& frame,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color_cloud) const {
    const boost::filesystem::path file(frame.path);
    if (!boost::filesystem::exists(file) || !boost::filesystem::is_regular_file(file) || boost::filesystem::file_size(file) == 0) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid image PLY file: %s", frame.path.c_str());
        return false;
    }

    pcl::PCLPointCloud2 packed_cloud;
    if (pcl::io::loadPLYFile(frame.path, packed_cloud) < 0) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to read image PLY file: %s", frame.path.c_str());
        return false;
    }

    cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromPCLPointCloud2(packed_cloud, *cloud);
    if (cloud->empty()) {
        RCLCPP_ERROR(node_->get_logger(), "Image PLY contains no points: %s", frame.path.c_str());
        return false;
    }

    color_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    pcl::fromPCLPointCloud2(packed_cloud, *color_cloud);
    if (color_cloud->size() != cloud->size()) {
        RCLCPP_ERROR(node_->get_logger(), "Image PLY XYZ/RGB point counts differ: %s", frame.path.c_str());
        return false;
    }
    return true;
}

bool LocalizationApp::LoadInitialPose(Eigen::Matrix4f& pose) const {
    pose = Eigen::Matrix4f::Identity();
    if (init_guess_file_.empty()) {
        const float x = static_cast<float>(initial_pose_[0]);
        const float y = static_cast<float>(initial_pose_[1]);
        const float z = static_cast<float>(initial_pose_[2]);
        const float yaw = static_cast<float>(initial_pose_[3]);
        pose(0, 0) = std::cos(yaw);
        pose(0, 1) = -std::sin(yaw);
        pose(1, 0) = std::sin(yaw);
        pose(1, 1) = std::cos(yaw);
        pose(0, 3) = x;
        pose(1, 3) = y;
        pose(2, 3) = z;
    } else {
        std::ifstream input(init_guess_file_);
        if (!input.is_open()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open init guess: %s", init_guess_file_.c_str());
            return false;
        }

        std::string line;
        for (int row = 0; row < 4; ++row) {
            if (!std::getline(input, line)) {
                RCLCPP_ERROR(node_->get_logger(), "Init guess must contain four rows.");
                return false;
            }
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream values(line);
            for (int column = 0; column < 4; ++column) {
                if (!(values >> pose(row, column))) {
                    RCLCPP_ERROR(node_->get_logger(), "Invalid value in init guess row %d.", row + 1);
                    return false;
                }
            }
        }
    }

    if (!IsFinite(pose)) {
        RCLCPP_ERROR(node_->get_logger(), "Initial pose contains NaN or Inf.");
        return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Initial pose: x=%.3f, y=%.3f, z=%.3f, yaw=%.6f",
                pose(0, 3), pose(1, 3), pose(2, 3), std::atan2(pose(1, 0), pose(0, 0)));
    return true;
}

// ** ================================================================== **
// ** ==================== input data/map processing =================== **
// ** ================================================================== **
pcl::PointCloud<pcl::PointXYZ>::Ptr LocalizationApp::ScalePlyPointCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input, const double scale) const {
    auto scaled = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    scaled->points.reserve(input->size());
    const float factor = static_cast<float>(scale);
    for (const auto& point : input->points) {
        pcl::PointXYZ output;
        output.x = factor * point.x;
        output.y = factor * point.y;
        output.z = factor * point.z;
        scaled->points.push_back(output);
    }
    scaled->width = static_cast<std::uint32_t>(scaled->size());
    scaled->height = 1;
    scaled->is_dense = input->is_dense;
    return scaled;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr LocalizationApp::ScalePlyColorCloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& input, const double scale) const {
    auto scaled = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    scaled->points.reserve(input->size());
    const float factor = static_cast<float>(scale);

    for (const auto& point : input->points) {
        pcl::PointXYZRGB output = point;
        output.x = factor * point.x;
        output.y = factor * point.y;
        output.z = factor * point.z;
        scaled->points.push_back(output);
    }

    scaled->width = static_cast<std::uint32_t>(scaled->size());
    scaled->height = 1;
    scaled->is_dense = input->is_dense;
    return scaled;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LocalizationApp::FilterInputCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input) const {
    auto finite = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    finite->points.reserve(input->size());

    for (const auto& point : input->points) {
        if (pcl::isFinite(point)) finite->points.push_back(point);
    }
    finite->width = static_cast<std::uint32_t>(finite->size());
    finite->height = 1;
    finite->is_dense = true;

    if (input_voxel_leaf_size_ <= 0.0 || finite->empty()) return finite;

    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(finite);
    const float leaf = static_cast<float>(input_voxel_leaf_size_);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*filtered);
    return filtered;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr LocalizationApp::FilterColorCloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& input) const {
    auto finite = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    finite->points.reserve(input->size());

    for (const auto& point : input->points) {
        if (pcl::isFinite(point)) finite->points.push_back(point);
    }
    finite->width = static_cast<std::uint32_t>(finite->size());
    finite->height = 1;
    finite->is_dense = true;

    if (input_voxel_leaf_size_ <= 0.0 || finite->empty()) return finite;

    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
    voxel.setInputCloud(finite);
    const float leaf = static_cast<float>(input_voxel_leaf_size_);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*filtered);
    return filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LocalizationApp::CropSubmap(const Eigen::Matrix4f& center_pose) const {
    auto submap = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    const float x = center_pose(0, 3);
    const float y = center_pose(1, 3);
    const float z = center_pose(2, 3);
    const float half_xy = static_cast<float>(submap_half_size_);
    const float half_z = static_cast<float>(submap_half_height_);

    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(map_cloud_);
    crop.setMin(Eigen::Vector4f(x - half_xy, y - half_xy, z - half_z, 1.0f));
    crop.setMax(Eigen::Vector4f(x + half_xy, y + half_xy, z + half_z, 1.0f));
    crop.filter(*submap);
    return submap;
}

// ** ================================================================== **
// ** =========================== Localization ========================= **
// ** ================================================================== **
bool LocalizationApp::SetupRegistration() {
    if (registration_backend_ == "pcl_ndt") {
        RCLCPP_INFO(node_->get_logger(), "Registration backend: pcl_ndt (CPU, explicitly selected).");
        return true;
    }
    try {
        int device_count = 0;
        CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count < 1) throw std::runtime_error("No CUDA device; refusing CPU fallback.");
        CheckCuda(cudaSetDevice(0), "cudaSetDevice");
        CheckCuda(cudaFree(nullptr), "CUDA context initialization");
        cudaDeviceProp device{};
        CheckCuda(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
        if (registration_backend_ == "cuda_robust_icp") {
            RCLCPP_INFO(node_->get_logger(), "Registration backend: cuda_robust_icp on %s; CUDA exact nearest-neighbor search, CPU robust prior-constrained 4x4 solve. CPU fallback disabled.", device.name);
            return true;
        }
        cuda_ndt_ = std::make_unique<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>>();
        cuda_ndt_->setResolution(ndt_resolution_);
        cuda_ndt_->setDistanceMode(cuda_ndt_mode_ == "P2D" ? fast_gicp::NDTDistanceMode::P2D : fast_gicp::NDTDistanceMode::D2D);
        cuda_ndt_->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
        cuda_ndt_->setTransformationEpsilon(cuda_translation_epsilon_);
        cuda_ndt_->setRotationEpsilon(cuda_rotation_epsilon_);
        cuda_ndt_->setMaximumIterations(ndt_max_iterations_);
        RCLCPP_INFO(node_->get_logger(), "Registration backend: cuda_ndt %s/DIRECT7 on %s (CC %d.%d); translation epsilon %.4f, rotation epsilon %.4f. CPU fallback disabled.",
                    cuda_ndt_mode_.c_str(), device.name, device.major, device.minor, cuda_translation_epsilon_, cuda_rotation_epsilon_);
        RCLCPP_INFO(node_->get_logger(), "CUDA uses LM; ndt_step_size and ndt_transformation_epsilon apply only to pcl_ndt. PCL fitness remains CPU.");
        return true;
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node_->get_logger(), "CUDA NDT initialization failed: %s", error.what());
        return false;
    }
}

bool LocalizationApp::LoadSensorPriors() {
    std::ifstream stream(sensor_prior_file_);
    std::string line;
    const std::size_t expected = max_frames_ > 0 ? std::min(ply_frames_.size(), static_cast<std::size_t>(max_frames_)) : ply_frames_.size();
    if (!std::getline(stream, line) || line.find("timestamp_ns,x,y,z,qx,qy,qz,qw") != 0) {
        RCLCPP_ERROR(node_->get_logger(), "Cannot read sensor prior CSV/header: %s", sensor_prior_file_.c_str());
        return false;
    }
    sensor_priors_.clear();
    while (std::getline(stream, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        std::int64_t stamp;
        double x, y, z, qx, qy, qz, qw;
        if (!(row >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw) || sensor_priors_.size() >= expected ||
            stamp != ply_frames_[sensor_priors_.size()].timestamp_ns) {
            RCLCPP_ERROR(node_->get_logger(), "Invalid or mismatched sensor prior row %zu.", sensor_priors_.size());
            return false;
        }
        Eigen::Quaterniond quaternion(qw, qx, qy, qz);
        if (!quaternion.coeffs().allFinite() || std::abs(quaternion.norm() - 1) > 0.02) return false;
        Eigen::Matrix4f prior = Eigen::Matrix4f::Identity();
        prior.block<3, 3>(0, 0) = quaternion.normalized().toRotationMatrix().cast<float>();
        prior.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z).cast<float>();
        if (!IsFinite(prior)) return false;
        sensor_priors_.push_back(prior);
    }
    if (sensor_priors_.size() != expected) {
        RCLCPP_ERROR(node_->get_logger(), "Expected %zu sensor priors, got %zu.", expected, sensor_priors_.size());
        return false;
    }
    RCLCPP_INFO(node_->get_logger(), "sensor_guided: %zu synchronized GNSS/IMU priors; bounded XY/yaw robust 4-DOF map refinement. Offline input; GNSS is not ground truth.", sensor_priors_.size());
    return true;
}

bool LocalizationApp::RunLocalization() {
    Eigen::Matrix4f pose;
    if (!LoadInitialPose(pose)) return false;

    const bool guided = localization_profile_ == "sensor_guided";
    if (guided ? !LoadSensorPriors() : !OpenDB3File()) return false;
    DB3Frame db3_frame;
    Eigen::Vector4d correction = Eigen::Vector4d::Zero();
    GeometryConfig geometry_config;
    geometry_config.mode = geometry_mode_ == "directional" ? GeometryMode::Directional :
        (geometry_mode_ == "diagnostic" ? GeometryMode::Diagnostic : GeometryMode::Off);
    geometry_config.normal_gain = geometry_normal_gain_;
    geometry_config.tangent_weight = geometry_tangent_weight_;
    std::ofstream geometry_csv;
    if (geometry_config.enabled()) {
        geometry_csv.open((boost::filesystem::path(save_path_) / "geometry.csv").string());
        if (!geometry_csv) return false;
        geometry_csv.precision(12);
        WriteGeometryCsvHeader(geometry_csv);
        std::ofstream config_stream((boost::filesystem::path(save_path_) / "geometry_config.json").string());
        if (!config_stream) return false;
        config_stream.precision(15);
        WriteGeometryConfigJson(config_stream, geometry_config);
        RCLCPP_INFO(node_->get_logger(), "Geometry %s: CPU map-neighborhood PCA; CUDA registration NN retained. Normal gain %.3f, tangent weight %.3f; data-only scaled information is not covariance.",
                    geometry_mode_.c_str(), geometry_normal_gain_, geometry_tangent_weight_);
    }
    std::ofstream guidance;
    if (guided) {
        guidance.open((boost::filesystem::path(save_path_) / "guidance.csv").string());
        if (!guidance) return false;
        guidance.precision(12);
        guidance << "idx,timestamp_ns,prior_x,prior_y,prior_z,prior_yaw,dx,dy,dz,dyaw,cuda_converged,cuda_seed_used,map_accepted,inliers,iterations,prior_cost,final_cost,prior_clipped_rmse,final_clipped_rmse,update_status,candidate_inliers,candidate_reliable_points,candidate_unique_targets\n";
    }

    std::size_t processed = 0;
    std::size_t failed = 0;
    std::size_t geometry_fallbacks = 0;
    std::ofstream timings((boost::filesystem::path(save_path_) / "timings.csv").string());
    if (!timings) {
        RCLCPP_ERROR(node_->get_logger(), "Cannot open timings.csv.");
        return false;
    }
    timings << "idx,timestamp_ns,backend,input_points,submap_points,load_ms,filter_ms,crop_ms,ndt_ms,fitness_ms,publish_ms,total_ms,converged,score\n";
    timings.precision(10);
    double ndt_sum_ms = 0.0, fitness_sum_ms = 0.0, frames_sum_ms = 0.0;
    for (const auto& frame : ply_frames_) {
        if (!rclcpp::ok()) break;
        if (max_frames_ > 0 && processed >= static_cast<std::size_t>(max_frames_)) break;
        const auto frame_start = Clock::now();
        if (!guided && !LoadDB3Frame(frame.timestamp_ns, db3_frame)) return false;
        Eigen::Matrix4f prior = pose;
        if (guided) {
            prior = sensor_priors_.at(processed);
            // GNSS altitude is constant in these bags. Carry map-refined height,
            // while anchoring XY and orientation to the current sensor observation.
            if (processed > 0) prior(2, 3) = pose(2, 3);
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_color;
        if (!LoadPlyFrame(frame, raw_cloud, raw_color)) return false;
        const double load_ms = ElapsedMs(frame_start);
        auto phase_start = Clock::now();
        const auto input = FilterInputCloud(ScalePlyPointCloud(raw_cloud, ply_scale_));
        const double filter_ms = ElapsedMs(phase_start);
        phase_start = Clock::now();
        const auto submap = CropSubmap(prior);
        const double crop_ms = ElapsedMs(phase_start);
        if (submap->empty()) {
            RCLCPP_ERROR(node_->get_logger(), "Empty submap at frame %zu.", processed);
            return false;
        }

        // ================= TODO：Localization =================
        // 請在此實作自己的定位演算法；目前提供基礎 NDT 作為起點。
        // ======================================================
        // 可用輸入：input 為已縮放與降採樣的 image point clouds，
        // 　　　　　submap 為局部地圖， 
        // 　　　　　pose 為上一幀接受的姿態（第一幀使用初始姿態）。
        // 感測器資料：db3_frame.gnss_pose、db3_frame.imu_data；
        // 　　　　　　使用前請檢查 has_gnss_pose、has_imu_data，並確認時間與座標系是否適用。
        // 預期輸出：candidate 為點雲座標系到 map 座標系的 4x4 姿態矩陣， score 為配準評分，converged 表示本幀定位是否成功。
        // 可替換 AlignNDT() 或加入 GNSS / IMU 融合與自訂成功判定。
        Eigen::Matrix4f candidate = pose;
        double score = std::numeric_limits<double>::infinity();
        bool converged = false;
        GuidedMetrics metrics;
        try {
            if (registration_backend_ == "cuda_robust_icp") {
                candidate = prior;
                last_ndt_ms_ = last_fitness_ms_ = 0.0;
            } else {
                converged = AlignNDT(input, submap, prior, candidate, score);
            }
            if (guided) {
                const bool cuda_converged = converged;
                const auto refine_start = Clock::now();
                Eigen::Vector4d next_correction;
                converged = RefineSensorGuided(input, submap, prior, candidate, cuda_converged,
                    correction, sensor_smoothing_alpha_, sensor_max_xy_correction_,
                    sensor_max_yaw_correction_deg_ * M_PI / 180.0, candidate, next_correction, metrics,
                    registration_backend_ == "cuda_robust_icp", geometry_config);
                last_ndt_ms_ += ElapsedMs(refine_start);
                correction = next_correction;
                score = metrics.final_rmse * metrics.final_rmse;
                guidance << processed << ',' << frame.timestamp_ns << ',' << prior(0, 3) << ',' << prior(1, 3) << ',' << prior(2, 3)
                    << ',' << std::atan2(prior(1, 0), prior(0, 0)) << ',' << correction[0] << ',' << correction[1] << ',' << correction[2] << ',' << correction[3]
                    << ',' << cuda_converged << ',' << metrics.used_cuda_seed << ',' << converged << ',' << metrics.inliers << ',' << metrics.iterations
                    << ',' << metrics.prior_cost << ',' << metrics.final_cost << ',' << metrics.prior_rmse << ',' << metrics.final_rmse
                    << ',' << GuidedUpdateStatusName(metrics.update_status) << ',' << metrics.candidate_inliers
                    << ',' << metrics.candidate_reliable_points << ',' << metrics.candidate_unique_targets << '\n';
                guidance.flush();
                if (geometry_config.enabled()) {
                    WriteGeometryCsvRow(geometry_csv, processed, frame.timestamp_ns, converged, metrics.geometry);
                    geometry_csv.flush();
                }
                // A rejected scan update uses this frame's sensor prior, never a
                // stationary previous pose which would accumulate motion error.
                pose = candidate;
            }
        } catch (const std::exception& error) {
            RCLCPP_ERROR(node_->get_logger(), "Registration runtime error at frame %zu: %s", processed, error.what());
            return false;
        }
        if (converged) {
            pose = candidate;
        } else {
            ++failed;
            if (guided && IsGeometrySupportFallback(metrics.update_status)) {
                ++geometry_fallbacks;
                RCLCPP_INFO(node_->get_logger(), "Map update skipped at frame %zu: %s (candidate reliable_points=%d/10, unique_targets=%d/6); using current sensor prior.",
                            processed, GuidedUpdateStatusName(metrics.update_status), metrics.candidate_reliable_points, metrics.candidate_unique_targets);
            } else if (guided) {
                RCLCPP_WARN(node_->get_logger(), "Map update rejected at frame %zu: %s (candidate inliers=%d); using current sensor prior.",
                            processed, GuidedUpdateStatusName(metrics.update_status), metrics.candidate_inliers);
            } else {
                RCLCPP_WARN(node_->get_logger(), "%s scan update failed at frame %zu; using previous pose.", registration_backend_.c_str(), processed);
            }
        }
        // ======================================================
        
        // Always write one result per loaded frame, preserving input/output correspondence.
        phase_start = Clock::now();
        const auto color = FilterColorCloud(ScalePlyColorCloud(raw_color, ply_scale_));
        PublishFrame(input, color, submap, pose, frame.timestamp_ns);
        WritePose(pose);
        const double publish_ms = ElapsedMs(phase_start);
        const double total_ms = ElapsedMs(frame_start);
        timings << processed << ',' << frame.timestamp_ns << ',' << registration_backend_ << ',' << input->size() << ',' << submap->size()
                << ',' << load_ms << ',' << filter_ms << ',' << crop_ms << ',' << last_ndt_ms_ << ',' << last_fitness_ms_
                << ',' << publish_ms << ',' << total_ms << ',' << converged << ',' << score << '\n';
        timings.flush();
        ndt_sum_ms += last_ndt_ms_;
        fitness_sum_ms += last_fitness_ms_;
        frames_sum_ms += total_ms;
        ++processed;
        rclcpp::spin_some(node_);
        if (processed == 1 || processed % 25 == 0) {
            RCLCPP_INFO(node_->get_logger(), "Frame %zu/%zu: %s=%s, score=%.4f, pose=(%.3f, %.3f, %.3f), NDT=%.2f ms, fitness=%.2f ms, frame=%.2f ms",
                        processed, ply_frames_.size(), registration_backend_.c_str(), converged ? "converged" : (guided ? "sensor_prior" : "failed"), score, pose(0, 3), pose(1, 3), pose(2, 3), last_ndt_ms_, last_fitness_ms_, total_ms);
        }
    }
    if (guided) {
        RCLCPP_INFO(node_->get_logger(), "Finished: %zu poses, %zu accepted map updates, %zu sensor-prior fallbacks (%zu geometry support, %zu other map rejections).",
                    processed, processed - failed, failed, geometry_fallbacks, failed - geometry_fallbacks);
    } else {
        RCLCPP_INFO(node_->get_logger(), "Finished: %zu poses, %zu failed alignments.", processed, failed);
    }
    RCLCPP_INFO(node_->get_logger(), "Timing sums: NDT %.3f s, fitness %.3f s, frames %.3f s; startup map excluded.", ndt_sum_ms / 1000.0, fitness_sum_ms / 1000.0, frames_sum_ms / 1000.0);
    return processed > 0;
}

bool LocalizationApp::AlignNDT(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                               const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                               const Eigen::Matrix4f& initial_guess,
                               Eigen::Matrix4f& pose, double& score) {
    last_ndt_ms_ = 0.0;
    last_fitness_ms_ = 0.0;
    const auto ndt_start = Clock::now();
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>* registration = cuda_ndt_.get();
    auto registration_target = target;
    Eigen::Matrix4f registration_guess = initial_guess;
    Eigen::Vector3f map_origin = Eigen::Vector3f::Zero();
    if (registration_backend_ == "pcl_ndt") {
        ndt.setResolution(ndt_resolution_);
        ndt.setStepSize(ndt_step_size_);
        ndt.setTransformationEpsilon(ndt_transformation_epsilon_);
        ndt.setMaximumIterations(ndt_max_iterations_);
        registration = &ndt;
    } else {
        // CUDA voxel moments use float: E[xx^T]-E[x]E[x]^T loses precision at
        // kilometre-scale map coordinates. Translate target/guess into a local
        // map frame; quantize the origin to preserve the voxel grid phase.
        for (int axis = 0; axis < 3; ++axis)
            map_origin[axis] = static_cast<float>(std::floor(initial_guess(axis, 3) / ndt_resolution_) * ndt_resolution_);
        auto local_target = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*target);
        for (auto& point : *local_target) point.getVector3fMap() -= map_origin;
        registration_target = local_target;
        registration_guess.block<3, 1>(0, 3) -= map_origin;
    }
    registration->setInputSource(source);
    registration->setInputTarget(registration_target);

    pcl::PointCloud<pcl::PointXYZ> aligned;
    registration->align(aligned, registration_guess);
    if (registration_backend_ == "cuda_ndt") {
        CheckCuda(cudaGetLastError(), "CUDA NDT kernel launch");
        CheckCuda(cudaDeviceSynchronize(), "CUDA NDT completion");
    }
    last_ndt_ms_ = ElapsedMs(ndt_start);
    if (!registration->hasConverged()) return false;
    if (registration_backend_ == "cuda_ndt" &&
        (!cuda_ndt_->getFinalHessian().allFinite() || cuda_ndt_->getFinalHessian().norm() < 1e-9)) return false;

    pose = registration->getFinalTransformation();
    pose.block<3, 1>(0, 3) += map_origin;
    if (!IsFinite(pose)) return false;
    const auto fitness_start = Clock::now();
    score = registration->getFitnessScore();
    last_fitness_ms_ = ElapsedMs(fitness_start);
    return IsFinite(pose) && std::isfinite(score);
}

// ** ================================================================== **
// ** ========================== Rviz $ Record ========================= **
// ** ================================================================== **
void LocalizationApp::PublishGlobalMap() {
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(*map_cloud_, message);
    message.header.frame_id = "map";
    message.header.stamp = node_->now();
    map_pub_->publish(message);
}

void LocalizationApp::PublishFrame(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input,
                                   const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& color,
                                   const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& submap,
                                   const Eigen::Matrix4f& pose, const std::int64_t timestamp_ns) {
    const rclcpp::Time recorded_time(timestamp_ns);
    sensor_msgs::msg::PointCloud2 input_message;
    if (color && !color->empty()) {
        pcl::toROSMsg(*color, input_message);
    } else {
        pcl::toROSMsg(*input, input_message);
    }
    input_message.header.frame_id = "image_point_cloud";
    input_message.header.stamp = recorded_time;
    input_cloud_pub_->publish(input_message);

    sensor_msgs::msg::PointCloud2 submap_message;
    pcl::toROSMsg(*submap, submap_message);
    submap_message.header.frame_id = "map";
    submap_message.header.stamp = recorded_time;
    submap_pub_->publish(submap_message);

    tf_broadcaster_->sendTransform(ToTransformStamped(pose, "map", "image_point_cloud", timestamp_ns));

    const auto pose_message = ToPoseStamped(pose, timestamp_ns);
    pose_pub_->publish(pose_message);
    path_.header = pose_message.header;
    path_.poses.push_back(pose_message);
    path_pub_->publish(path_);
}

void LocalizationApp::WritePose(const Eigen::Matrix4f& pose) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            result_file_ << pose(row, column);
            if (row == 2 && column == 3) {
                result_file_ << '\n';
            } else {
                result_file_ << ' ';
            }
        }
    }
    result_file_.flush();
}

// ** ================================================================== **
// ** ======================= Messages transform ======================= **
// ** ================================================================== **
geometry_msgs::msg::TransformStamped LocalizationApp::ToTransformStamped(const Eigen::Matrix4f& matrix,
                                                                         const std::string& parent_frame,
                                                                         const std::string& child_frame,
                                                                         const std::int64_t timestamp_ns) const {
    geometry_msgs::msg::TransformStamped message;
    message.header.stamp = rclcpp::Time(timestamp_ns);
    message.header.frame_id = parent_frame;
    message.child_frame_id = child_frame;
    message.transform.translation.x = matrix(0, 3);
    message.transform.translation.y = matrix(1, 3);
    message.transform.translation.z = matrix(2, 3);

    tf2::Matrix3x3 rotation(matrix(0, 0), matrix(0, 1), matrix(0, 2),
                            matrix(1, 0), matrix(1, 1), matrix(1, 2),
                            matrix(2, 0), matrix(2, 1), matrix(2, 2));
    tf2::Quaternion quaternion;
    rotation.getRotation(quaternion);
    quaternion.normalize();
    message.transform.rotation.x = quaternion.x();
    message.transform.rotation.y = quaternion.y();
    message.transform.rotation.z = quaternion.z();
    message.transform.rotation.w = quaternion.w();
    return message;
}

geometry_msgs::msg::PoseStamped LocalizationApp::ToPoseStamped(const Eigen::Matrix4f& matrix,  const std::int64_t timestamp_ns) const {
    const auto transform = ToTransformStamped(matrix, "map", "image_point_cloud", timestamp_ns);
    geometry_msgs::msg::PoseStamped pose;
    pose.header = transform.header;
    pose.pose.position.x = transform.transform.translation.x;
    pose.pose.position.y = transform.transform.translation.y;
    pose.pose.position.z = transform.transform.translation.z;
    pose.pose.orientation = transform.transform.rotation;
    return pose;
}

bool LocalizationApp::IsFinite(const Eigen::Matrix4f& matrix) const {
    return matrix.array().isFinite().all();
}

bool LocalizationApp::TimestampNsFromPlyFilename(const boost::filesystem::path& file, std::int64_t& timestamp_ns) {
    const std::string stem = file.stem().string();
    const std::string marker = "_ts_";
    const std::size_t marker_position = stem.rfind(marker);
    if (marker_position == std::string::npos) return false;

    try {
        const std::string nanoseconds = stem.substr(marker_position + marker.size());
        std::size_t parsed = 0;
        const auto value = std::stoll(nanoseconds, &parsed);
        if (parsed != nanoseconds.size()) return false;
        timestamp_ns = value;
        return timestamp_ns > 0;
    } catch (const std::exception&) {
        return false;
    }
}

void LocalizationApp::Shutdown() {
    if (result_file_.is_open()) result_file_.close();
    tf_broadcaster_.reset();
    input_cloud_pub_.reset();
    submap_pub_.reset();
    map_pub_.reset();
    pose_pub_.reset();
    path_pub_.reset();
    node_.reset();
    if (rclcpp::ok()) rclcpp::shutdown();
}

}  // namespace localization

int main(int argc, char** argv) {
    localization::LocalizationApp app;
    return app.Start(argc, argv);
}
