#include "localization/localization.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>

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

int LocalizationApp::Start(int argc, char** argv) {
    rclcpp::init(argc, argv);
    node_ = std::make_shared<rclcpp::Node>("image_point_cloud_localizer");

    if (!LoadParameters()) {
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

    if (!IndexPlyFiles(ply_path_) || !LoadMap()) {
        Shutdown();
        return -1;
    }

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
    node_->declare_parameter<double>("ply_scale", ply_scale_);

    node_->declare_parameter<bool>("keep_alive", keep_alive_);
    node_->declare_parameter<bool>("publish_global_map", publish_global_map_);
    node_->declare_parameter<int>("max_frames", max_frames_);


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

    ndt_resolution_ = node_->get_parameter("ndt_resolution").as_double();
    ndt_step_size_ = node_->get_parameter("ndt_step_size").as_double();
    ndt_transformation_epsilon_ = node_->get_parameter("ndt_transformation_epsilon").as_double();
    ply_scale_ = node_->get_parameter("ply_scale").as_double();

    keep_alive_ = node_->get_parameter("keep_alive").as_bool();
    publish_global_map_ = node_->get_parameter("publish_global_map").as_bool();
    max_frames_ = node_->get_parameter("max_frames").as_int();


    if (ply_path_.empty() || map_path_.empty() || save_path_.empty() || save_file_.empty()) {
        RCLCPP_ERROR(node_->get_logger(), "PLY, map and output paths are required.");
        return false;
    }
    if (init_guess_file_.empty() &&
        (initial_pose_.size() != 4 || !std::all_of(initial_pose_.begin(), initial_pose_.end(), [](double value) { return std::isfinite(value); }))) {
        RCLCPP_ERROR(node_->get_logger(), "initial_pose must contain finite [x, y, z, yaw].");
        return false;
    }
    const std::vector<double> positive = {ply_scale_, submap_half_size_, submap_half_height_, ndt_resolution_, ndt_step_size_, ndt_transformation_epsilon_};
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
    const boost::filesystem::path cache_path = map_cache_file_.empty() ? boost::filesystem::path() : map_directory / map_cache_file_;
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
            if (pcl::io::savePCDFileBinary(cache_path.string(), *map_cloud_) == 0) {
                RCLCPP_INFO(node_->get_logger(), "Saved map cache: %s", cache_path.string().c_str());
            } else {
                RCLCPP_WARN(node_->get_logger(), "Failed to save map cache: %s", cache_path.string().c_str());
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
bool LocalizationApp::RunLocalization() {
    Eigen::Matrix4f pose;
    if (!LoadInitialPose(pose)) return false;

    if (!OpenDB3File()) return false;
    DB3Frame db3_frame;

    std::size_t processed = 0;
    std::size_t failed = 0;
    for (const auto& frame : ply_frames_) {
        if (!rclcpp::ok()) break;
        if (max_frames_ > 0 && processed >= static_cast<std::size_t>(max_frames_)) break;
        if (!LoadDB3Frame(frame.timestamp_ns, db3_frame)) return false;
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_color;
        if (!LoadPlyFrame(frame, raw_cloud, raw_color)) return false;
        const auto input = FilterInputCloud(ScalePlyPointCloud(raw_cloud, ply_scale_));
        const auto submap = CropSubmap(pose);
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
        const bool converged = AlignNDT(input, submap, pose, candidate, score);
        if (converged) {
            pose = candidate;
        } else {
            ++failed;
            RCLCPP_WARN(node_->get_logger(), "NDT failed at frame %zu; keeping previous pose.", processed);
        }
        // ======================================================
        
        // Always write one result per loaded frame, preserving input/output correspondence.
        const auto color = FilterColorCloud(ScalePlyColorCloud(raw_color, ply_scale_));
        PublishFrame(input, color, submap, pose, frame.timestamp_ns);
        WritePose(pose);
        ++processed;
        rclcpp::spin_some(node_);
        if (processed == 1 || processed % 25 == 0) {
            RCLCPP_INFO(node_->get_logger(), "Frame %zu/%zu: NDT=%s, score=%.4f, pose=(%.3f, %.3f, %.3f)",
                        processed, ply_frames_.size(), converged ? "converged" : "failed", score, pose(0, 3), pose(1, 3), pose(2, 3));
        }
    }
    RCLCPP_INFO(node_->get_logger(), "Finished: %zu poses, %zu failed alignments.", processed, failed);
    return processed > 0;
}

bool LocalizationApp::AlignNDT(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                               const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                               const Eigen::Matrix4f& initial_guess,
                               Eigen::Matrix4f& pose, double& score) const {
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setResolution(ndt_resolution_);
    ndt.setStepSize(ndt_step_size_);
    ndt.setTransformationEpsilon(ndt_transformation_epsilon_);
    ndt.setMaximumIterations(ndt_max_iterations_);
    ndt.setInputSource(source);
    ndt.setInputTarget(target);

    pcl::PointCloud<pcl::PointXYZ> aligned;
    ndt.align(aligned, initial_guess);
    if (!ndt.hasConverged()) return false;

    pose = ndt.getFinalTransformation();
    score = ndt.getFitnessScore();
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
