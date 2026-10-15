#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

using namespace std;

rclcpp::Node::SharedPtr node;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub;
rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr radar_pose_pub;
nav_msgs::msg::Path path;

void radar_pose_publisher(float x, float y, float z, float yaw) {
    geometry_msgs::msg::PoseStamped pose;
    tf2::Quaternion quaternion;
    quaternion.setRPY(0.0, 0.0, yaw);
    quaternion.normalize();

    pose.header.stamp = node->now();
    pose.header.frame_id = "map";

    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = z;

    pose.pose.orientation.x = quaternion.x();
    pose.pose.orientation.y = quaternion.y();
    pose.pose.orientation.z = quaternion.z();
    pose.pose.orientation.w = quaternion.w();
    radar_pose_pub->publish(pose);

    path.header.stamp = node->now();
    path.header.frame_id = "map";
    path.poses.push_back(pose);
    path_pub->publish(path);
}

geometry_msgs::msg::TransformStamped matrix4fToTransformStamped(
    const Eigen::Matrix4f& matrix,
    const std::string& parent_frame,
    const std::string& child_frame) {

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = node->now();
    tf_msg.header.frame_id = parent_frame;
    tf_msg.child_frame_id = child_frame;

    tf_msg.transform.translation.x = matrix(0, 3);
    tf_msg.transform.translation.y = matrix(1, 3);
    tf_msg.transform.translation.z = matrix(2, 3);

    tf2::Matrix3x3 rotation(
        matrix(0, 0), matrix(0, 1), matrix(0, 2),
        matrix(1, 0), matrix(1, 1), matrix(1, 2),
        matrix(2, 0), matrix(2, 1), matrix(2, 2));

    tf2::Quaternion q;
    rotation.getRotation(q);
    q.normalize();

    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();

    return tf_msg;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    node = std::make_shared<rclcpp::Node>("map_visualization");

    node->declare_parameter<std::string>("map_file", "path_to_map.pcd");
    node->declare_parameter<std::string>("t_file", "path_to_radar_t_list.txt");
    node->declare_parameter<std::string>("pose_file", "path_to_result.txt");
    node->declare_parameter<std::string>("pcd_folder", "path_to_pcd_folder");

    const string map_file = node->get_parameter("map_file").as_string();
    const string t_file = node->get_parameter("t_file").as_string();
    const string pose_file = node->get_parameter("pose_file").as_string();
    const string pcd_folder = node->get_parameter("pcd_folder").as_string();

    auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    auto map_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/visual/map", map_qos);
    auto radar_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/visual/radar", rclcpp::QoS(1));
    radar_pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/visual/tranformed_radar_pose", rclcpp::QoS(1));
    path_pub = node->create_publisher<nav_msgs::msg::Path>(
        "/visual/localization_path", rclcpp::QoS(1));

    auto tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);

    pcl::PointCloud<pcl::PointXYZI>::Ptr map_pc(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_file, *map_pc) == -1) {
        PCL_ERROR("Couldn't read map file: %s\n", map_file.c_str());
        rclcpp::shutdown();
        return -1;
    }

    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(map_pc);
    pass.setFilterFieldName("intensity");
    pass.setFilterLimits(10, 256);
    pass.setFilterLimitsNegative(false);
    pass.filter(*map_pc);

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*map_pc, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = node->now();
    map_pub->publish(map_msg);

    std::this_thread::sleep_for(std::chrono::seconds(3));

    string line;
    vector<string> t_list;

    ifstream t_in(t_file);
    ifstream pose_in(pose_file);

    if (!t_in.is_open()) {
        RCLCPP_ERROR(node->get_logger(), "Error opening timestamp file: %s", t_file.c_str());
        rclcpp::shutdown();
        return -1;
    }

    while (getline(t_in, line) && rclcpp::ok()) {
        const string pcd_file = pcd_folder + "/" + line + ".pcd";
        t_list.push_back(pcd_file);
    }

    if (!pose_in.is_open()) {
        RCLCPP_ERROR(node->get_logger(), "Error opening pose file: %s", pose_file.c_str());
        rclcpp::shutdown();
        return -1;
    }

    size_t idx = 0;
    while (getline(pose_in, line) && rclcpp::ok()) {
        if (idx >= t_list.size()) {
            RCLCPP_WARN(node->get_logger(), "Pose file contains more entries than timestamp/PCD list. Stopping.");
            break;
        }

        string s;
        stringstream ss(line);
        vector<string> record;
        while (getline(ss, s, ' ')) {
            if (!s.empty()) {
                record.push_back(s);
            }
        }

        if (record.size() < 12) {
            RCLCPP_WARN(node->get_logger(), "Skipping malformed pose row %zu: expected at least 12 values.", idx);
            ++idx;
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(t_list[idx], *radar_pc) == -1) {
            PCL_ERROR("Couldn't read radar PCD: %s\n", t_list[idx].c_str());
            ++idx;
            continue;
        }

        RCLCPP_WARN(node->get_logger(), "%s", record[0].c_str());

        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform <<
            stof(record[0]), stof(record[1]), stof(record[2]), stof(record[3]),
            stof(record[4]), stof(record[5]), stof(record[6]), stof(record[7]),
            stof(record[8]), stof(record[9]), stof(record[10]), stof(record[11]),
            0, 0, 0, 1;

        const auto tf_msg = matrix4fToTransformStamped(transform, "map", "radar");
        tf_broadcaster->sendTransform(tf_msg);

        sensor_msgs::msg::PointCloud2 radar_msg;
        pcl::toROSMsg(*radar_pc, radar_msg);
        radar_msg.header.frame_id = "radar";
        radar_msg.header.stamp = node->now();
        radar_pub->publish(radar_msg);

        radar_pose_publisher(
            transform(0, 3),
            transform(1, 3),
            transform(2, 3),
            atan2(transform(1, 0), transform(0, 0)));

        ++idx;
        rclcpp::spin_some(node);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    rclcpp::shutdown();
    return 0;
}
