#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Use the same PointXYZ VoxelGrid and binary PCD output as LocalizationApp::LoadMap.
bool DownsampleMap(const rclcpp::Node::SharedPtr& node) {
    const auto input = node->get_parameter("input_file").as_string();
    const auto output = node->get_parameter("output_file").as_string();
    const double leaf_size = node->get_parameter("map_voxel_leaf_size").as_double();
    if (input.empty() || output.empty() || !std::isfinite(leaf_size) || leaf_size < 0.0) {
        RCLCPP_ERROR(node->get_logger(), "Input/output paths must be non-empty and voxel size finite and non-negative.");
        return false;
    }
    const auto input_path = std::filesystem::weakly_canonical(input);
    const auto output_path = std::filesystem::weakly_canonical(output);
    if (input_path == output_path ||
        (std::filesystem::exists(output_path) && std::filesystem::equivalent(input_path, output_path))) {
        RCLCPP_ERROR(node->get_logger(), "Output must differ from the input map.");
        return false;
    }
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    RCLCPP_INFO(node->get_logger(), "Loading map: %s", input.c_str());
    if (pcl::io::loadPCDFile(input, *cloud) < 0 || cloud->empty()) {
        RCLCPP_ERROR(node->get_logger(), "Failed to read a non-empty map: %s", input.c_str());
        return false;
    }
    const auto original_size = cloud->size();
    const auto crop_min = node->get_parameter("crop_min").as_double_array();
    const auto crop_max = node->get_parameter("crop_max").as_double_array();
    if (!crop_min.empty() || !crop_max.empty()) {
        if (crop_min.size() != 3 || crop_max.size() != 3) return false;
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(crop_min[axis]) || !std::isfinite(crop_max[axis]) || crop_min[axis] >= crop_max[axis]) return false;
        auto cropped = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(crop_min[0],crop_min[1],crop_min[2],1));
        crop.setMax(Eigen::Vector4f(crop_max[0],crop_max[1],crop_max[2],1));
        crop.filter(*cropped);
        cloud = cropped;
        RCLCPP_INFO(node->get_logger(), "Explicit crop: %zu -> %zu points.", original_size,cloud->size());
    }
    if (cloud->empty()) return false;
    if (leaf_size > 0.0) {
        pcl::PointXYZ low, high;
        pcl::getMinMax3D(*cloud,low,high);
        const Eigen::Array3d cells = ((high.getVector3fMap()-low.getVector3fMap()).cast<double>().array()/leaf_size).ceil()+1;
        if (!cells.isFinite().all() || cells.prod() > std::numeric_limits<std::int32_t>::max()) {
            RCLCPP_ERROR(node->get_logger(), "VoxelGrid integer grid would overflow; use explicit crop bounds or a larger leaf. Refusing unfiltered output.");
            return false;
        }
        auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        const float leaf = static_cast<float>(leaf_size);
        voxel.setLeafSize(leaf, leaf, leaf);
        voxel.filter(*filtered);
        cloud = filtered;
    }
    if (cloud->empty()) {
        RCLCPP_ERROR(node->get_logger(), "Filtered map is empty.");
        return false;
    }
    const auto parent = output_path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    if (pcl::io::savePCDFileBinary(output, *cloud) < 0) {
        RCLCPP_ERROR(node->get_logger(), "Failed to save map: %s", output.c_str());
        return false;
    }
    RCLCPP_INFO(node->get_logger(), "Map: %zu -> %zu points (leaf %.3f m), saved to %s",
                original_size, cloud->size(), leaf_size, output.c_str());
    return true;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("map_modifier");
    node->declare_parameter<std::string>("input_file", "/root/data/Track1/map.pcd");
    node->declare_parameter<std::string>("output_file", "/root/data/Track1/map_downsampled_1m.pcd");
    node->declare_parameter<double>("map_voxel_leaf_size", 1.0);
    node->declare_parameter<std::vector<double>>("crop_min", std::vector<double>{});
    node->declare_parameter<std::vector<double>>("crop_max", std::vector<double>{});
    bool success = false;
    try {
        success = DownsampleMap(node);
    } catch (const std::exception& error) {
        RCLCPP_ERROR(node->get_logger(), "Map preprocessing failed: %s", error.what());
    }
    rclcpp::shutdown();
    return success ? 0 : 1;
}
