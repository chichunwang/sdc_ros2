#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

std::vector<Eigen::Matrix4f> LoadPoses(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read poses: " + path);
    std::vector<Eigen::Matrix4f> poses;
    std::string line;
    while (std::getline(file, line)) {
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        std::istringstream values(line);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 4; ++j)
            if (!(values >> pose(i, j)) || !std::isfinite(pose(i, j))) throw std::runtime_error("Invalid pose row");
        std::string extra;
        if (values >> extra) throw std::runtime_error("Extra pose values");
        poses.push_back(pose);
    }
    return poses;
}

Cloud::Ptr Filter(const Cloud::ConstPtr& source, float scale, float leaf) {
    auto finite = std::make_shared<Cloud>();
    for (const auto& input : *source) {
        pcl::PointXYZ point(input.x * scale, input.y * scale, input.z * scale);
        if (pcl::isFinite(point)) finite->push_back(point);
    }
    finite->is_dense = true;
    if (leaf <= 0.0f) return finite;
    auto output = std::make_shared<Cloud>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(finite);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*output);
    return output;
}

double Quantile(const std::vector<float>& sorted, double q) {
    const double index = q * (sorted.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(index));
    const auto high = static_cast<std::size_t>(std::ceil(index));
    return sorted[low] + (index - low) * (sorted[high] - sorted[low]);
}

void WriteStats(std::ofstream& file, int idx, const char* backend, std::vector<float> distances, std::size_t map_points) {
    if (distances.empty()) throw std::runtime_error("No nearest-neighbor distances");
    std::sort(distances.begin(), distances.end());
    double sum = 0.0, squared = 0.0, trimmed = 0.0;
    std::array<std::size_t, 3> inliers{};
    const auto retained = static_cast<std::size_t>(std::ceil(distances.size() * 0.95));
    for (std::size_t i = 0; i < distances.size(); ++i) {
        const double d = distances[i];
        sum += d;
        squared += d * d;
        if (i < retained) trimmed += d * d;
        inliers[0] += d <= 0.3;
        inliers[1] += d <= 0.5;
        inliers[2] += d <= 1.0;
    }
    const double n = distances.size();
    file << idx << ',' << backend << ',' << distances.size() << ',' << map_points << ','
         << std::sqrt(squared / n) << ',' << sum / n << ',' << Quantile(distances, .5) << ','
         << Quantile(distances, .9) << ',' << Quantile(distances, .95) << ','
         << inliers[0] / n << ',' << inliers[1] / n << ',' << inliers[2] / n << ',' << std::sqrt(trimmed / retained) << '\n';
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr << "map_input map_leaf ply_directory scale input_leaf cpu_poses gpu_poses filename_list output_directory\n";
        return 2;
    }
    try {
        const float map_leaf = std::stof(argv[2]), scale = std::stof(argv[4]), input_leaf = std::stof(argv[5]);
        std::array<std::vector<Eigen::Matrix4f>, 2> poses{LoadPoses(argv[6]), LoadPoses(argv[7])};
        std::ifstream manifest(argv[8]);
        std::vector<std::string> filenames;
        std::string filename;
        while (std::getline(manifest, filename)) if (!filename.empty()) filenames.push_back(filename);
        if (filenames.empty() || filenames.size() != poses[0].size() || filenames.size() != poses[1].size())
            throw std::runtime_error("Input/pose frame count mismatch");
        auto raw_map = std::make_shared<Cloud>();
        if (pcl::io::loadPCDFile(argv[1], *raw_map) < 0) throw std::runtime_error("Cannot read common map");
        auto map = map_leaf > 0 ? Filter(raw_map, 1.0f, map_leaf) : raw_map;
        if (map->empty()) throw std::runtime_error("Empty common map");
        if (map_leaf > 0 && pcl::io::savePCDFileBinary(std::string(argv[9]) + "/derived_map_1m.pcd", *map) != 0)
            throw std::runtime_error("Cannot save derived map");
        raw_map.reset();
        pcl::KdTreeFLANN<pcl::PointXYZ> tree;
        tree.setInputCloud(map);
        std::ofstream metrics(std::string(argv[9]) + "/map_residuals.csv");
        metrics.precision(12);
        metrics << "idx,backend,input_points,map_points,rmse_m,mean_m,median_m,p90_m,p95_m,inlier_0_3,inlier_0_5,inlier_1_0,trimmed_rmse95_m\n";
        std::ofstream cloud_plot(std::string(argv[9]) + "/frame_clouds.csv");
        cloud_plot.precision(10);
        cloud_plot << "idx,backend,x,y,z\n";
        std::array<std::vector<float>, 2> pooled;
        constexpr std::array<const char*, 2> labels{"CPU", "CUDA"};
        for (std::size_t idx = 0; idx < filenames.size(); ++idx) {
            pcl::PCLPointCloud2 packed;
            if (pcl::io::loadPLYFile(std::string(argv[3]) + "/" + filenames[idx], packed) < 0)
                throw std::runtime_error("Cannot read PLY: " + filenames[idx]);
            auto raw = std::make_shared<Cloud>();
            pcl::fromPCLPointCloud2(packed, *raw);
            auto input = Filter(raw, scale, input_leaf);
            for (std::size_t backend = 0; backend < 2; ++backend) {
                Cloud transformed;
                // Pose already maps scaled source to map; never apply init/extrinsics again.
                pcl::transformPointCloud(*input, transformed, poses[backend][idx]);
                std::vector<float> distances;
                distances.reserve(transformed.size());
                std::vector<int> index(1);
                std::vector<float> squared(1);
                for (const auto& point : transformed) {
                    if (tree.nearestKSearch(point, 1, index, squared) != 1 || !std::isfinite(squared[0]))
                        throw std::runtime_error("Nearest-neighbor lookup failed");
                    distances.push_back(std::sqrt(std::max(0.0f, squared[0])));
                    if (idx == 0 || idx + 1 == filenames.size())
                        cloud_plot << idx << ',' << labels[backend] << ',' << point.x << ',' << point.y << ',' << point.z << '\n';
                }
                WriteStats(metrics, static_cast<int>(idx), labels[backend], distances, map->size());
                pooled[backend].insert(pooled[backend].end(), distances.begin(), distances.end());
            }
        }
        for (std::size_t backend = 0; backend < 2; ++backend)
            WriteStats(metrics, -1, labels[backend], pooled[backend], map->size());
        Eigen::Vector3f center = poses[0][0].block<3, 1>(0, 3);
        std::ofstream map_plot(std::string(argv[9]) + "/map_view.csv");
        map_plot.precision(10);
        map_plot << "x,y,z\n";
        for (std::size_t i = 0; i < map->size(); i += 2) {
            const auto& p = (*map)[i];
            if (std::abs(p.x - center.x()) < 150 && std::abs(p.y - center.y()) < 150 && std::abs(p.z - center.z()) < 30)
                map_plot << p.x << ',' << p.y << ',' << p.z << '\n';
        }
        std::cout << "Evaluated " << filenames.size() << " frames against " << map->size() << " fixed map points, both backends.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
