// Thin streaming adapter for the unchanged historical sensor-guided ICP core.
// This file supplies no GNSS observations and implements no registration math.
// The caller supplies one causal motion prior at a time; the core's historical
// regularizer and float/PCL preprocessing remain unchanged.
#include "localization/sensor_guided.h"
#include <cuda_ndt_backend/nearest_neighbor.h>

#include <cuda_runtime_api.h>
#include <Eigen/Geometry>
#include <pcl/common/point_tests.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Options {
    std::string map_path;
    double scale = 3.0;
    std::string geometry = "directional";
    double map_leaf = 1.0;
    double input_leaf = 0.3;
    double half_xy = 100.0;
    double half_z = 30.0;
    double max_xy = 1.5;
    double max_yaw_deg = 8.0;
    double alpha = 0.25;
    bool cpu_reference = false;
};

double ElapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double ParseNumber(const std::string& value, const std::string& name) {
    std::size_t consumed = 0;
    const double number = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(number))
        throw std::invalid_argument(name + " must be a finite number");
    return number;
}

void ValidateFloat(double value, const std::string& name, bool allow_zero = false) {
    const float as_float = static_cast<float>(value);
    if (!std::isfinite(value) || !std::isfinite(as_float) ||
        (allow_zero ? value < 0.0 : value <= 0.0) ||
        (value > 0.0 && as_float <= 0.0f))
        throw std::invalid_argument(name + " is outside the usable positive float range");
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--cpu-reference") {
            options.cpu_reference = true;
            continue;
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--map") options.map_path = value;
        else if (key == "--geometry") options.geometry = value;
        else if (key == "--scale") options.scale = ParseNumber(value, key);
        else if (key == "--map-leaf") options.map_leaf = ParseNumber(value, key);
        else if (key == "--input-leaf") options.input_leaf = ParseNumber(value, key);
        else if (key == "--half-xy") options.half_xy = ParseNumber(value, key);
        else if (key == "--half-z") options.half_z = ParseNumber(value, key);
        else if (key == "--max-xy") options.max_xy = ParseNumber(value, key);
        else if (key == "--max-yaw-deg") options.max_yaw_deg = ParseNumber(value, key);
        else if (key == "--alpha") options.alpha = ParseNumber(value, key);
        else throw std::invalid_argument("Unknown option: " + key);
    }
    if (options.map_path.empty() || !std::filesystem::path(options.map_path).is_absolute())
        throw std::invalid_argument("--map must name an absolute PCD path");
    if (options.geometry != "directional" && options.geometry != "off")
        throw std::invalid_argument("--geometry must be directional or off");
    ValidateFloat(options.scale, "--scale");
    ValidateFloat(options.map_leaf, "--map-leaf", true);
    ValidateFloat(options.input_leaf, "--input-leaf", true);
    ValidateFloat(options.half_xy, "--half-xy");
    ValidateFloat(options.half_z, "--half-z");
    ValidateFloat(options.max_xy, "--max-xy");
    if (!std::isfinite(options.max_yaw_deg) || options.max_yaw_deg <= 0.0 || options.max_yaw_deg > 180.0)
        throw std::invalid_argument("--max-yaw-deg must lie in (0, 180]");
    if (!std::isfinite(options.alpha) || options.alpha < 0.0 || options.alpha > 1.0)
        throw std::invalid_argument("--alpha must lie in [0, 1]");
    return options;
}

// Keep even PCL's C-level stdout diagnostics out of the machine protocol.
// Preserve the caller's original stdout pipe, then route ordinary stdout to
// stderr. Only Write() uses the preserved pipe. This adapter targets Linux.
class ProtocolOutput {
 public:
    ProtocolOutput() {
        std::cout.flush();
        std::fflush(stdout);
        const int fd = ::dup(STDOUT_FILENO);
        if (fd < 0) throw std::runtime_error("Cannot duplicate protocol stdout");
        stream_ = ::fdopen(fd, "w");
        if (!stream_) {
            ::close(fd);
            throw std::runtime_error("Cannot open protocol stdout stream");
        }
        if (::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
            std::fclose(stream_);
            stream_ = nullptr;
            throw std::runtime_error("Cannot redirect diagnostic stdout");
        }
    }
    ~ProtocolOutput() { if (stream_) std::fclose(stream_); }
    ProtocolOutput(const ProtocolOutput&) = delete;
    ProtocolOutput& operator=(const ProtocolOutput&) = delete;

    void Write(const std::string& line) {
        if (std::fwrite(line.data(), 1, line.size(), stream_) != line.size() ||
            std::fputc('\n', stream_) == EOF || std::fflush(stream_) != 0)
            throw std::runtime_error("Cannot write protocol response");
    }
 private:
    std::FILE* stream_ = nullptr;
};

Cloud::Ptr FinitePoints(const Cloud::ConstPtr& input, float scale = 1.0f) {
    auto output = std::make_shared<Cloud>();
    output->points.reserve(input->size());
    for (const auto& point : input->points) {
        if (!pcl::isFinite(point)) continue;
        pcl::PointXYZ scaled;
        scaled.x = scale * point.x;
        scaled.y = scale * point.y;
        scaled.z = scale * point.z;
        // Equivalent to the old float scale followed by finite-input removal.
        if (pcl::isFinite(scaled)) output->points.push_back(scaled);
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = true;
    return output;
}

Cloud::Ptr Voxelize(const Cloud::Ptr& input, double leaf_size) {
    if (leaf_size == 0.0 || input->empty()) return input;
    auto output = std::make_shared<Cloud>();
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(input);
    const float leaf = static_cast<float>(leaf_size);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*output);
    return output;
}

Cloud::Ptr CropSubmap(const Cloud::ConstPtr& map, const Eigen::Matrix4f& prior,
                      const Options& options) {
    auto submap = std::make_shared<Cloud>();
    const float x = prior(0, 3), y = prior(1, 3), z = prior(2, 3);
    const float half_xy = static_cast<float>(options.half_xy);
    const float half_z = static_cast<float>(options.half_z);
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(map);
    crop.setMin(Eigen::Vector4f(x - half_xy, y - half_xy, z - half_z, 1.0f));
    crop.setMax(Eigen::Vector4f(x + half_xy, y + half_xy, z + half_z, 1.0f));
    crop.filter(*submap);
    return submap;
}

void ValidatePose(const Eigen::Matrix4d& pose) {
    if (!pose.allFinite()) throw std::invalid_argument("Nonfinite pose matrix");
    if ((pose.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).cwiseAbs().maxCoeff() > 1e-6)
        throw std::invalid_argument("Pose matrix bottom row must be 0 0 0 1");
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-3 ||
        std::abs(rotation.determinant() - 1.0) > 1e-3)
        throw std::invalid_argument("Pose rotation must be proper and orthonormal");
    if (!pose.cast<float>().allFinite())
        throw std::invalid_argument("Pose lies outside legacy float matrix range");
}

void CheckCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

double WarmCuda(bool cpu_reference) {
    if (cpu_reference) return 0.0;
    const auto start = Clock::now();
    int devices = 0;
    CheckCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices < 1) throw std::runtime_error("CUDA backend requires a CUDA device");
    {
        const std::vector<float> target{0.0f, 0.0f, 0.0f};
        cuda_ndt_backend::NearestNeighbor nearest(target, 1);
        std::vector<int> indices;
        nearest.Query(target, indices);
        if (indices.size() != 1 || indices[0] != 0)
            throw std::runtime_error("CUDA nearest-neighbor warmup returned invalid index");
    }
    CheckCuda(cudaDeviceSynchronize(), "CUDA warmup completion");
    return ElapsedMs(start);
}

int Run(const Options& options) {
    ProtocolOutput protocol;
    localization::GeometryConfig geometry;
    geometry.mode = options.geometry == "directional" ? localization::GeometryMode::Directional :
                                                       localization::GeometryMode::Off;
    // Remaining GeometryConfig defaults are those of the unchanged old core:
    // gain=2, tangent=.05, radius=2.5m, neighbors=16, minimum=6, planarity=.2.
    geometry.Validate();
    const localization::HeightConstraint height_carry;
    std::cerr << "Old ICP bridge: fixed PLY scale=" << options.scale
              << ", geometry=" << options.geometry
              << ", NN=" << (options.cpu_reference ? "CPU_REFERENCE_SYNTHETIC_ONLY" : "CUDA")
              << "; priors supplied by caller; no GNSS/bag/prior-file reader\n";

    const auto preparation_start = Clock::now();
    auto raw_map = std::make_shared<Cloud>();
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(options.map_path, *raw_map) < 0)
        throw std::runtime_error("Cannot read map PCD: " + options.map_path);
    auto finite_map = FinitePoints(raw_map);
    const std::size_t raw_count = raw_map->size();
    raw_map.reset();
    auto map = Voxelize(finite_map, options.map_leaf);
    finite_map.reset();
    if (map->empty()) throw std::runtime_error("Map is empty after finite/voxel filtering");
    const double preparation_ms = ElapsedMs(preparation_start);
    const double warmup_ms = WarmCuda(options.cpu_reference);
    std::cerr << "Map: raw=" << raw_count << ", filtered=" << map->size() << '\n';
    {
        std::ostringstream ready;
        ready << std::setprecision(17) << "READY " << map->size() << ' '
              << preparation_ms << ' ' << warmup_ms;
        protocol.Write(ready.str());
    }

    Eigen::Vector4d previous_correction = Eigen::Vector4d::Zero();
    std::string line;
    std::size_t frame_index = 0;
    while (std::getline(std::cin, line)) {
        std::istringstream request(line);
        std::string operation, ply_path;
        if (!(request >> operation) || operation != "FRAME")
            throw std::invalid_argument("Expected FRAME request at local frame " + std::to_string(frame_index));
        request >> std::ws;
        if (request.peek() != '"' || !(request >> std::quoted(ply_path)) ||
            !std::filesystem::path(ply_path).is_absolute())
            throw std::invalid_argument("FRAME requires a quoted absolute PLY path");
        Eigen::Matrix4d prior_double;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                if (!(request >> prior_double(row, col)))
                    throw std::invalid_argument("FRAME requires exactly 16 row-major matrix values");
        request >> std::ws;
        if (!request.eof()) throw std::invalid_argument("Unexpected trailing FRAME input");
        ValidatePose(prior_double);
        const Eigen::Matrix4f prior = prior_double.cast<float>();

        const auto frame_start = Clock::now();
        auto raw_cloud = std::make_shared<Cloud>();
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path, *raw_cloud) < 0)
            throw std::runtime_error("Cannot read frame PLY: " + ply_path);
        const double load_ms = ElapsedMs(frame_start);
        const auto registration_start = Clock::now();
        const auto input = Voxelize(FinitePoints(raw_cloud, static_cast<float>(options.scale)), options.input_leaf);
        raw_cloud.reset();
        const auto submap = CropSubmap(map, prior, options);
        if (submap->empty())
            throw std::runtime_error("Empty submap at local frame " + std::to_string(frame_index));
        Eigen::Matrix4f output = prior;
        Eigen::Vector4d next_correction = Eigen::Vector4d::Zero();
        localization::GuidedMetrics metrics;
        const bool accepted = localization::RefineSensorGuided(
            input, submap, prior, prior, false, previous_correction,
            options.alpha, options.max_xy, options.max_yaw_deg * kPi / 180.0,
            output, next_correction, metrics, !options.cpu_reference, geometry, height_carry);
        if (!options.cpu_reference)
            CheckCuda(cudaDeviceSynchronize(), "ICP frame completion");
        const double registration_ms = ElapsedMs(registration_start);
        const double frame_ms = ElapsedMs(frame_start);
        ValidatePose(output.cast<double>());
        if (!next_correction.allFinite()) throw std::runtime_error("Old ICP returned a nonfinite correction");
        // Exact old localization.cpp behavior: the core returns zero correction
        // on rejection, and it replaces the previous correction on EVERY frame.
        previous_correction = next_correction;

        std::ostringstream response;
        response << std::setprecision(17) << "RESULT " << (accepted ? 1 : 0)
                 << ' ' << load_ms << ' ' << registration_ms << ' ' << frame_ms
                 << ' ' << localization::GuidedUpdateStatusName(metrics.update_status)
                 << ' ' << metrics.inliers << ' ' << metrics.iterations
                 << ' ' << metrics.prior_cost << ' ' << metrics.final_cost
                 << ' ' << metrics.prior_rmse << ' ' << metrics.final_rmse
                 << ' ' << metrics.candidate_inliers << ' ' << metrics.candidate_reliable_points
                 << ' ' << metrics.candidate_unique_targets;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col) response << ' ' << output(row, col);
        for (int i = 0; i < 4; ++i) response << ' ' << next_correction[i];
        protocol.Write(response.str());
        ++frame_index;
    }
    if (!std::cin.eof()) throw std::runtime_error("Error reading FRAME request stream");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "old_icp_bridge --map /absolute/map.pcd [--scale 3] "
                         "[--geometry directional|off] [--map-leaf 1] [--input-leaf .3] "
                         "[--half-xy 100] [--half-z 30] [--max-xy 1.5] "
                         "[--max-yaw-deg 8] [--alpha .25] [--cpu-reference]\n"
                         "stdin: FRAME \"/absolute/frame.ply\" <16 row-major prior values>\n"
                         "CPU reference is explicitly selected for synthetic tests only; no CUDA fallback.\n";
            return 0;
        }
        return Run(ParseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Old ICP bridge failed: " << error.what() << '\n';
        return 2;
    }
}
