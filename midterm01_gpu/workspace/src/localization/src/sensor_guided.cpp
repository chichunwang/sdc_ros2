#include "localization/sensor_guided.h"
#include <cuda_ndt_backend/nearest_neighbor.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <pcl/kdtree/kdtree_flann.h>

namespace localization {
namespace {
using V4 = Eigen::Vector4d;
struct Evaluation {
    double cost = 0.0, mse = 0.0;
    int inliers = 0;
    Eigen::Matrix4d hessian = Eigen::Matrix4d::Zero();
    V4 gradient = V4::Zero();
};
double Wrap(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }
}  // namespace

bool RefineSensorGuided(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                        const Eigen::Matrix4f& prior, const Eigen::Matrix4f& cuda_seed,
                        bool cuda_valid, const Eigen::Vector4d& previous_correction,
                        double smoothing_alpha, double max_xy, double max_yaw,
                        Eigen::Matrix4f& output, Eigen::Vector4d& correction,
                        GuidedMetrics& metrics, bool use_cuda_nn) {
    const Eigen::Matrix4f saved_seed = cuda_seed;
    correction.setZero();
    output = prior;
    if (source->size() < 10 || target->size() < 10) return false;
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    if (!use_cuda_nn) tree.setInputCloud(target);
    const Eigen::Matrix3d rotation_prior = prior.block<3, 3>(0, 0).cast<double>();
    const Eigen::Vector3d translation_prior = prior.block<3, 1>(0, 3).cast<double>();
    std::unique_ptr<cuda_ndt_backend::NearestNeighbor> gpu_nn;
    std::vector<int> gpu_to_target;
    if (use_cuda_nn) {
        Eigen::Vector3d low = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d high = -low;
        double radius = 0.0;
        for (const auto& point : *source) {
            const Eigen::Vector3d rotated = rotation_prior * point.getVector3fMap().cast<double>();
            low = low.cwiseMin(rotated);
            high = high.cwiseMax(rotated);
            radius = std::max(radius, rotated.head<2>().norm());
        }
        // Any neighbor capable of influencing the 2 m capped objective lies
        // within this box for every allowed xyz/yaw correction. The angular
        // arc bound radius*angle covers the entire bounded optimization path.
        const Eigen::Vector3d margin(max_xy + radius * max_yaw + 2.01,
                                      max_xy + radius * max_yaw + 2.01, 3.51);
        low -= margin;
        high += margin;
        std::vector<float> local_target;
        local_target.reserve(target->size() * 3);
        for (std::size_t index = 0; index < target->size(); ++index) {
            const auto& point = (*target)[index];
            const Eigen::Vector3f local = (point.getVector3fMap().cast<double>() - translation_prior).cast<float>();
            if ((local.cast<double>().array() < low.array()).any() || (local.cast<double>().array() > high.array()).any()) continue;
            local_target.insert(local_target.end(), {local.x(), local.y(), local.z()});
            gpu_to_target.push_back(static_cast<int>(index));
        }
        if (local_target.empty()) {
            // No map point can enter the inlier region. This is an explicit
            // sensor-only result, not a failed CUDA call or CPU fallback.
            metrics.prior_cost = metrics.final_cost = (0.5 * (2.0 - 0.25)) / (0.3 * 0.3);
            metrics.prior_rmse = metrics.final_rmse = 2.0;
            return false;
        }
        gpu_nn = std::make_unique<cuda_ndt_backend::NearestNeighbor>(local_target, static_cast<int>(source->size()));
    }
    // Standard deviations: GNSS XY 0.5 m; carried height 1 m; IMU yaw 3 deg.
    const V4 precision(4.0, 4.0, 1.0, 1.0 / std::pow(3.0 * M_PI / 180.0, 2));
    constexpr double huber = 0.5, cutoff = 2.0, cloud_precision = 1.0 / (0.3 * 0.3);
    const double normalize = cloud_precision / static_cast<double>(source->size());
    const auto valid = [&](const V4& x) {
        return x.allFinite() && x.head<2>().norm() <= max_xy && std::abs(x[2]) <= 1.5 && std::abs(x[3]) <= max_yaw;
    };
    const auto evaluate = [&](const V4& x, bool derivatives) {
        Evaluation e;
        e.cost = 0.5 * x.dot(precision.cwiseProduct(x));
        e.gradient = precision.cwiseProduct(x);
        e.hessian = precision.asDiagonal();
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(x[3], Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation_prior;
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        std::vector<int> gpu_indices;
        if (gpu_nn) {
            std::vector<float> queries;
            queries.reserve(source->size() * 3);
            for (const auto& point : *source) {
                const Eigen::Vector3f local = (rotation * point.getVector3fMap().cast<double>() + x.head<3>()).cast<float>();
                queries.insert(queries.end(), {local.x(), local.y(), local.z()});
            }
            gpu_nn->Query(queries, gpu_indices);
        }
        std::size_t source_index = 0;
        for (const auto& point : *source) {
            const Eigen::Vector3d rotated = rotation * point.getVector3fMap().cast<double>();
            const Eigen::Vector3d transformed = rotated + translation_prior + x.head<3>();
            pcl::PointXYZ query;
            query.getVector3fMap() = transformed.cast<float>();
            if (gpu_nn) {
                const int local_index = gpu_indices[source_index++];
                if (local_index < 0 || static_cast<std::size_t>(local_index) >= gpu_to_target.size())
                    throw std::runtime_error("CUDA NN returned an invalid local index");
                indices[0] = gpu_to_target[local_index];
            }
            else if (tree.nearestKSearch(query, 1, indices, distances) != 1) {
                e.cost = std::numeric_limits<double>::infinity();
                return e;
            }
            if (indices[0] < 0 || static_cast<std::size_t>(indices[0]) >= target->size())
                throw std::runtime_error("CUDA NN returned an invalid map index");
            const Eigen::Vector3d residual = transformed - (*target)[indices[0]].getVector3fMap().cast<double>();
            const double length = residual.norm();
            // Constant outlier cost prevents reducing the objective just by losing matches.
            const double clipped = std::min(length, cutoff);
            const double loss = clipped <= huber ? 0.5 * clipped * clipped : huber * (clipped - 0.5 * huber);
            e.cost += normalize * loss;
            e.mse += clipped * clipped / static_cast<double>(source->size());
            if (length > cutoff) continue;
            ++e.inliers;
            if (!derivatives) continue;
            Eigen::Matrix<double, 3, 4> jacobian = Eigen::Matrix<double, 3, 4>::Zero();
            jacobian.block<3, 3>(0, 0).setIdentity();
            jacobian.col(3) = Eigen::Vector3d(-rotated.y(), rotated.x(), 0.0);
            const double weight = normalize * std::min(1.0, huber / std::max(length, 1e-12));
            e.hessian.noalias() += weight * jacobian.transpose() * jacobian;
            e.gradient.noalias() += weight * jacobian.transpose() * residual;
        }
        return e;
    };
    const Evaluation baseline = evaluate(V4::Zero(), false);
    metrics.prior_cost = baseline.cost;
    metrics.prior_rmse = std::sqrt(baseline.mse);
    V4 x = V4::Zero();
    const V4 carried = previous_correction * 0.8;
    if (valid(carried) && evaluate(carried, false).cost < baseline.cost) x = carried;
    if (cuda_valid) {
        V4 proposal;
        proposal.head<3>() = (saved_seed.block<3, 1>(0, 3) - prior.block<3, 1>(0, 3)).cast<double>();
        proposal[3] = Wrap(std::atan2(saved_seed(1, 0), saved_seed(0, 0)) - std::atan2(prior(1, 0), prior(0, 0)));
        // CUDA roll/pitch are deliberately not propagated into the road-vehicle state.
        if (valid(proposal) && evaluate(proposal, false).cost < evaluate(x, false).cost) {
            x = proposal;
            metrics.used_cuda_seed = true;
        }
    }
    for (int iteration = 0; iteration < 15; ++iteration) {
        const Evaluation current = evaluate(x, true);
        if (current.inliers < 10 || !std::isfinite(current.cost)) break;
        V4 step = -current.hessian.ldlt().solve(current.gradient);
        if (!step.allFinite()) break;
        const double scale = std::min({1.0, 0.25 / std::max(step.head<3>().norm(), 1e-12),
                                       (2.0 * M_PI / 180) / std::max(std::abs(step[3]), 1e-12)});
        step *= scale;
        bool accepted = false;
        for (int search = 0; search < 8; ++search) {
            const V4 next = x + step;
            if (valid(next) && evaluate(next, false).cost < current.cost) {
                x = next;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        metrics.iterations = iteration + 1;
        if (!accepted || (step.head<3>().norm() < 1e-3 && std::abs(step[3]) < 1e-4)) break;
    }
    V4 smoothed = (1 - smoothing_alpha) * previous_correction + smoothing_alpha * x;
    // An old correction must never bypass the current sensor bound or objective.
    Evaluation final = valid(smoothed) ? evaluate(smoothed, false) : baseline;
    if (!valid(smoothed) || final.cost > baseline.cost) {
        smoothed = x;
        final = evaluate(x, false);
    }
    if (final.inliers < 10 || !std::isfinite(final.cost) || final.cost > baseline.cost + 1e-9) {
        metrics.final_cost = baseline.cost;
        metrics.final_rmse = metrics.prior_rmse;
        metrics.inliers = baseline.inliers;
        return false;
    }
    correction = smoothed;
    output.block<3, 3>(0, 0) = (Eigen::AngleAxisd(correction[3], Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation_prior).cast<float>();
    output.block<3, 1>(0, 3) = (translation_prior + correction.head<3>()).cast<float>();
    metrics.final_cost = final.cost;
    metrics.final_rmse = std::sqrt(final.mse);
    metrics.inliers = final.inliers;
    return true;
}
}  // namespace localization
