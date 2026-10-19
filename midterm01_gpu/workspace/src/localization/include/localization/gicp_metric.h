#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

namespace localization {
struct GicpMetric {
    Eigen::Matrix3d weight;
    Eigen::Matrix3d yaw_derivative;
};

// Regularized surface geometry, not calibrated measurement covariance. Each
// cloud contributes half variance; agreeing normals recover the scored metric.
inline GicpMetric MakeGicpMetric(const Eigen::Vector3d& target_normal,
                                const Eigen::Vector3d& source_normal,
                                const Eigen::Matrix3d& rotation, double tangent_weight) {
    if (!std::isfinite(tangent_weight) || tangent_weight <= 0 || tangent_weight > 1)
        throw std::invalid_argument("GICP requires tangent weight in (0,1]");
    const auto covariance = [tangent_weight](const Eigen::Vector3d& normal) -> Eigen::Matrix3d {
        return Eigen::Matrix3d::Identity() / tangent_weight +
            (1.0 - 1.0 / tangent_weight) * normal * normal.transpose();
    };
    const Eigen::Matrix3d rotated = rotation * covariance(source_normal) * rotation.transpose();
    const Eigen::Matrix3d combined = 0.5 * (covariance(target_normal) + rotated);
    Eigen::LDLT<Eigen::Matrix3d> solver(combined);
    if (solver.info() != Eigen::Success || !solver.isPositive())
        throw std::runtime_error("Nonpositive GICP surface covariance");
    GicpMetric result;
    result.weight = solver.solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d generator = Eigen::Matrix3d::Zero();
    generator(0, 1) = -1; generator(1, 0) = 1;
    const Eigen::Matrix3d derivative = .5 * (generator * rotated - rotated * generator);
    result.yaw_derivative = -result.weight * derivative * result.weight;
    if (!result.weight.allFinite() || !result.yaw_derivative.allFinite())
        throw std::runtime_error("Nonfinite GICP metric");
    return result;
}
}  // namespace localization
