#include "localization/gicp_metric.h"
#include "localization/sensor_guided.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

void Require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}

int main(int argc, char**) {
    try {
        using namespace localization;
        const auto rotation = [](double yaw) -> Eigen::Matrix3d {
            return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        };
        const Eigen::Vector3d normal = Eigen::Vector3d(1, 2, 3).normalized();
        const auto identical = MakeGicpMetric(normal, normal, Eigen::Matrix3d::Identity(), .05);
        const Eigen::Matrix3d expected = .05 * Eigen::Matrix3d::Identity() + .95 * normal * normal.transpose();
        Require(identical.weight.isApprox(expected, 1e-12), "Agreeing normals must recover directional metric");
        const Eigen::Vector3d source_normal = Eigen::Vector3d(-2, 1, 1).normalized();
        const auto metric = MakeGicpMetric(normal, source_normal, rotation(.24), .05);
        Require(MakeGicpMetric(-normal, -source_normal, rotation(.24), .05).weight.isApprox(metric.weight, 1e-12),
                "Normal signs must not alter GICP metric");
        Require(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(metric.weight).eigenvalues().minCoeff() > 0,
                "Metric must remain positive definite under disagreeing normals");
        const double h = 1e-6;
        const Eigen::Matrix3d difference = (MakeGicpMetric(normal, source_normal, rotation(.24+h), .05).weight -
            MakeGicpMetric(normal, source_normal, rotation(.24-h), .05).weight) / (2*h);
        Require(difference.isApprox(metric.yaw_derivative, 1e-7), "Rotating covariance yaw derivative is wrong");
        // Verify the complete Mahalanobis + Huber yaw derivative, including
        // both the transformed source point and the rotating covariance.
        for (double distance : {.2, 3.}) {
            const Eigen::Vector3d point(distance, -.7*distance, .2*distance), target(-.4, .1, -.3);
            const auto cost = [&](double yaw) {
                const auto m = MakeGicpMetric(normal, source_normal, rotation(yaw), .05);
                const Eigen::Vector3d r = rotation(yaw)*point-target;
                const double d = std::sqrt(r.dot(m.weight*r));
                return d <= .5 ? .5*d*d : .5*(d-.25);
            };
            const Eigen::Vector3d p = rotation(.24)*point, r = p-target, j(-p.y(), p.x(), 0);
            const double length = std::sqrt(r.dot(metric.weight*r));
            const double analytic = std::min(1., .5/std::max(length, 1e-12)) *
                (j.dot(metric.weight*r) + .5*r.dot(metric.yaw_derivative*r));
            Require(std::abs((cost(.24+h)-cost(.24-h))/(2*h)-analytic) < 1e-7,
                    "Huber GICP objective/gradient mismatch");
        }
        GeometryConfig config; config.mode = GeometryMode::Gicp;
        GeometryConfig invalid = config; invalid.tangent_weight = 0;
        bool rejected = false;
        try { invalid.Validate(); } catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected, "Singular GICP covariance must fail validation");

        using Cloud = pcl::PointCloud<pcl::PointXYZ>;
        auto source = std::make_shared<Cloud>(), target = std::make_shared<Cloud>();
        for (int i = -10; i <= 10; ++i) for (int j = -10; j <= 10; ++j) {
            source->push_back(pcl::PointXYZ(i*.5f, j*.5f, 0));
            source->push_back(pcl::PointXYZ(6, i*.5f, 2+j*.5f));
            source->push_back(pcl::PointXYZ(i*.5f, 7, 2+j*.5f));
        }
        for (const auto& p : *source) target->push_back(pcl::PointXYZ(p.x-4800, p.y+2274, p.z+110));
        Eigen::Matrix4f prior = Eigen::Matrix4f::Identity();
        prior.block<3,3>(0,0) = rotation(.02).cast<float>();
        prior.block<3,1>(0,3) = Eigen::Vector3f(-4799.85f, 2273.9f, 110.2f);
        Eigen::Matrix4f output; Eigen::Vector4d correction; GuidedMetrics metrics;
        const bool cuda = argc > 1;
        Require(RefineSensorGuided(source, target, prior, prior, false, Eigen::Vector4d::Zero(),
            1, 1.5, .14, output, correction, metrics, cuda, config), "Known-geometry GICP update failed");
        Require(metrics.final_cost < metrics.prior_cost && metrics.source_reliable_points > 100,
                "Known geometry must reduce objective using actual source planes");
        Require(correction[0] < 0 && correction[1] > 0 && correction[2] < 0 && correction[3] < 0,
                "Known pose perturbation must be corrected in the right directions");
        Require(correction.head<2>().norm() <= 1.5 && std::abs(correction[3]) <= .14,
                "GICP bypassed prior bounds");
        if (cuda) {
            Eigen::Matrix4f cpu; Eigen::Vector4d delta; GuidedMetrics diagnostics;
            Require(RefineSensorGuided(source, target, prior, prior, false, Eigen::Vector4d::Zero(),
                1, 1.5, .14, cpu, delta, diagnostics, false, config), "CPU reference failed");
            Require(cpu.isApprox(output, 1e-8) && delta.isApprox(correction, 1e-5), "CPU/CUDA GICP mismatch");
        }
        auto line = std::make_shared<Cloud>();
        for (int i=0; i<30; ++i) line->push_back(pcl::PointXYZ(i*.2f, 0, 0));
        Require(!RefineSensorGuided(line, target, prior, prior, false, Eigen::Vector4d::Zero(),
            1, 1.5, .14, output, correction, metrics, cuda, config), "Unsupported source line silently fell back to directional ICP");
        Require(output.isApprox(prior) && correction.isZero() && metrics.source_reliable_points == 0,
                "Unsupported GICP must retain the current sensor prior");
        std::cout << "GICP metric, signs, positive definiteness, finite-difference gradients, bounds, source support and known geometry passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
