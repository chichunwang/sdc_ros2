#include "localization/sensor_guided.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char**) {
    try {
        using Cloud = pcl::PointCloud<pcl::PointXYZ>;
        const bool gpu = argc > 1;
        auto source = std::make_shared<Cloud>();
        auto target = std::make_shared<Cloud>();
        for (int i = 0; i < 120; ++i) {
            pcl::PointXYZ p;
            p.getVector3fMap() = Eigen::Vector3f(.3f * (i % 12), .3f * (i / 12), .1f * std::sin(i));
            source->push_back(p);
            p.x -= 4800; p.y += 2274; p.z += 110;
            target->push_back(p);
        }
        const Cloud original = *source;
        Eigen::Matrix4f initial = Eigen::Matrix4f::Identity();
        initial.block<3, 1>(0, 3) = Eigen::Vector3f(-4800, 2274, 110);
        Eigen::Matrix4f legacy = initial, anchored = initial;
        Eigen::Vector4d legacy_delta = Eigen::Vector4d::Zero(), anchored_delta = legacy_delta;
        localization::HeightConstraint constraint;
        constraint.anchored = true; constraint.anchor_z = 110;
        // Deliberately biased depth is NOT vehicle motion. Incremental scan
        // updates can integrate it; an absolute nominal-height model must not.
        for (int frame = 0; frame < 150; ++frame) {
            *source = original;
            for (auto& p : *source) p.z += .025f * frame;
            for (int mode = 0; mode < 2; ++mode) {
                auto& pose = mode ? anchored : legacy;
                auto& previous = mode ? anchored_delta : legacy_delta;
                Eigen::Matrix4f output;
                Eigen::Vector4d delta;
                localization::GuidedMetrics metrics;
                localization::RefineSensorGuided(source, target, pose, pose, false,
                    previous, .25, 1.5, 8 * M_PI / 180, output, delta, metrics, gpu,
                    localization::GeometryConfig(), mode ? constraint : localization::HeightConstraint());
                pose = output; previous = delta;
                Require(metrics.final_cost <= metrics.prior_cost + 1e-9, "Accepted height objective increased");
                if (mode) Require(std::abs(pose(2, 3) - 110) <= 1.50002, "Absolute height bound violated over time");
            }
        }
        Require(110 - legacy(2, 3) > 3, "Adversarial scene did not exercise incremental drift");
        Require(110 - anchored(2, 3) < 1.50002, "Anchored height integrated biased depth");
        // A changing sensor altitude preserves the relative offset. It does
        // not freeze world Z, or apply yesterday's dz as a second displacement.
        Require(std::abs(constraint.Predict(112, 109, 110) - 111) < 1e-12, "Nominal altitude change must propagate");
        // Opposite ends of the absolute band can differ by >1.5 m in ONE
        // update. In CUDA this also exercises the expanded target-search box.
        *source = original;
        for (auto& p : *source) p.z += .5f;
        Eigen::Matrix4f prior = initial, output;
        prior(2, 3) += 1.4f;
        Eigen::Vector4d delta;
        localization::GuidedMetrics metrics;
        Require(localization::RefineSensorGuided(source, target, prior, prior, false,
            Eigen::Vector4d::Zero(), 1, 1.5, 8 * M_PI / 180, output, delta, metrics, gpu,
            localization::GeometryConfig(), constraint), "Opposite-band map update rejected");
        Require(delta[2] < -1.5 && output(2, 3) >= 108.49998, "Bound applied to increment instead of total offset");
        // A rejected update retains the predicted height, current XY/attitude,
        // and zero correction; it cannot advance the old vertical increment.
        for (auto& p : *target) p.x += 500;
        Require(!localization::RefineSensorGuided(source, target, prior, prior, false,
            Eigen::Vector4d::Constant(.2), .25, 1.5, 8 * M_PI / 180, output, delta, metrics, gpu,
            localization::GeometryConfig(), constraint), "No-overlap height update must reject");
        Require(output.isApprox(prior) && delta.isZero(), "Rejected height must retain current prior");
        std::cout << "Height bias accumulation, absolute bound, moving altitude, opposite-band search, and no-overlap fallback passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
