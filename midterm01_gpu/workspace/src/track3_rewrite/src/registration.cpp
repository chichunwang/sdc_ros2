#include "track3_rewrite/core.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace t3 {
bool Pose::finite() const {
  return rotation.allFinite() && translation.allFinite() && std::isfinite(scale) && scale > 0.
      && (rotation.transpose()*rotation-Eigen::Matrix3d::Identity()).norm() < 1e-7
      && std::abs(rotation.determinant()-1.) < 1e-7;
}
Cloud transform(const Cloud& cloud, const Pose& pose) {
  Cloud result; result.reserve(cloud.size());
  for (const auto& p : cloud) {
    const Eigen::Vector3d q = pose.scale*pose.rotation*Eigen::Vector3d(p.x,p.y,p.z)+pose.translation;
    result.push_back({q.x(),q.y(),q.z()});
  }
  return result;
}
Pose compose_camera_metres(const Pose& reference, const Pose& local) {
  Pose result;
  result.rotation = reference.rotation*local.rotation;
  result.translation = reference.rotation*local.translation+reference.translation;
  result.scale = local.scale;  // local target ALREADY uses camera metres
  return result;
}
Vector7 observable_step(const Matrix7& h, const Vector7& b, double length,
                        bool scale, int& rank, Vector7& eigenvalues) {
  if (!h.allFinite() || !b.allFinite() || !std::isfinite(length) || length <= 0.)
    throw std::invalid_argument("Invalid normal equations");
  const int dimensions = scale ? 7 : 6;
  Vector7 units = Vector7::Ones();
  units.tail<4>().setConstant(1./length);
  const Eigen::MatrixXd d = units.head(dimensions).asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solve(d*h.topLeftCorner(dimensions,dimensions)*d);
  if (solve.info() != Eigen::Success) throw std::runtime_error("Small eigensolve failed");
  eigenvalues.setZero(); eigenvalues.head(dimensions) = solve.eigenvalues();
  // Dimensionless condition cutoff derived from sqrt(float64 epsilon), not a
  // claim about calibrated sensor uncertainty or a requirement for rank 7.
  const double threshold = std::max(1e-14, solve.eigenvalues().maxCoeff()*std::sqrt(std::numeric_limits<double>::epsilon()));
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(dimensions);
  rank = 0;
  for (int k=0;k<dimensions;++k) {
    const double value = solve.eigenvalues()[k];
    if (value <= threshold) continue;
    const auto axis = solve.eigenvectors().col(k);
    delta -= axis*(axis.dot(d*b.head(dimensions))/value);
    ++rank;
  }
  Vector7 result = Vector7::Zero(); result.head(dimensions) = d*delta;
  return result;
}
Fit align(const Geometry& target, const Cloud& raw, const Pose& seed,
          Metric metric, bool estimate_scale, const std::vector<double>& radii) {
  if (!seed.finite() || raw.size()<10 || radii.empty()) throw std::invalid_argument("Invalid alignment input");
  Fit result; result.pose = seed;
  double length2 = 0.;
  for (const auto& p : raw) length2 += p.x*p.x+p.y*p.y+p.z*p.z;
  const double length = std::max(1.,seed.scale*std::sqrt(length2/raw.size()));
  bool scale_domain_stop=false;
  for (double radius : radii) {
    if (!std::isfinite(radius) || radius<=0.) throw std::invalid_argument("Invalid search radius");
    const auto start = target.evaluate(raw,result.pose,radius,metric,estimate_scale);
    if (radius==radii.front()) result.initial_cost=start.training_cost;
    result.converged=false;
    scale_domain_stop=false;
    for (int iteration=0;iteration<60;++iteration) {
      ++result.iterations;
      const auto current = target.evaluate(raw,result.pose,radius,metric,estimate_scale);
      if (current.training_support<7) { result.reason="insufficient_training_correspondences"; break; }
      Vector7 step = observable_step(current.h,current.gradient,length,estimate_scale,result.rank,result.eigenvalues);
      if (result.rank==0) { result.reason="no_observed_direction"; break; }
      const double factor=std::min({1.,radius/std::max(step.head<3>().norm(),1e-20),
          .08726646259971647/std::max(step.segment<3>(3).norm(),1e-20),
          .1/std::max(std::abs(step[6]),1e-20)});
      step*=factor;
      if (step.head<3>().norm()<1e-5 && step.segment<3>(3).norm()<1e-6 && std::abs(step[6])<1e-6) {
        result.converged=true; result.reason="stationary_observed_modes"; break;
      }
      bool moved=false;
      bool scale_proposal_blocked=false;
      for (double fraction : {1.,.5,.25,.125,.0625,.03125}) {
        Pose proposal=result.pose;
        proposal.translation+=fraction*step.head<3>();
        const Eigen::Vector3d rotation=fraction*step.segment<3>(3);
        if (rotation.norm()>0.) proposal.rotation=Eigen::AngleAxisd(rotation.norm(),rotation.normalized()).toRotationMatrix()*proposal.rotation;
        if (estimate_scale) proposal.scale*=std::exp(fraction*step[6]);
        if (!proposal.finite()) continue;
        // One-way nearest-neighbor Sim3 has a degenerate scale->0 minimum.
        // Use an explicit factor-two seed domain, separate from the per-step
        // trust cap. Bootstrap has an independent ground-derived scale seed.
        if (estimate_scale && (proposal.scale<=seed.scale*.5 || proposal.scale>=seed.scale*2.)) {scale_proposal_blocked=true;continue;}
        const auto next=target.evaluate(raw,proposal,radius,metric,estimate_scale);
        if (next.training_cost<current.training_cost-1e-12) {
          result.pose=proposal; moved=true; result.reason="training_descent"; break;
        }
      }
      if (!moved) { scale_domain_stop=scale_proposal_blocked;result.reason=scale_domain_stop?"scale_domain_active":"no_training_descent";break; }
    }
  }
  const auto before=target.evaluate(raw,seed,radii.back(),metric,estimate_scale);
  const auto after=target.evaluate(raw,result.pose,radii.back(),metric,estimate_scale);
  observable_step(after.h,after.gradient,length,estimate_scale,result.rank,result.eigenvalues);
  result.initial_cost=before.training_cost;
  result.final_cost=after.training_cost;
  result.validation_before=before.validation_cost; result.validation_after=after.validation_cost;
  result.training_support=after.training_support; result.validation_support=after.validation_support;
  // Held-out can veto this fit, never select a different candidate. No arbitrary
  // absolute residual/rank/visibility/consensus or post-fit motion gate is copied.
  const bool scale_interior=!estimate_scale || (result.pose.scale>seed.scale*(.5+1e-6) && result.pose.scale<seed.scale*(2.-1e-6));
  const bool dimension_supported=metric==Metric::Point && estimate_scale ? result.rank==7 : result.rank>0;
  result.usable=result.pose.finite() && dimension_supported && scale_interior && !scale_domain_stop && after.training_support>=7
      && after.validation_support>=3 && after.training_cost<=before.training_cost+1e-10
      && after.validation_cost<=before.validation_cost+1e-10;
  if (!result.usable) result.reason += ";heldout_or_support_not_improved";
  if(!scale_interior)result.reason+=";scale_domain_boundary";
  if(scale_domain_stop)result.reason+=";scale_domain_exhausted";
  if(!dimension_supported)result.reason+=";required_transform_directions_unobserved";
  synchronize();
  return result;
}
void Memory::commit(const Cloud& raw, std::int64_t ns, const Pose* map, const Pose* pair) {
  if (ns<=0 || ns<=output_ns) throw std::invalid_argument("Only current increasing nanosecond timestamps allowed");
  if (!reference && !map) throw std::runtime_error("First frame has no supported map initialization");
  if ((map && !map->finite()) || (pair && !pair->finite())) throw std::invalid_argument("Invalid state update");
  if (map || pair) {
    published=map ? *map : *pair;
    reference=std::make_unique<Reference>(Reference{published,raw,ns});
  }
  if (map) map_ns=ns;
  output_ns=ns;
}
}  // namespace t3
