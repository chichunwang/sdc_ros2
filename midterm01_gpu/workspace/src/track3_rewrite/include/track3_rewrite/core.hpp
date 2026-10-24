#pragma once
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace t3 {
struct Point { double x, y, z; };
using Cloud = std::vector<Point>;
using Matrix7 = Eigen::Matrix<double, 7, 7>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
struct Pose {
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  double scale = 3.;
  bool finite() const;
};
struct Evaluation {
  Matrix7 h = Matrix7::Zero();
  Vector7 gradient = Vector7::Zero();
  double training_cost = 1., validation_cost = 1.;
  int training_support = 0, validation_support = 0;
};
enum class Metric { Point, Plane };
class Geometry {
public:
  Geometry(const Cloud& target, double cell_size, bool normals);
  ~Geometry();
  Geometry(const Geometry&) = delete;
  Evaluation evaluate(const Cloud& source, const Pose& pose, double radius,
                      Metric metric, bool estimate_scale) const;
  std::vector<std::int64_t> nearest(const Cloud& queries, double radius) const;
  std::vector<Point> normals() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class StaticMap {
public:
  explicit StaticMap(const std::string& npy_path);
  ~StaticMap();
  Cloud crop(const Eigen::Vector3d& center, double xy, double z, double leaf) const;
  std::size_t size() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
Cloud voxel(const Cloud& cloud, double leaf);
Cloud transform(const Cloud& cloud, const Pose& pose);
struct Fit {
  Pose pose;
  bool usable = false;
  bool converged = false;
  int rank = 0, iterations = 0;
  double initial_cost = 1., final_cost = 1., validation_before = 1., validation_after = 1.;
  int training_support = 0, validation_support = 0;
  std::string reason = "not_run";
  Vector7 eigenvalues = Vector7::Zero();
};
// A data-only eigensolve. Unobserved modes receive exactly zero correction.
Vector7 observable_step(const Matrix7& h, const Vector7& gradient, double length,
                        bool scale, int& rank, Vector7& eigenvalues);
Fit align(const Geometry& target, const Cloud& raw, const Pose& seed,
          Metric metric, bool estimate_scale, const std::vector<double>& radii);
Pose compose_camera_metres(const Pose& map_reference, const Pose& local);
struct Reference { Pose pose; Cloud raw; std::int64_t timestamp_ns = 0; };
struct Memory {
  Pose published;
  std::int64_t output_ns = 0, map_ns = 0;
  std::unique_ptr<Reference> reference;
  void commit(const Cloud& raw, std::int64_t ns, const Pose* map, const Pose* pair);
};
void synchronize();
std::array<std::size_t, 2> memory_info();
}  // namespace t3
