// Synthetic numerical oracle only. Never used by the localization runner.
#include "localization/sensor_guided.h"
#include <Eigen/Core>
#include <memory>
extern "C" int original_guided(const float* src,int ns,const float* dst,int nt,
 const float* pose,const double* previous,double alpha,double* result) {
 try {
  using Cloud=pcl::PointCloud<pcl::PointXYZ>;
  auto s=std::make_shared<Cloud>(),t=std::make_shared<Cloud>();
  for(int i=0;i<ns;i++)s->push_back(pcl::PointXYZ(src[3*i],src[3*i+1],src[3*i+2]));
  for(int i=0;i<nt;i++)t->push_back(pcl::PointXYZ(dst[3*i],dst[3*i+1],dst[3*i+2]));
  Eigen::Matrix4f prior;
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)prior(i,j)=pose[4*i+j];
  Eigen::Vector4d prev;
  for(int i=0;i<4;i++)prev[i]=previous[i];
  Eigen::Matrix4f out;Eigen::Vector4d correction;localization::GuidedMetrics m;
  localization::GeometryConfig cfg;cfg.mode=localization::GeometryMode::Directional;
  bool accepted=localization::RefineSensorGuided(s,t,prior,prior,false,prev,alpha,1.5,8*acos(-1.)/180.,out,correction,m,false,cfg);
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)result[4*i+j]=out(i,j);
  for(int i=0;i<4;i++)result[16+i]=correction[i];
  result[20]=m.prior_cost;result[21]=m.final_cost;result[22]=m.prior_rmse;
  result[23]=m.final_rmse;result[24]=m.inliers;result[25]=m.candidate_reliable_points;
  result[26]=m.candidate_unique_targets;result[27]=m.iterations;
  return accepted?1:0;
 } catch (...) { return -1; }
}
