// Data-flow reference: midterm01/workspace/src/localization/src/localization.cpp.
// This implementation does not link the old application, its solver, or Python gates.
#include "track3_rewrite/core.hpp"
#include <pcl/io/ply_io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <openssl/evp.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point begin){return std::chrono::duration<double>(Clock::now()-begin).count();}
volatile std::sig_atomic_t interrupted=0;
void signal_handler(int value){interrupted=value;}
struct Frame{int idx;std::int64_t ns;std::string path;};
std::string quote(const std::string& s) {
  std::ostringstream out;out<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c<32)out<<' ';else out<<c;}out<<'"';return out.str();
}
std::string file_sha256(const std::string& path) {
  std::ifstream stream(path,std::ios::binary);
  if(!stream)throw std::runtime_error("Cannot hash current source: "+path);
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  if(!context || EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 init failed");
  std::array<char,65536> buffer;
  while(stream.read(buffer.data(),buffer.size()) || stream.gcount())
    if(EVP_DigestUpdate(context.get(),buffer.data(),stream.gcount())!=1)throw std::runtime_error("SHA256 update failed");
  if(!stream.eof())throw std::runtime_error("Source read failed during SHA256");
  unsigned char digest[EVP_MAX_MD_SIZE];unsigned length=0;
  if(EVP_DigestFinal_ex(context.get(),digest,&length)!=1)throw std::runtime_error("SHA256 final failed");
  std::ostringstream out;for(unsigned i=0;i<length;++i)out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(digest[i]);return out.str();
}
t3::Pose read_initial(const std::string& path,double scale) {
  std::ifstream stream(path);std::string text((std::istreambuf_iterator<char>(stream)),{});
  std::replace(text.begin(),text.end(),',',' ');std::istringstream values(text);Eigen::Matrix4d matrix;
  for(int a=0;a<4;++a)for(int b=0;b<4;++b)if(!(values>>matrix(a,b)))throw std::runtime_error("Invalid initial pose");
  double extra;if(values>>extra)throw std::runtime_error("Extra initial pose values");
  if(!matrix.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),1e-9))throw std::runtime_error("Invalid initial pose bottom row");
  t3::Pose pose;pose.rotation=matrix.topLeftCorner<3,3>();pose.translation=matrix.topRightCorner<3,1>();pose.scale=scale;
  if(!pose.finite())throw std::runtime_error("Expected proper T_map_source and positive scale");return pose;
}
t3::Cloud read_ply(const std::string& path) {
  pcl::PointCloud<pcl::PointXYZ> input;
  if(pcl::io::loadPLYFile(path,input)!=0)throw std::runtime_error("Could not read PLY: "+path);
  t3::Cloud cloud;cloud.reserve(input.size());
  for(auto p:input){if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::runtime_error("Nonfinite source point");cloud.push_back({p.x,p.y,p.z});}
  if(cloud.size()<10)throw std::runtime_error("Insufficient source points");return cloud;
}
std::optional<Eigen::Vector4d> plane(const t3::Cloud& points,double distance,bool horizontal) {
  if(points.size()<10)return std::nullopt;
  auto cloud=std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  for(auto p:points)cloud->push_back(pcl::PointXYZ(float(p.x),float(p.y),float(p.z)));
  pcl::SACSegmentation<pcl::PointXYZ> segment(false);
  segment.setModelType(horizontal?pcl::SACMODEL_PERPENDICULAR_PLANE:pcl::SACMODEL_PLANE);
  segment.setMethodType(pcl::SAC_RANSAC);segment.setOptimizeCoefficients(true);
  segment.setMaxIterations(100);segment.setDistanceThreshold(distance);
  segment.setAxis(Eigen::Vector3f::UnitZ());segment.setEpsAngle(.35);segment.setInputCloud(cloud);
  pcl::PointIndices inliers;pcl::ModelCoefficients coefficients;segment.segment(inliers,coefficients);
  if(coefficients.values.size()!=4 || inliers.indices.size()<10)return std::nullopt;
  Eigen::Vector4d result;for(int i=0;i<4;++i)result[i]=coefficients.values[i];
  result/=result.head<3>().norm();if(result[2]<0.)result=-result;
  return result.allFinite()?std::optional<Eigen::Vector4d>(result):std::nullopt;
}
std::optional<t3::Pose> ground_seed(const t3::Cloud& raw,const t3::Cloud& map,const t3::Pose& initial,double raw_leaf,double map_leaf) {
  // One small startup hypothesis. Training only; not a vehicle-height calibration.
  std::vector<double> heights;for(std::size_t i=0;i<raw.size();++i)if(i%5)heights.push_back(raw[i].z);
  std::sort(heights.begin(),heights.end());if(heights.empty())return std::nullopt;
  const double lower=heights[heights.size()/2];
  t3::Cloud source,ground;
  for(std::size_t i=0;i<raw.size();++i)if(i%5 && raw[i].z<=lower)source.push_back(raw[i]);
  for(auto p:map)if(std::abs(p.x-initial.translation.x())<10. && std::abs(p.y-initial.translation.y())<10. && p.z<initial.translation.z())ground.push_back(p);
  auto a=plane(source,raw_leaf/2.,false),b=plane(ground,map_leaf/2.,true);
  if(!a || !b || (*a)[3]<=0.)return std::nullopt;
  const double height=b->head<3>().dot(initial.translation)+(*b)[3];
  if(height<=0.)return std::nullopt;
  t3::Pose result=initial;result.scale=height/(*a)[3];
  const Eigen::Vector3d from=initial.rotation*a->head<3>();
  result.rotation=Eigen::Quaterniond::FromTwoVectors(from,b->head<3>()).toRotationMatrix()*initial.rotation;
  return result.finite()?std::optional<t3::Pose>(result):std::nullopt;
}
void pose_json(std::ostream& out,const t3::Pose& p) {
  out<<'[';for(int a=0;a<4;++a){if(a)out<<',';out<<'[';for(int b=0;b<4;++b){if(b)out<<',';out<<(a==3?(b==3?1.:0.):b==3?p.translation[a]:p.rotation(a,b));}out<<']';}out<<']';
}
void fit_json(std::ostream& out,const t3::Fit& f) {
  out<<"{\"usable\":"<<(f.usable?"true":"false")<<",\"reason\":"<<quote(f.reason)
     <<",\"rank\":"<<f.rank<<",\"iterations\":"<<f.iterations<<",\"converged\":"<<(f.converged?"true":"false")
     <<",\"training_cost\":"<<f.final_cost<<",\"training_before\":"<<f.initial_cost<<",\"validation_before\":"<<f.validation_before<<",\"validation_after\":"<<f.validation_after
     <<",\"training_support\":"<<f.training_support<<",\"validation_support\":"<<f.validation_support<<",\"scale\":"<<f.pose.scale<<",\"pose\":";
  pose_json(out,f.pose);out<<",\"eigenvalues\":[";for(int i=0;i<7;++i){if(i)out<<',';out<<f.eigenvalues[i];}out<<"]}";
}
} // namespace
int main(int argc,char** argv) {
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2){if(i+1>=argc){std::cerr<<"Expected option/value pairs\n";return 2;}args[argv[i]]=argv[i+1];}
  if(!args.count("--out")){std::cerr<<"Missing --out\n";return 2;}
  const auto out=std::filesystem::path(args["--out"]);
  if(!std::filesystem::is_directory(out)){std::cerr<<"Failed: output directory must already exist\n";return 2;}
  for(const char* name:{"result.txt","diagnostic.csv","timings.csv","frames.csv","observations.jsonl","native_status.json","native_pid.txt","registration_candidates.jsonl","source_frames.jsonl"}) {
    if(std::filesystem::exists(out/name)||std::filesystem::is_symlink(out/name)){std::cerr<<"Failed: refusing existing output "<<name<<'\n';return 2;}
  }
  std::ofstream(out/"native_pid.txt")<<getpid()<<'\n';
  const auto started=Clock::now();int processed=0,accepted=0,pairs=0,lost=0,failed_idx=-1,code=1;
  std::string stage="arguments",failure;std::int64_t full_rank_map_ns=0;
  std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);
  try {
    const double initial_scale=args.count("--initial-scale")?std::stod(args["--initial-scale"]):3.;
    const double raw_leaf=.1,map_leaf=1.; // original template .3m / nominal 3, and 1m map leaf
    t3::Memory memory;memory.published=read_initial(args.at("--init"),initial_scale);
    std::ifstream list(args.at("--frames"));std::vector<Frame> frames;std::string line;
    while(std::getline(list,line)) {
      std::istringstream row(line);Frame frame;if(!(row>>frame.idx>>frame.ns>>frame.path))throw std::runtime_error("Invalid frame manifest");
      if(frame.idx!=int(frames.size()) || frame.ns<=0 || (!frames.empty() && frame.ns<=frames.back().ns))throw std::runtime_error("Invalid frame order");
      frames.push_back(frame);
    }
    if(frames.empty())throw std::runtime_error("Empty frame manifest");
    stage="map_preparation";t3::StaticMap map(args.at("--map"));
    Eigen::Vector3d center=memory.published.translation;
    auto tile=map.crop(center,100.,30.,map_leaf);
    auto scene=std::make_unique<t3::Geometry>(tile,map_leaf,true);
    std::ofstream matrices(out/"result.txt"),csv(out/"diagnostic.csv"),timings(out/"timings.csv"),manifest(out/"frames.csv"),observations(out/"observations.jsonl"),fit_records(out/"registration_candidates.jsonl"),source_records(out/"source_frames.jsonl");
    if(!matrices||!csv||!timings||!manifest||!observations||!fit_records||!source_records)throw std::runtime_error("Cannot open output files");
    for(auto* file:{&matrices,&csv,&timings,&manifest,&observations,&fit_records})*file<<std::setprecision(17);
    csv<<"idx,x,y,yaw\n";timings<<"idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale\n";manifest<<"idx,timestamp_ns,filename\n";
    std::cout<<"Preparation: map_points="<<map.size()<<", tile_points="<<tile.size()<<", time="<<std::fixed<<std::setprecision(3)<<seconds(started)<<" s (excluded from frame timings)\n"<<std::flush;
    for(const auto& frame:frames) {
      if(interrupted)throw std::runtime_error("Interrupted by signal "+std::to_string(interrupted));
      failed_idx=frame.idx;stage="source_load";const auto begin=Clock::now();
      const auto source_sha=file_sha256(frame.path);
      auto raw=t3::voxel(read_ply(frame.path),raw_leaf);t3::synchronize();const double load_ms=1000.*seconds(begin);
      stage="registration";const auto reg=Clock::now();
      std::optional<t3::Pose> pair_pose,map_pose;t3::Fit pair_fit,map_fit;std::vector<t3::Fit> bootstrap;
      t3::Pose prediction=memory.published;
      const auto reference_ns=memory.reference?memory.reference->timestamp_ns:0;
      if(memory.reference) {
        t3::Pose local;local.scale=memory.reference->pose.scale;
        t3::Pose metric_reference;metric_reference.scale=memory.reference->pose.scale;
        const auto metric=t3::transform(memory.reference->raw,metric_reference);
        t3::Geometry relative(metric,map_leaf,false);
        pair_fit=t3::align(relative,raw,local,t3::Metric::Point,true,{4.,2.,1.});
        if(pair_fit.usable){pair_pose=t3::compose_camera_metres(memory.reference->pose,pair_fit.pose);prediction=*pair_pose;}
      }
      if((prediction.translation-center).cwiseAbs().maxCoeff()>25.) {
        center=prediction.translation;scene.reset();tile=map.crop(center,100.,30.,map_leaf);scene=std::make_unique<t3::Geometry>(tile,map_leaf,true);
      }
      const bool bootstrapping=!memory.reference;
      if(bootstrapping) {
        bootstrap.push_back(t3::align(*scene,raw,prediction,t3::Metric::Point,true,{8.,4.,2.,1.}));
        if(auto seed=ground_seed(raw,tile,prediction,raw_leaf,map_leaf))bootstrap.push_back(t3::align(*scene,raw,*seed,t3::Metric::Point,true,{8.,4.,2.,1.}));
        auto winner=bootstrap.end();
        for(auto it=bootstrap.begin();it!=bootstrap.end();++it)if(it->pose.finite() && it->training_support>=7 && it->rank>0 && (winner==bootstrap.end() || it->final_cost<winner->final_cost))winner=it;
        if(winner!=bootstrap.end())map_fit=*winner;
      } else {
        // Pair estimates current metric scale. Map planes update only observed
        // pose directions; they cannot silently shrink the scale each frame.
        map_fit=t3::align(*scene,raw,prediction,t3::Metric::Plane,false,{4.,2.,1.});
      }
      if(map_fit.usable)map_pose=map_fit.pose;
      t3::synchronize();const double registration_ms=1000.*seconds(reg),frame_ms=1000.*seconds(begin);
      if(file_sha256(frame.path)!=source_sha)throw std::runtime_error("Current source changed during registration");
      source_records<<"{\"idx\":"<<frame.idx<<",\"timestamp_ns\":"<<frame.ns<<",\"filename\":"<<quote(std::filesystem::path(frame.path).filename().string())<<",\"sha256\":"<<quote(source_sha)<<"}\n"<<std::flush;
      if(!source_records)throw std::runtime_error("Source hash record write failure");
      fit_records<<"{\"idx\":"<<frame.idx<<",\"timestamp_ns\":"<<frame.ns<<",\"map\":";fit_json(fit_records,map_fit);
      fit_records<<",\"pair\":";fit_json(fit_records,pair_fit);fit_records<<",\"bootstrap\":[";
      for(std::size_t i=0;i<bootstrap.size();++i){if(i)fit_records<<',';fit_json(fit_records,bootstrap[i]);}fit_records<<"]}\n"<<std::flush;
      if(!fit_records)throw std::runtime_error("Candidate diagnostic write failure");
      if(interrupted)throw std::runtime_error("Interrupted by signal "+std::to_string(interrupted));
      stage="state_commit";memory.commit(raw,frame.ns,map_pose?&*map_pose:nullptr,pair_pose?&*pair_pose:nullptr);
      if(map_pose && map_fit.rank==(bootstrapping?7:6))full_rank_map_ns=frame.ns;
      const double unsupported=double(frame.ns-memory.map_ns)/1e9;
      const double full_rank_age=full_rank_map_ns?double(frame.ns-full_rank_map_ns)/1e9:double(frame.ns-frames.front().ns)/1e9;
      const bool is_lost=full_rank_age>2.;
      stage="output";const auto& pose=memory.published;
      for(int a=0;a<3;++a)for(int b=0;b<4;++b)matrices<<(a||b?" ":"")<<(b==3?pose.translation[a]:pose.rotation(a,b));matrices<<'\n';
      csv<<frame.idx<<','<<pose.translation.x()<<','<<pose.translation.y()<<','<<std::atan2(pose.rotation(1,0),pose.rotation(0,0))<<'\n';
      timings<<frame.idx<<','<<frame.idx<<','<<frame.ns<<",cuda_float64_native_observable_icp,"<<load_ms<<','<<registration_ms<<','<<frame_ms<<','<<(map_pose?"true":"false")<<','<<pose.scale<<'\n';
      manifest<<frame.idx<<','<<frame.ns<<','<<std::filesystem::path(frame.path).filename().string()<<'\n';
      observations<<"{\"idx\":"<<frame.idx<<",\"original_idx\":"<<frame.idx<<",\"timestamp_ns\":"<<frame.ns<<",\"input_cutoff_ns\":"<<frame.ns
          <<",\"imu_used\":false,\"gnss_used\":false,\"accepted\":"<<(map_pose?"true":"false")<<",\"scale\":"<<pose.scale
          <<",\"tracking_lost\":"<<(is_lost?"true":"false")<<",\"unsupported_seconds\":"<<unsupported<<",\"source\":"<<quote(map_pose?"map":pair_pose?"pair":"held")
          <<",\"map_partial_update\":"<<(map_pose && map_fit.rank<(bootstrapping?7:6)?"true":"false")
          <<",\"last_full_rank_map_ns\":"<<full_rank_map_ns<<",\"seconds_without_full_rank_map\":"<<full_rank_age
          <<",\"elapsed_seconds\":"<<frame_ms/1000.<<",\"pair_reference_timestamp_ns\":"<<reference_ns<<",\"map_fit\":";
      fit_json(observations,map_fit);observations<<",\"pair_fit\":";fit_json(observations,pair_fit);observations<<",\"bootstrap_candidates\":[";
      for(std::size_t i=0;i<bootstrap.size();++i){if(i)observations<<',';fit_json(observations,bootstrap[i]);}observations<<"],\"pose\":";pose_json(observations,pose);observations<<"}\n";
      for(auto* file:{&matrices,&csv,&timings,&manifest,&observations}){file->flush();if(!*file)throw std::runtime_error("Output write failure");}
      ++processed;accepted+=bool(map_pose);pairs+=bool(pair_pose);lost+=is_lost;
      std::cout<<"Frame "<<processed<<'/'<<frames.size()<<": idx="<<frame.idx<<", map="<<(map_pose?"accepted":"rejected")<<", scale="<<std::fixed<<std::setprecision(6)<<pose.scale
          <<", load="<<std::setprecision(2)<<load_ms<<" ms, registration="<<registration_ms<<" ms, frame="<<frame_ms<<" ms\n"<<std::flush;
    }
    code=0;stage="complete";failed_idx=-1;
  }catch(const std::exception& error){failure=error.what();std::cerr<<"Failed: "<<failure<<'\n'<<std::flush;}
  std::ofstream status(out/"native_status.json");status<<std::setprecision(17)<<"{\"processed_frames\":"<<processed<<",\"map_accepted_frames\":"<<accepted<<",\"pair_accepted_frames\":"<<pairs
      <<",\"tracking_lost_frames\":"<<lost<<",\"returncode\":"<<code<<",\"stage\":"<<quote(stage)<<",\"failed_original_idx\":"<<failed_idx
      <<",\"failure\":"<<quote(failure)<<",\"total_seconds\":"<<seconds(started)<<"}\n";
  return code;
}
