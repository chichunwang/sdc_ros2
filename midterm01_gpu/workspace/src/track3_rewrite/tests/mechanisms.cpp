#include "track3_rewrite/core.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
t3::Cloud room() {
  std::mt19937 rng(20261024);std::uniform_real_distribution<double> unit(0.,1.);t3::Cloud result;
  for(int i=0;i<2200;++i) {
    result.push_back({1.+12.*unit(rng),-5.+10.*unit(rng),-2.});
    result.push_back({13.,-5.+10.*unit(rng),-2.+5.*unit(rng)});
    result.push_back({1.+12.*unit(rng),5.,-2.+5.*unit(rng)});
    result.push_back({1.,-5.+10.*unit(rng),-2.+5.*unit(rng)});
  }
  return result;
}
t3::Cloud inverse(const t3::Cloud& target,const t3::Pose& p) {
  t3::Cloud result;for(auto q:target){const Eigen::Vector3d raw=p.rotation.transpose()*(Eigen::Vector3d(q.x,q.y,q.z)-p.translation)/p.scale;result.push_back({raw.x(),raw.y(),raw.z()});}return result;
}
}
int main() {
  int passed=0;
  try {
    const auto target=room();
    t3::Geometry points(target,1.,true);
    {
      std::mt19937 rng(1);std::uniform_real_distribution<double> uniform(-7.,15.);t3::Cloud queries;
      for(int i=0;i<300;++i)queries.push_back({uniform(rng),uniform(rng),uniform(rng)});
      auto gpu=points.nearest(queries,2.);
      for(std::size_t i=0;i<queries.size();++i){int best=-1;double distance=4.;for(std::size_t j=0;j<target.size();++j){auto q=queries[i],p=target[j];double d=std::pow(q.x-p.x,2)+std::pow(q.y-p.y,2)+std::pow(q.z-p.z,2);if(d<distance||(d==distance&&(best<0||int(j)<best))){best=j;distance=d;}}require(gpu[i]==best,"Exact CUDA nearest neighbor mismatch");}
      ++passed;std::cout<<"PASS exact bounded float64 nearest neighbors\n";
    }
    {
      t3::Cloud ties={{-1,0,0},{1,0,0},{0,2,0}};t3::Geometry grid(ties,1.,false);
      auto ids=grid.nearest({{0,0,0},{100,100,100}},1.);
      require(ids[0]==0&&ids[1]==-1,"Tie/radius/missing behavior mismatch");++passed;std::cout<<"PASS deterministic tie, radius boundary and missing support\n";
    }
    {
      auto result=t3::voxel({{.01,.01,.01},{.02,.02,.02},{1.01,0,0}},.1);
      require(result.size()==2&&result[0].x==.01&&result[1].x==1.01,"Voxel did not keep original first rows");++passed;std::cout<<"PASS stable original-row GPU voxel\n";
    }
    {
      auto normals=points.normals();int valid=0;for(std::size_t i=0;i<normals.size();i+=4)if(std::abs(normals[i].z)>.99)++valid;
      require(valid>1800,"Independent register PCA did not recover floor normals");++passed;std::cout<<"PASS register-local 3x3 PCA\n";
    }
    t3::Pose truth;truth.scale=4.5;truth.rotation=Eigen::AngleAxisd(.04,Eigen::Vector3d::UnitZ()).toRotationMatrix();truth.translation={.2,-.1,.3};
    const auto raw=inverse(target,truth);
    {
      t3::Pose seed=truth;seed.translation+=Eigen::Vector3d(.1,-.08,.06);seed.scale=4.6;
      const auto evaluation=points.evaluate(raw,seed,2.,t3::Metric::Point,true);
      const double eps=1e-7;const int train=raw.size()-(raw.size()+4)/5;
      for(int axis=0;axis<7;++axis) {
        auto plus=seed,minus=seed;
        if(axis<3){plus.translation[axis]+=eps;minus.translation[axis]-=eps;}
        else if(axis<6){auto direction=Eigen::Vector3d::Unit(axis-3);plus.rotation=Eigen::AngleAxisd(eps,direction).toRotationMatrix()*seed.rotation;minus.rotation=Eigen::AngleAxisd(-eps,direction).toRotationMatrix()*seed.rotation;}
        else {plus.scale*=std::exp(eps);minus.scale*=std::exp(-eps);}
        const double difference=(points.evaluate(raw,plus,2.,t3::Metric::Point,true).training_cost-points.evaluate(raw,minus,2.,t3::Metric::Point,true).training_cost)/(2*eps);
        std::cout<<"gradient axis="<<axis<<" numeric="<<difference<<" analytic="<<evaluation.gradient[axis]/train<<'\n';
        if(axis==5) {
          for(double interval:{1e-6,1e-7}) {
            auto upper=seed,lower=seed;
            upper.rotation=Eigen::AngleAxisd(interval,Eigen::Vector3d::UnitZ()).toRotationMatrix()*seed.rotation;
            lower.rotation=Eigen::AngleAxisd(-interval,Eigen::Vector3d::UnitZ()).toRotationMatrix()*seed.rotation;
            auto ids_upper=points.nearest(t3::transform(raw,upper),2.),ids_lower=points.nearest(t3::transform(raw,lower),2.);
            int changes=0;for(std::size_t i=0;i<raw.size();++i)if(i%5 && ids_upper[i]!=ids_lower[i])++changes;
            std::cout<<"finite_difference_interval="<<interval<<" changed_training_correspondences="<<changes<<'\n';
            if(interval==eps)require(changes==0,"Gradient check crossed a correspondence boundary");
          }
        }
        require(std::abs(difference-evaluation.gradient[axis]/train)<1e-5,"Analytical objective gradient mismatch");
      }
      ++passed;std::cout<<"PASS analytic 7D objective gradient finite differences\n";
    }
    {
      auto seed=truth;seed.scale=4.7;seed.translation+=Eigen::Vector3d(.2,-.15,.1);
      auto fit=t3::align(points,raw,seed,t3::Metric::Point,true,{4.,2.,1.});
      std::cout<<"known_sim3 translation="<<(fit.pose.translation-truth.translation).norm()<<" scale="<<fit.pose.scale<<" rank="<<fit.rank<<" reason="<<fit.reason<<'\n';
      require(fit.usable&&(fit.pose.translation-truth.translation).norm()<.02&&std::abs(fit.pose.scale/truth.scale-1)<.003,"Known Sim3 recovery failed");++passed;std::cout<<"PASS known Sim3 recovery\n";
    }
    {
      t3::Cloud plane;for(int x=-20;x<=20;++x)for(int y=-20;y<=20;++y)plane.push_back({x*.2,y*.2,0});
      t3::Geometry scene(plane,1.,true);auto source=plane;t3::Pose seed;seed.scale=1.;seed.translation={.2,-.15,.15};
      const auto fit=t3::align(scene,source,seed,t3::Metric::Plane,false,{2.,1.});
      std::cout<<"plane rank="<<fit.rank<<" position="<<fit.pose.translation.transpose()<<" usable="<<fit.usable<<'\n';
      require(fit.usable&&fit.rank==3&&std::abs(fit.pose.translation.z())<1e-4,"Plane partial update failed");
      require(std::abs(fit.pose.translation.x()-.2)<1e-10&&std::abs(fit.pose.translation.y()+.15)<1e-10,"Unobserved plane directions changed");++passed;std::cout<<"PASS rank-3 plane update preserves unobserved XY\n";
    }
    {
      auto moved=raw;for(auto& p:moved)p.x+=1000.;t3::Pose seed;seed.scale=3.;
      auto fit=t3::align(points,moved,seed,t3::Metric::Point,true,{2.,1.});
      require(!fit.usable&&fit.training_support==0,"No-overlap must be rejected");++passed;std::cout<<"PASS no-overlap rejection\n";
    }
    {
      t3::Pose previous;previous.scale=6.;previous.translation={660,-222,10};previous.rotation=Eigen::AngleAxisd(.4,Eigen::Vector3d::UnitZ()).toRotationMatrix();
      t3::Pose local;local.scale=6.2;local.translation={.1,.03,.02};const auto composed=t3::compose_camera_metres(previous,local);
      require(composed.scale==6.2&&(composed.translation-(previous.translation+previous.rotation*local.translation)).norm()<1e-12,"Metric scale multiplied twice");++passed;std::cout<<"PASS camera-metre composition\n";
    }
    {
      t3::Memory memory;t3::Pose pose;memory.commit(raw,100,&pose,nullptr);memory.commit(raw,200,nullptr,nullptr);
      require(memory.output_ns==200&&memory.reference->timestamp_ns==100&&memory.map_ns==100,"Unsupported frame replaced supported reference");
      bool failed=false;try{memory.commit(raw,200,&pose,nullptr);}catch(const std::invalid_argument&){failed=true;}require(failed,"Timestamp allowed a past-frame rewrite");
      memory.commit(raw,300,nullptr,&pose);require(memory.reference->timestamp_ns==300&&memory.map_ns==100,"Pair support reset map age");++passed;std::cout<<"PASS causal reference/map-anchor state\n";
    }
    {
      t3::Matrix7 h=t3::Matrix7::Zero();h(2,2)=10.;t3::Vector7 b=t3::Vector7::Ones(),eigen;int rank;
      auto step=t3::observable_step(h,b,5.,false,rank,eigen);
      require(rank==1&&std::abs(step[2]+.1)<1e-12&&step[0]==0.&&step[1]==0.,"Projected normal-equation update failed");++passed;std::cout<<"PASS data-only observable eigensolve\n";
    }
    {
      t3::Cloud singleton(20,t3::Point{0,0,0});t3::Geometry degenerate(singleton,1.,false);
      t3::Pose seed;seed.scale=3.;
      auto fit=t3::align(degenerate,raw,seed,t3::Metric::Point,true,{8.,4.,2.,1.});
      require(!fit.usable,"Scale-collapse toward a single target was accepted");
      require(fit.pose.scale>=seed.scale*.5&&fit.pose.scale<=seed.scale*2.,"Scale left its declared domain");
      ++passed;std::cout<<"PASS singular Sim3 scale-collapse rejection\n";
    }
    std::cout<<"Mechanisms passed: "<<passed<<"\n";return 0;
  }catch(const std::exception& error){std::cerr<<"FAIL after "<<passed<<": "<<error.what()<<'\n';return 1;}
}
