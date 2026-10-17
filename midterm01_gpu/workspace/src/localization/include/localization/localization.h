#ifndef LOCALIZATION_LOCALIZATION_H_
#define LOCALIZATION_LOCALIZATION_H_

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>


namespace localization {

struct PlyFrameInfo {
    std::int64_t timestamp_ns = 0;
    std::string path;
};

// Latest sensor messages at or before a PLY timestamp; retained across frames.
struct DB3Frame {
    geometry_msgs::msg::PoseStamped gnss_pose;
    sensor_msgs::msg::Imu imu_data;
    bool has_gnss_pose = false;
    bool has_imu_data = false;
    std::int64_t gnss_bag_timestamp_ns = 0;
    std::int64_t imu_bag_timestamp_ns = 0;
};

class LocalizationApp {
public:
    int Start(int argc, char** argv);

private:
    // ** Data loading & Parameter setting **
    bool LoadParameters();
    void SetupRosInterfaces();
    bool LoadMap();
    bool OpenDB3File();
    bool LoadDB3Frame(std::int64_t timestamp_ns, DB3Frame& frame);
    bool IndexPlyFiles(const std::string& directory_path);
    bool LoadPlyFrame(const PlyFrameInfo& frame, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& color_cloud) const;
    bool LoadInitialPose(Eigen::Matrix4f& pose) const;
    bool LoadSensorPriors();
    

    // ** input data/map processing **
    pcl::PointCloud<pcl::PointXYZ>::Ptr ScalePlyPointCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input, double scale) const;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ScalePlyColorCloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& input, double scale) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr FilterInputCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input) const;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr FilterColorCloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& input) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr CropSubmap(const Eigen::Matrix4f& center_pose) const;

    // ** Localization **
    bool RunLocalization();
    bool SetupRegistration();
    bool AlignNDT(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                  const Eigen::Matrix4f& initial_guess,
                  Eigen::Matrix4f& pose, double& score);
    
    // ** Rviz $ Record **
    void PublishGlobalMap();
    void PublishFrame(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input,
                      const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& color,
                      const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& submap,
                      const Eigen::Matrix4f& pose, std::int64_t timestamp_ns);
    void WritePose(const Eigen::Matrix4f& pose);

    // ** Messages transform **
    geometry_msgs::msg::TransformStamped ToTransformStamped(const Eigen::Matrix4f& matrix,
                                                            const std::string& parent_frame,
                                                            const std::string& child_frame,
                                                            std::int64_t timestamp_ns) const;
    geometry_msgs::msg::PoseStamped ToPoseStamped(const Eigen::Matrix4f& matrix, std::int64_t timestamp_ns) const;
    bool IsFinite(const Eigen::Matrix4f& matrix) const;
    bool TimestampNsFromPlyFilename(const boost::filesystem::path& file, std::int64_t& timestamp_ns);
    void Shutdown();

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr input_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr submap_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    nav_msgs::msg::Path path_;

    std::unique_ptr<rosbag2_cpp::Reader> bag_reader_;
    rosbag2_storage::SerializedBagMessageSharedPtr next_db3_message_;
    std::string map_path_;
    std::string map_file_;
    std::string map_cache_file_;
    std::string bag_path_;
    std::string ply_path_;
    std::string init_guess_file_;
    std::vector<double> initial_pose_;
    std::string save_path_;
    std::string save_file_;
    std::string localization_profile_ = "legacy";
    std::string sensor_prior_file_;
    std::vector<Eigen::Matrix4f> sensor_priors_;
    double sensor_max_xy_correction_ = 1.5;
    double sensor_max_yaw_correction_deg_ = 8.0;
    double sensor_smoothing_alpha_ = 0.25;
    std::string geometry_mode_ = "off";
    double geometry_normal_gain_ = 2.0;
    double geometry_tangent_weight_ = 0.05;
    

    double map_voxel_leaf_size_ = 1.0;
    double input_voxel_leaf_size_ = 0.3;
    int min_input_points_ = 10;
    double submap_half_size_ = 100.0;
    double submap_half_height_ = 30.0;

    double ndt_resolution_ = 8.0;
    double ndt_step_size_ = 0.8;
    double ndt_transformation_epsilon_ = 0.05;
    int ndt_max_iterations_ = 30;
    std::string registration_backend_ = "cuda_ndt";
    std::string cuda_ndt_mode_ = "D2D";
    double cuda_translation_epsilon_ = 0.01;
    double cuda_rotation_epsilon_ = 0.002;
    std::unique_ptr<fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ>> cuda_ndt_;
    double last_ndt_ms_ = 0.0;
    double last_fitness_ms_ = 0.0;
    double ply_scale_ = 3.0;

    bool publish_global_map_ = false;
    int max_frames_ = 0;
    bool keep_alive_ = true;

    std::ofstream result_file_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
    std::vector<PlyFrameInfo> ply_frames_;

};

}  // namespace localization

#endif  // LOCALIZATION_LOCALIZATION_H_
