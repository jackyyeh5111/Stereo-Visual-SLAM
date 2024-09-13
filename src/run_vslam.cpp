#include <cmath>
#include <iostream>
#include "stereo_visual_slam_main/library_include.hpp"
#include "stereo_visual_slam_main/types_def.hpp"
#include <vector>
#include <string>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include "stereo_visual_slam_main/visual_odometry.hpp"
#include <stereo_visual_slam_main/map.hpp>
#include <stereo_visual_slam_main/optimization.hpp>

int main(int argc, char **argv)
{
    std::string dataset = "/home/jackyyeh/dataset/demo_dataset/sequences/00/";
    
    vslam::Map my_map = vslam::Map();
    vslam::VO my_VO(dataset, my_map);

    // define camera parameters
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573;
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                 0, fy, cy,
                 0, 0, 1);

    for (int ite = 0; ite < 4541; ite++)
    {
        auto start = high_resolution_clock::now();
     
        bool not_lost = true;
        bool if_insert_keyframe = false;
        not_lost = my_VO.pipeline(if_insert_keyframe);


        cout << "[TIME] Pipeline function: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << " ms" << endl;

        // debug printing
        // learned: accessing non-existing [i] in unordered would create a new component
        // learned: use iterator to traverse unordered map
        // learned: at() throws out-of-range exception whereas operator[] shows undefined behavior.
        // std::cout << "Num of landmarks: " << my_map.landmarks_.size() << std::endl;
        // std::cout << "Num of keyframes: " << my_map.keyframes_.size() << std::endl;
        // for (auto &kf : my_map.keyframes_)
        // {
        //     std::cout << "  Num of features in keyframe: " << kf.second.keyframe_id_ << " - " << kf.second.features_.size() << std::endl;
        // }

        // Optimization
        if (if_insert_keyframe && my_map.keyframes_.size() >= 10)
        {
            // reject the outliers
            vslam::optimize_map(my_map.keyframes_, my_map.landmarks_, K, false, false, 5);
            vslam::optimize_map(my_map.keyframes_, my_map.landmarks_, K, false, false, 5);
            // do the optimization
            vslam::optimize_map(my_map.keyframes_, my_map.landmarks_, K, true, false, 10);
        }

        // Optimization (Pose Only)
        if (if_insert_keyframe && my_map.keyframes_.size() >= 10)
        {
            vslam::optimize_pose_only(my_map.keyframes_, my_map.landmarks_, K, true, 10);
        }

        if (not_lost == false)
        {
            break;
        }
    }

    return 0;
}