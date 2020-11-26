// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once

#ifdef OPENCV

#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <exception>

// Convert rs2::frame to cv::Mat
static cv::Mat frame_to_mat(const rs2::frame &f) {
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        return Mat(Size(w, h), CV_8UC3, (void *) f.get_data(), Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_RGB8) {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void *) f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    } else if (f.get_profile().format() == RS2_FORMAT_Z16) {
        return Mat(Size(w, h), CV_16UC1, (void *) f.get_data(), Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_Y8) {
        return Mat(Size(w, h), CV_8UC1, (void *) f.get_data(), Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32) {
        return Mat(Size(w, h), CV_32FC1, (void *) f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

// Converts depth frame to a matrix of doubles with distances in meters
static cv::Mat depth_frame_to_meters(const rs2::depth_frame &f) {
    cv::Mat dm = frame_to_mat(f);
    dm.convertTo(dm, CV_64F);
    dm = dm * f.get_units();
    return dm;
}

void getRealsense3DPoint(rs2::depth_frame const &depthFrame, int x, int y, float (&point)[3]) {
    float position[2];
    position[0] = static_cast<float>(x);
    position[1] = static_cast<float>(y);

    // De-project from pixel to point in 3D
    rs2_intrinsics intr = depthFrame.get_profile().as<rs2::video_stream_profile>().get_intrinsics(); // Calibration data
    // Get the distance at the given pixel
    rs2_deproject_pixel_to_point(point, &intr, position, depthFrame.get_distance(x, y));
}

#endif