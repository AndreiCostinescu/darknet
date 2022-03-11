#include <darknet/images/image_opencv_realsense.h>
#include <iostream>

#ifdef DARKNET_USE_OPENCV

#include <darknet/images/realsense-opencv-helpers.hpp>
#include <darknet/utils/utils.h>

using std::cerr;
using std::endl;

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif

extern "C" {

extern "C" image get_image_from_realsense(int w, int h, int c, mat_cv **in_img, void **in_depth, int letterbox) {
    return get_image_realsense_explicit(w, h, c, in_img, in_depth, letterbox, 1);
}

image get_image_realsense_explicit(int w, int h, int c, mat_cv **in_img, void **in_depth, int letterbox,
                                   int depthAsMat) {
    c = c ? c : 3;
    cv::Mat *src = 0, *depth = 0;
    static int once = 1;
    static rs2::pipeline pipe;
    static rs2::align alignTo(RS2_STREAM_COLOR);
    if (once) {
        once = 0;
        try {
            pipe.start();
        } catch (const rs2::error &e) {
            printf("RealSense error calling %s (%s): %s\n", e.get_failed_function().c_str(),
                   e.get_failed_args().c_str(), e.what());
            error("RealSense error!\n");
        }
        printf("Started Realsense pipeline!\n");
    }
    try {
        rs2::frameset currentFrame;
        currentFrame = pipe.wait_for_frames();
        currentFrame = alignTo.process(currentFrame);  // Make sure the frames are spatially aligned

        if (depthAsMat) {
            *in_depth = nullptr;
            depth = new cv::Mat(depth_frame_to_meters(currentFrame.get_depth_frame()));
        } else {
            *in_depth = (void *) new rs2::depth_frame(currentFrame.get_depth_frame());
        }
        src = new cv::Mat(frame_to_mat(currentFrame.get_color_frame()));
    } catch (const rs2::error &e) {
        printf("RealSense error calling %s (%s): %s\n", e.get_failed_function().c_str(), e.get_failed_args().c_str(),
               e.what());
        error("RealSense error!\n");
    } catch (std::exception &e) {
        char errorBuffer[512];
        strcat(errorBuffer, "Error in get_image_from_realsense while processing!\n");
        strcat(errorBuffer, e.what());
        error(errorBuffer);
    }

    image im = process_image_and_depth_explicit(w, h, c, (mat_cv *) src, in_img, (void *) depth, in_depth,
                                                letterbox, depthAsMat);
    /*
    printf("In image size = (%d x %d)\n", (*((cv::Mat **) in_img))->rows, (*((cv::Mat **) in_img))->cols);
    if (depthAsMat) {
        printf("In depth size = (%d x %d)\n", (*((cv::Mat **) in_depth))->rows, (*((cv::Mat **) in_depth))->cols);
    }
    //*/
    return im;
}

extern "C" void release_depth_frame(void **depth_frame) {
    try {
        rs2::depth_frame *rs2_depth_frame = (rs2::depth_frame *) (*depth_frame);
        // printf("Releasing depth frame rs2_depth_frame %d\n", rs2_depth_frame);
        if (rs2_depth_frame != 0) {
            // printf("Content at (0, 0) is %f\n", rs2_depth_frame->get_distance(0, 0));
        }
        delete rs2_depth_frame;
        *depth_frame = nullptr;
    } catch (...) {
        cerr << " OpenCV/Realsense exception: depth_frame " << depth_frame << " can't be released! \n";
    }
}

}   // extern "C"

#endif // DARKNET_USE_OPENCV
