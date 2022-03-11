#ifndef IMAGE_OPENCV_REALSENSE_H
#define IMAGE_OPENCV_REALSENSE_H

#include <darknet/images/image_opencv.h>
#include <functional>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DARKNET_USE_OPENCV

image get_image_from_realsense(int w, int h, int c, mat_cv **in_img, void **in_depth, int letterbox);
image get_image_realsense_explicit(int w, int h, int c, mat_cv **in_img, void **in_depth, int letterbox,
                                   int depthAsMat);
void release_depth_frame(void **depth_frame);

#endif  // DARKNET_USE_OPENCV

#ifdef __cplusplus
}
#endif

#endif // IMAGE_OPENCV_REALSENSE_H
