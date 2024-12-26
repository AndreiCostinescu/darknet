#ifndef DEMO_REALSENSE_H
#define DEMO_REALSENSE_H

#include <darknet/images/image.h>

#ifdef __cplusplus
extern "C" {
#endif
void demo_realsense(
        char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename,
        char **names, int classes, int avgframes, int frame_skip, char *prefix, char *out_filename, int mjpeg_port,
        int dontdraw_bbox, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec,
        char *http_post_host, int benchmark, int benchmark_layers, int use_realsense, int depth_as_mat);
#ifdef __cplusplus
}
#endif

#endif  // DEMO_REALSENSE_H