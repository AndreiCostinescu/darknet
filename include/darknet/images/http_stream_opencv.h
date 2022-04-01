#ifndef HTTP_STREAM_OPENCV_H
#define HTTP_STREAM_OPENCV_H

#include <darknet/images/image_opencv.h>

#ifdef __cplusplus
extern "C" {
#endif

void send_mjpeg(mat_cv *mat, int port, int timeout, int quality);

int send_http_post_request(char *http_post_host, int server_port, const char *videosource, detection *dets, int nboxes,
                           int classes, char **names, long long int frame_id, int ext_output, int timeout);

#ifdef __cplusplus
}
#endif

#endif // HTTP_STREAM_OPENCV_H
