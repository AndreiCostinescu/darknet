#ifndef DATA_OPENCV_H
#define DATA_OPENCV_H

#include <darknet/darknet.h>
#include <darknet/utils/data.h>

#ifdef __cplusplus
extern "C" {
#endif

data load_data_detection_cv(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes,
                            int use_flip, int gaussian_noise, int use_blur, int use_mixup, float jitter, float resize,
                            float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed,
                            int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip,
                            int contrastive_color, int show_imgs);

data load_data_augment_cv(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min,
                          int max, int w, int h, float angle, float aspect, float hue, float saturation, float exposure,
                          int use_mixup, int use_blur, int show_imgs, float label_smooth_eps, int dontuse_opencv,
                          int contrastive);

#ifdef __cplusplus
}
#endif

#endif
