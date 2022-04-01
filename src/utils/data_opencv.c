//
// Created by andrei on 01.04.22.
//

#include <darknet/utils/data_opencv.h>
#include <darknet/images/http_stream.h>
#include <darknet/images/image_opencv.h>
#include <darknet/utils/utils.h>

data load_data_detection_cv(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes,
                            int use_flip, int use_gaussian_noise, int use_blur, int use_mixup,
                            float jitter, float resize, float hue, float saturation, float exposure, int mini_batch,
                            int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive,
                            int contrastive_jit_flip, int contrastive_color, int show_imgs) {
    const int random_index = random_gen();
    c = c ? c : 3;

    if (use_mixup == 2 || use_mixup == 4) {
        printf("\n cutmix=1 - isn't supported for Detector (use cutmix=1 only for Classifier) \n");
        if (check_mistakes) getchar();
        if (use_mixup == 2) use_mixup = 0;
        else use_mixup = 3;
    }
    if (use_mixup == 3 && letter_box) {
        //printf("\n Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters \n");
        //if (check_mistakes) getchar();
        //exit(0);
    }
    if (random_gen() % 2 == 0) use_mixup = 0;
    int i;

    int *cut_x = NULL, *cut_y = NULL;
    if (use_mixup == 3) {
        cut_x = (int *) calloc(n, sizeof(int));
        cut_y = (int *) calloc(n, sizeof(int));
        const float min_offset = 0.2; // 20%
        for (i = 0; i < n; ++i) {
            cut_x[i] = rand_int(w * min_offset, w * (1 - min_offset));
            cut_y[i] = rand_int(h * min_offset, h * (1 - min_offset));
        }
    }

    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float **) xcalloc(d.X.rows, sizeof(float *));
    d.X.cols = h * w * c;

    float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale = 0;
    float resize_r1 = 0, resize_r2 = 0;
    float dhue = 0, dsat = 0, dexp = 0, flip = 0, blur = 0;
    int augmentation_calculated = 0, gaussian_noise = 0;

    d.y = make_matrix(n, truth_size * boxes);
    int i_mixup = 0;
    for (i_mixup = 0; i_mixup <= use_mixup; i_mixup++) {
        if (i_mixup) augmentation_calculated = 0;   // recalculate augmentation for the 2nd sequence if(track==1)

        char **random_paths;
        if (track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
        else random_paths = get_random_paths_custom(paths, n, m, contrastive);

        for (i = 0; i < n; ++i) {
            float *truth = (float *) xcalloc(truth_size * boxes, sizeof(float));
            const char *filename = random_paths[i];

            int flag = (c >= 3);
            mat_cv *src;
            src = load_image_mat_cv(filename, flag);
            if (src == NULL) {
                printf("\n Error in load_data_detection() - OpenCV \n");
                fflush(stdout);
                if (check_mistakes) {
                    getchar();
                }
                continue;
            }

            int oh = get_height_mat(src);
            int ow = get_width_mat(src);

            int dw = (ow * jitter);
            int dh = (oh * jitter);

            float resize_down = resize, resize_up = resize;
            if (resize_down > 1.0) resize_down = 1 / resize_down;
            int min_rdw = ow * (1 - (1 / resize_down)) / 2;   // < 0
            int min_rdh = oh * (1 - (1 / resize_down)) / 2;   // < 0

            if (resize_up < 1.0) resize_up = 1 / resize_up;
            int max_rdw = ow * (1 - (1 / resize_up)) / 2;     // > 0
            int max_rdh = oh * (1 - (1 / resize_up)) / 2;     // > 0
            //printf(" down = %f, up = %f \n", (1 - (1 / resize_down)) / 2, (1 - (1 / resize_up)) / 2);

            if (!augmentation_calculated || !track) {
                augmentation_calculated = 1;
                resize_r1 = random_float();
                resize_r2 = random_float();

                if (!contrastive || contrastive_jit_flip || i % 2 == 0) {
                    r1 = random_float();
                    r2 = random_float();
                    r3 = random_float();
                    r4 = random_float();

                    flip = use_flip ? random_gen() % 2 : 0;
                }

                r_scale = random_float();

                if (!contrastive || contrastive_color || i % 2 == 0) {
                    dhue = rand_uniform_strong(-hue, hue);
                    dsat = rand_scale(saturation);
                    dexp = rand_scale(exposure);
                }

                if (use_blur) {
                    int tmp_blur = rand_int(0, 2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
                    if (tmp_blur == 0) blur = 0;
                    else if (tmp_blur == 1) blur = 1;
                    else blur = use_blur;
                }

                if (use_gaussian_noise && rand_int(0, 1) == 1) gaussian_noise = use_gaussian_noise;
                else gaussian_noise = 0;
            }

            int pleft = rand_precalc_random(-dw, dw, r1);
            int pright = rand_precalc_random(-dw, dw, r2);
            int ptop = rand_precalc_random(-dh, dh, r3);
            int pbot = rand_precalc_random(-dh, dh, r4);

            if (resize < 1) {
                // downsize only
                pleft += rand_precalc_random(min_rdw, 0, resize_r1);
                pright += rand_precalc_random(min_rdw, 0, resize_r2);
                ptop += rand_precalc_random(min_rdh, 0, resize_r1);
                pbot += rand_precalc_random(min_rdh, 0, resize_r2);
            } else {
                pleft += rand_precalc_random(min_rdw, max_rdw, resize_r1);
                pright += rand_precalc_random(min_rdw, max_rdw, resize_r2);
                ptop += rand_precalc_random(min_rdh, max_rdh, resize_r1);
                pbot += rand_precalc_random(min_rdh, max_rdh, resize_r2);
            }

            //printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);

            //float scale = rand_precalc_random(.25, 2, r_scale); // unused currently
            //printf(" letter_box = %d \n", letter_box);

            if (letter_box) {
                float img_ar = (float) ow / (float) oh;
                float net_ar = (float) w / (float) h;
                float result_ar = img_ar / net_ar;
                //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if (result_ar > 1)  // sheight - should be increased
                {
                    float oh_tmp = ow / net_ar;
                    float delta_h = (oh_tmp - oh) / 2;
                    ptop = ptop - delta_h;
                    pbot = pbot - delta_h;
                    //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                } else  // swidth - should be increased
                {
                    float ow_tmp = oh * net_ar;
                    float delta_w = (ow_tmp - ow) / 2;
                    pleft = pleft - delta_w;
                    pright = pright - delta_w;
                    //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
                }

                //printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);
            }

            // move each 2nd image to the corner - so that most of it was visible
            if (use_mixup == 3 && random_gen() % 2 == 0) {
                if (flip) {
                    if (i_mixup == 0) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 1) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 2) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
                    if (i_mixup == 3) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
                } else {
                    if (i_mixup == 0) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 1) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 2) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
                    if (i_mixup == 3) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
                }
            }

            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;

            float sx = (float) swidth / ow;
            float sy = (float) sheight / oh;

            float dx = ((float) pleft / ow) / sx;
            float dy = ((float) ptop / oh) / sy;


            int min_w_h = fill_truth_detection(filename, boxes, truth_size, truth, classes, flip, dx, dy, 1. / sx,
                                               1. / sy, w, h);
            //for (int z = 0; z < boxes; ++z) if(truth[z*truth_size] > 0) printf(" track_id = %f \n", truth[z*truth_size + 5]);
            //printf(" truth_size = %d \n", truth_size);

            if ((min_w_h / 8) < blur && blur > 1)
                blur = min_w_h / 8;   // disable blur if one of the objects is too small

            image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp,
                                               gaussian_noise, blur, boxes, truth_size, truth);

            if (use_mixup == 0) {
                d.X.vals[i] = ai.data;
                memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
            } else if (use_mixup == 1) {
                if (i_mixup == 0) {
                    d.X.vals[i] = ai.data;
                    memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
                } else if (i_mixup == 1) {
                    image old_img = make_empty_image(w, h, c);
                    old_img.data = d.X.vals[i];
                    //show_image(ai, "new");
                    //show_image(old_img, "old");
                    //wait_until_press_key_cv();
                    blend_images_cv(ai, 0.5, old_img, 0.5);
                    blend_truth(d.y.vals[i], boxes, truth_size, truth);
                    free_image(old_img);
                    d.X.vals[i] = ai.data;
                }
            } else if (use_mixup == 3) {
                if (i_mixup == 0) {
                    image tmp_img = make_image(w, h, c);
                    d.X.vals[i] = tmp_img.data;
                }

                if (flip) {
                    int tmp = pleft;
                    pleft = pright;
                    pright = tmp;
                }

                const int left_shift = min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft * w / ow)));
                const int top_shift = min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop * h / oh)));

                const int right_shift = min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright * w / ow)));
                const int bot_shift = min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot * h / oh)));


                int k, x, y;
                for (k = 0; k < c; ++k) {
                    for (y = 0; y < h; ++y) {
                        int j = y * w + k * w * h;
                        if (i_mixup == 0 && y < cut_y[i]) {
                            int j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift) * w + k * w * h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                        }
                        if (i_mixup == 1 && y < cut_y[i]) {
                            int j_src = left_shift + (y + h - cut_y[i] - bot_shift) * w + k * w * h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
                        }
                        if (i_mixup == 2 && y >= cut_y[i]) {
                            int j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i]) * w + k * w * h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                        }
                        if (i_mixup == 3 && y >= cut_y[i]) {
                            int j_src = left_shift + (top_shift + y - cut_y[i]) * w + k * w * h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
                        }
                    }
                }

                blend_truth_mosaic(d.y.vals[i], boxes, truth_size, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift,
                                   right_shift, top_shift, bot_shift, w, h, mosaic_bound);

                free_image(ai);
                ai.data = d.X.vals[i];
            }


            if (show_imgs && i_mixup == use_mixup)   // delete i_mixup
            {
                image tmp_ai = copy_image(ai);
                char buff[1000];
                //sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
                sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
                int t;
                for (t = 0; t < boxes; ++t) {
                    box b = float_to_box_stride(d.y.vals[i] + t * truth_size, 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.) * ai.w;
                    int right = (b.x + b.w / 2.) * ai.w;
                    int top = (b.y - b.h / 2.) * ai.h;
                    int bot = (b.y + b.h / 2.) * ai.h;
                    draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
                }

                save_image(tmp_ai, buff);
                if (show_imgs == 1) {
                    //char buff_src[1000];
                    //sprintf(buff_src, "src_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
                    //show_image_mat(src, buff_src);
                    show_image_cv(tmp_ai, buff);
                    wait_until_press_key_cv();
                }
                printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
                free_image(tmp_ai);
            }

            release_mat(&src);
            free(truth);
        }
        if (random_paths) free(random_paths);
    }


    return d;
}

data load_data_augment_cv(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min,
                          int max, int w, int h, float angle, float aspect, float hue, float saturation, float exposure,
                          int use_mixup, int use_blur, int show_imgs, float label_smooth_eps, int dontuse_opencv,
                          int contrastive) {
    char **paths_stored = paths;
    if (m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, use_flip, min, max, w, h, angle, aspect, hue, saturation, exposure,
                                   dontuse_opencv, contrastive);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy, label_smooth_eps, contrastive);

    if (use_mixup && rand_int(0, 1)) {
        char **paths_mix = get_random_paths(paths_stored, n, m);
        data d2 = {0};
        d2.shallow = 0;
        d2.X = load_image_augment_paths(paths_mix, n, use_flip, min, max, w, h, angle, aspect, hue, saturation,
                                        exposure, dontuse_opencv, contrastive);
        d2.y = load_labels_paths(paths_mix, n, labels, k, hierarchy, label_smooth_eps, contrastive);
        free(paths_mix);

        data d3 = {0};
        d3.shallow = 0;
        data d4 = {0};
        d4.shallow = 0;
        if (use_mixup >= 3) {
            char **paths_mix3 = get_random_paths(paths_stored, n, m);
            d3.X = load_image_augment_paths(paths_mix3, n, use_flip, min, max, w, h, angle, aspect, hue, saturation,
                                            exposure, dontuse_opencv, contrastive);
            d3.y = load_labels_paths(paths_mix3, n, labels, k, hierarchy, label_smooth_eps, contrastive);
            free(paths_mix3);

            char **paths_mix4 = get_random_paths(paths_stored, n, m);
            d4.X = load_image_augment_paths(paths_mix4, n, use_flip, min, max, w, h, angle, aspect, hue, saturation,
                                            exposure, dontuse_opencv, contrastive);
            d4.y = load_labels_paths(paths_mix4, n, labels, k, hierarchy, label_smooth_eps, contrastive);
            free(paths_mix4);
        }


        // mix
        int i, j;
        for (i = 0; i < d2.X.rows; ++i) {

            int mixup = use_mixup;
            if (use_mixup == 4) mixup = rand_int(2, 3); // alternate CutMix and Mosaic

            // MixUp -----------------------------------
            if (mixup == 1) {
                // mix images
                for (j = 0; j < d2.X.cols; ++j) {
                    d.X.vals[i][j] = (d.X.vals[i][j] + d2.X.vals[i][j]) / 2.0f;
                }

                // mix labels
                for (j = 0; j < d2.y.cols; ++j) {
                    d.y.vals[i][j] = (d.y.vals[i][j] + d2.y.vals[i][j]) / 2.0f;
                }
            }
                // CutMix -----------------------------------
            else if (mixup == 2) {
                const float min = 0.3;  // 0.3*0.3 = 9%
                const float max = 0.8;  // 0.8*0.8 = 64%
                const int cut_w = rand_int(w * min, w * max);
                const int cut_h = rand_int(h * min, h * max);
                const int cut_x = rand_int(0, w - cut_w - 1);
                const int cut_y = rand_int(0, h - cut_h - 1);
                const int left = cut_x;
                const int right = cut_x + cut_w;
                const int top = cut_y;
                const int bot = cut_y + cut_h;

                assert(cut_x >= 0 && cut_x <= w);
                assert(cut_y >= 0 && cut_y <= h);
                assert(cut_w >= 0 && cut_w <= w);
                assert(cut_h >= 0 && cut_h <= h);

                assert(right >= 0 && right <= w);
                assert(bot >= 0 && bot <= h);

                assert(top <= bot);
                assert(left <= right);

                const float alpha = (float) (cut_w * cut_h) / (float) (w * h);
                const float beta = 1 - alpha;

                int c, x, y;
                for (c = 0; c < 3; ++c) {
                    for (y = top; y < bot; ++y) {
                        for (x = left; x < right; ++x) {
                            int j = x + y * w + c * w * h;
                            d.X.vals[i][j] = d2.X.vals[i][j];
                        }
                    }
                }

                //printf("\n alpha = %f, beta = %f \n", alpha, beta);
                // mix labels
                for (j = 0; j < d.y.cols; ++j) {
                    d.y.vals[i][j] = d.y.vals[i][j] * beta + d2.y.vals[i][j] * alpha;
                }
            }
                // Mosaic -----------------------------------
            else if (mixup == 3) {
                const float min_offset = 0.2; // 20%
                const int cut_x = rand_int(w * min_offset, w * (1 - min_offset));
                const int cut_y = rand_int(h * min_offset, h * (1 - min_offset));

                float s1 = (float) (cut_x * cut_y) / (w * h);
                float s2 = (float) ((w - cut_x) * cut_y) / (w * h);
                float s3 = (float) (cut_x * (h - cut_y)) / (w * h);
                float s4 = (float) ((w - cut_x) * (h - cut_y)) / (w * h);

                int c, x, y;
                for (c = 0; c < 3; ++c) {
                    for (y = 0; y < h; ++y) {
                        for (x = 0; x < w; ++x) {
                            int j = x + y * w + c * w * h;
                            if (x < cut_x && y < cut_y) d.X.vals[i][j] = d.X.vals[i][j];
                            if (x >= cut_x && y < cut_y) d.X.vals[i][j] = d2.X.vals[i][j];
                            if (x < cut_x && y >= cut_y) d.X.vals[i][j] = d3.X.vals[i][j];
                            if (x >= cut_x && y >= cut_y) d.X.vals[i][j] = d4.X.vals[i][j];
                        }
                    }
                }

                for (j = 0; j < d.y.cols; ++j) {
                    const float max_s = 1;// max_val_cmp(s1, max_val_cmp(s2, max_val_cmp(s3, s4)));

                    d.y.vals[i][j] =
                            d.y.vals[i][j] * s1 / max_s + d2.y.vals[i][j] * s2 / max_s + d3.y.vals[i][j] * s3 / max_s +
                            d4.y.vals[i][j] * s4 / max_s;
                }
            }
        }

        free_data(d2);

        if (use_mixup >= 3) {
            free_data(d3);
            free_data(d4);
        }
    }

    if (use_blur) {
        int i;
        for (i = 0; i < d.X.rows; ++i) {
            if (random_gen() % 4 == 0) {
                image im = make_empty_image(w, h, 3);
                im.data = d.X.vals[i];
                int ksize = use_blur;
                if (use_blur == 1) ksize = 15;
                image blurred = blur_image(im, ksize);
                free_image(im);
                d.X.vals[i] = blurred.data;
                //if (i == 0) {
                //    show_image(im, "Not blurred");
                //    show_image(blurred, "blurred");
                //    wait_until_press_key_cv();
                //}
            }
        }
    }

    if (show_imgs) {
        int i, j;
        for (i = 0; i < d.X.rows; ++i) {
            image im = make_empty_image(w, h, 3);
            im.data = d.X.vals[i];
            char buff[1000];
            sprintf(buff, "aug_%d_%s_%d", i, basecfg((char *) paths[i]), random_gen());
            save_image(im, buff);

            char buff_string[1000];
            sprintf(buff_string, "\n Classes: ");
            for (j = 0; j < d.y.cols; ++j) {
                if (d.y.vals[i][j] > 0) {
                    char buff_tmp[100];
                    sprintf(buff_tmp, " %d (%f), ", j, d.y.vals[i][j]);
                    strcat(buff_string, buff_tmp);
                }
            }
            printf("%s \n", buff_string);

            if (show_imgs == 1) {
                show_image_cv(im, buff);
                wait_until_press_key_cv();
            }
        }
        printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
    }

    if (m) free(paths);

    return d;
}
