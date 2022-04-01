#include <darknet/layers/convolutional_layer_opencv.h>
#include <darknet/images/image_opencv.h>
#include <darknet/layers/box.h>
#include <darknet/utils/utils.h>
#include <stdio.h>

void assisted_excitation_forward_cv(convolutional_layer l, network_state state, int visualizeGroundTruth) {
    const int iteration_num = (*state.net.seen) / (state.net.batch * state.net.subdivisions);

    // epoch
    //const float epoch = (float)(*state.net.seen) / state.net.train_images_num;

    // calculate alpha
    //const float alpha = (1 + cos(3.141592 * iteration_num)) / (2 * state.net.max_batches);
    //const float alpha = (1 + cos(3.141592 * epoch)) / (2 * state.net.max_batches);
    float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches));

    if (l.assisted_excitation > 1) {
        if (iteration_num > l.assisted_excitation) alpha = 0;
        else alpha = (1 + cos(3.141592 * iteration_num / l.assisted_excitation));
    }

    //printf("\n epoch = %f, alpha = %f, seen = %d, max_batches = %d, train_images_num = %d \n",
    //    epoch, alpha, (*state.net.seen), state.net.max_batches, state.net.train_images_num);

    float *a_avg = (float *) xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));
    float *g = (float *) xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));

    int b;
    int w, h, c;

    l.max_boxes = state.net.num_boxes;
    l.truths = l.max_boxes * (4 + 1);

    for (b = 0; b < l.batch; ++b) {
        // calculate G
        int t;
        for (t = 0; t < state.net.num_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t * (4 + 1) + b * l.truths, 1);
            if (!truth.x) break;  // continue;

            int left = floor((truth.x - truth.w / 2) * l.out_w);
            int right = ceil((truth.x + truth.w / 2) * l.out_w);
            int top = floor((truth.y - truth.h / 2) * l.out_h);
            int bottom = ceil((truth.y + truth.h / 2) * l.out_h);

            for (w = left; w <= right; w++) {
                for (h = top; h < bottom; h++) {
                    g[w + l.out_w * h + l.out_w * l.out_h * b] = 1;
                }
            }
        }
    }

    for (b = 0; b < l.batch; ++b) {
        // calculate average A
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++) {
                    a_avg[w + l.out_w * (h + l.out_h * b)] += l.output[w + l.out_w * (h + l.out_h * (c + l.out_c * b))];
                }
                a_avg[w + l.out_w * (h + l.out_h * b)] /= l.out_c;  // a_avg / d
            }
        }
    }

    // change activation
    for (b = 0; b < l.batch; ++b) {
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++) {
                    // a = a + alpha(t) + e(c,i,j) = a + alpha(t) + g(i,j) * avg_a(i,j) / channels
                    l.output[w + l.out_w * (h + l.out_h * (c + l.out_c * b))] +=
                            alpha *
                            g[w + l.out_w * (h + l.out_h * b)] *
                            a_avg[w + l.out_w * (h + l.out_h * b)];

                    //l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] =
                    //    alpha * g[w + l.out_w*(h + l.out_h*b)] * a_avg[w + l.out_w*(h + l.out_h*b)];
                }
            }
        }
    }

    if (visualizeGroundTruth) {
        for (b = 0; b < l.batch; ++b) {
            image img = float_to_image(l.out_w, l.out_h, 1, &g[l.out_w * l.out_h * b]);
            char buff[100];
            sprintf(buff, "a_excitation_%d", b);
            show_image_cv(img, buff);

            image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w * l.out_h * l.out_c * b]);
            char buff2[100];
            sprintf(buff2, "a_excitation_act_%d", b);
            show_image_cv(img2, buff2);
            wait_key_cv(5);
        }
        wait_until_press_key_cv();
    }

    free(g);
    free(a_avg);
}

image *get_weights(convolutional_layer l) {
    image *weights = (image *) xcalloc(l.n, sizeof(image));
    int i;
    for (i = 0; i < l.n; ++i) {
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
        char buff[256];
        sprintf(buff, "filter%d", i);
        save_image(weights[i], buff);
        */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights) {
    image *single_weights = get_weights(l);
    show_images_cv(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    show_image_cv(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}
