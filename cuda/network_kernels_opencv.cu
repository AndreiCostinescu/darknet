#include <darknet/dark_cuda.h>
#include <cstdio>
#include <darknet/network.h>
#include <darknet/utils/blas.h>
#include <darknet/utils/utils.h>
#include <darknet/images/image_opencv.h>

void backward_network_gpu_cv(network net, network_state state) {
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *) calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *) calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    float *original_input = state.input;
    float *original_delta = state.delta;
    for (i = net.n - 1; i >= 0; --i) {
        state.index = i;
        layer l = net.layers[i];
        if (l.stopbackward == 1) break;
        if (l.stopbackward > get_current_iteration(net)) break;
        if (i == 0) {
            state.input = original_input;
            state.delta = original_delta;
        } else {
            layer prev = net.layers[i - 1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
            if (net.optimized_memory && !prev.keep_delta_gpu) {
                state.delta = net.state_delta_gpu;
            }
        }
        if (l.onlyforward) continue;

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

        l.backward_gpu(l, state);

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = l.type;
                avg_time_per_layer[i].time = took_time;
            } else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n bw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, l.type, took_time,
                   avg_time_per_layer[i].time);
        }

        if (i != 0) {
            layer prev = net.layers[i - 1];
            if (net.optimized_memory && state.delta && !prev.keep_delta_gpu) {
                if (prev.delta_gpu != state.delta)
                    simple_copy_ongpu(prev.outputs * prev.batch, state.delta, prev.delta_gpu);
                fill_ongpu(prev.outputs * prev.batch, 0, net.state_delta_gpu, 1);
            }
        }

        /*
        if(i != 0)
        {
            layer l = net.layers[i - 1];
            int state_delta_nan_inf = is_nan_or_inf(state.delta, l.outputs * l.batch);
            int state_input_nan_inf = is_nan_or_inf(state.input, l.outputs * l.batch);
            printf("\n i - %d  is_nan_or_inf(s.delta) = %d \n", i, state_delta_nan_inf);
            printf(" i - %d  is_nan_or_inf(s.input) = %d \n", i, state_input_nan_inf);
            if (state_delta_nan_inf || state_input_nan_inf) { printf(" found "); getchar(); }
        }
        */
    }

    if (net.adversarial && net.attention) {
        int img_size = net.w * net.h * net.c;
        float *original_input_cpu = (float *) xcalloc(img_size, sizeof(float));
        float *original_delta_cpu = (float *) xcalloc(img_size, sizeof(float));
        cuda_pull_array(original_input, original_input_cpu, img_size);
        cuda_pull_array(original_delta, original_delta_cpu, img_size);

        image attention_img = make_attention_image(img_size, original_delta_cpu, original_input_cpu, net.w, net.h,
                                                   net.c);
        show_image_cv(attention_img, "attention_img");
        resize_window_cv("attention_img", 500, 500);

        free_image(attention_img);

        free(original_input_cpu);
        free(original_delta_cpu);
    }
    if (net.adversarial) {
        int x_size = get_network_input_size(net) * net.batch;
        printf(" x_size = %d, original_delta = %p, original_input = %p, net.learning_rate = %f \n",
               x_size, original_delta, original_input, net.learning_rate);
        axpy_ongpu(x_size, net.learning_rate, original_delta, 1, original_input, 1);
        constrain_min_max_ongpu(x_size, 0, 1, original_input, 1);
    }

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (backward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - bw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id,
                   sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }
}

float train_network_datum_gpu_cv(network net, float *x, float *y) {
    *net.seen += net.batch;
    if (net.adversarial_lr && rand_int(0, 1) == 1 && get_current_iteration(net) > net.burn_in) {
        net.adversarial = 1;
        float lr_old = net.learning_rate;
        float scale = (get_current_iteration(net) / ((float) net.max_batches));
        //scale = sin(scale * M_PI);
        net.learning_rate = net.adversarial_lr * scale;
        int y_size = get_network_output_size(net) * net.batch;
        layer l = net.layers[net.n - 1];
        if (l.truths) y_size = l.truths * net.batch;
        float *truth_cpu = (float *) xcalloc(y_size, sizeof(float));

        const int img_size = net.w * net.h * net.c;
        float *old_input = (float *) xcalloc(img_size * net.batch, sizeof(float));
        memcpy(old_input, x, img_size * net.batch * sizeof(float));

        printf("\n adversarial training, adversarial_lr = %f \n", net.adversarial_lr * scale);

        forward_backward_network_gpu(net, x, truth_cpu);

        int b;
        for (b = 0; b < net.batch; ++b) {
            if (b % 2 == 1 && net.contrastive) {
                //printf(" b = %d old img, ", b);
                memcpy(x + img_size * b, old_input + img_size * b, img_size * sizeof(float));
            }
        }

        image im;
        im.w = net.w;
        im.h = net.h;
        im.c = net.c;
        im.data = x;
        show_image_cv(im, "adversarial data augmentation");
        resize_window_cv("adversarial data augmentation", 500, 500);
        wait_key_cv(1);

        free(old_input);
        free(truth_cpu);
        net.learning_rate = lr_old;
        net.adversarial = 0;
    }
    forward_backward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    //if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);
    const int sequence = get_sequence_value(net);
    //if (((*net.seen) / net.batch) % (net.subdivisions*sequence) == 0) update_network_gpu(net);

    return error;
}
