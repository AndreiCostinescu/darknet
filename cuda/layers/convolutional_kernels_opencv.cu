#include <cuda_runtime.h>
#include <darknet/layers/convolutional_layer.h>
#include <darknet/layers/box.h>
#include <darknet/dark_cuda.h>
#include <darknet/images/image_opencv.h>

void assisted_excitation_forward_gpu_cv(convolutional_layer l, network_state state, int visualizeGroundTruth) {
    const int iteration_num = get_current_iteration(
            state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);

    // epoch
    //const float epoch = (float)(*state.net.seen) / state.net.train_images_num;

    // calculate alpha
    //const float alpha = (1 + cos(3.141592 * iteration_num)) / (2 * state.net.max_batches);
    //const float alpha = (1 + cos(3.141592 * epoch)) / (2 * state.net.max_batches);
    float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches)) / 2;
    //float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches));

    if (l.assisted_excitation == 1) {
        if (iteration_num > state.net.max_batches / 2) return;
    } else {
        if (iteration_num < state.net.burn_in) return;
        else if (iteration_num > l.assisted_excitation) return;
        else
            alpha = (1 + cos(3.141592 * iteration_num / (state.net.burn_in + l.assisted_excitation))) /
                    2; // from 1 to 0
    }

    //printf("\n epoch = %f, alpha = %f, seen = %d, max_batches = %d, train_images_num = %d \n",
    //    epoch, alpha, (*state.net.seen), state.net.max_batches, state.net.train_images_num);

    //const int size = l.outputs * l.batch;

    float *a_avg = (float *) calloc(l.out_w * l.out_h * l.batch, sizeof(float));
    float *gt = (float *) calloc(l.out_w * l.out_h * l.batch, sizeof(float));

    int b;
    int w, h;

    l.max_boxes = state.net.num_boxes;
    l.truths = l.max_boxes * (4 + 1);

    int num_truth = l.batch * l.truths;
    float *truth_cpu = (float *) calloc(num_truth, sizeof(float));
    cuda_pull_array(state.truth, truth_cpu, num_truth);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());

    for (b = 0; b < l.batch; ++b) {
        // calculate G
        int t;
        for (t = 0; t < state.net.num_boxes; ++t) {
            box truth = float_to_box_stride(truth_cpu + t * (4 + 1) + b * l.truths, 1);
            if (!truth.x) break;  // continue;
            float beta = 0;
            //float beta = 1 - alpha; // from 0 to 1
            float dw = (1 - truth.w) * beta;
            float dh = (1 - truth.h) * beta;
            //printf(" alpha = %f, beta = %f, truth.w = %f, dw = %f, tw+dw = %f, l.out_w = %d \n", alpha, beta, truth.w, dw, truth.w+dw, l.out_w);

            int left = floor((truth.x - (dw + truth.w) / 2) * l.out_w);
            int right = ceil((truth.x + (dw + truth.w) / 2) * l.out_w);
            int top = floor((truth.y - (dh + truth.h) / 2) * l.out_h);
            int bottom = ceil((truth.y + (dh + truth.h) / 2) * l.out_h);
            if (left < 0) left = 0;
            if (top < 0) top = 0;
            if (right > l.out_w) right = l.out_w;
            if (bottom > l.out_h) bottom = l.out_h;

            for (w = left; w <= right; w++) {
                for (h = top; h < bottom; h++) {
                    gt[w + l.out_w * h + l.out_w * l.out_h * b] = 1;
                }
            }
        }
    }

    cuda_push_array(l.gt_gpu, gt, l.out_w * l.out_h * l.batch);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());

    // calc avg_output on GPU - for whole batch
    calc_avg_activation_gpu(l.output_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());

    // calc new output
    //assisted_activation2_gpu(1, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);  // AE3: gt increases (beta = 1 - alpha = 0)
    //assisted_activation2_gpu(alpha, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);
    assisted_activation_gpu(alpha, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);
    //cudaStreamSynchronize(get_cuda_stream());
    //CHECK_CUDA(cudaPeekAtLastError());



    if (visualizeGroundTruth) {
        cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
        cudaStreamSynchronize(get_cuda_stream());
        CHECK_CUDA(cudaPeekAtLastError());

        for (b = 0; b < l.batch; ++b) {
            printf(" Assisted Excitation alpha = %f \n", alpha);
            image img = float_to_image(l.out_w, l.out_h, 1, &gt[l.out_w * l.out_h * b]);
            char buff[100];
            sprintf(buff, "a_excitation_gt_%d", b);
            show_image_cv(img, buff);

            //image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
            image img2 = float_to_image_scaled(l.out_w, l.out_h, 1, &l.output[l.out_w * l.out_h * l.out_c * b]);
            char buff2[100];
            sprintf(buff2, "a_excitation_output_%d", b);
            show_image_cv(img2, buff2);

            /*
            int c = l.out_c;
            if (c > 4) c = 4;
            image img3 = float_to_image(l.out_w, l.out_h, c, &l.output[l.out_w*l.out_h*l.out_c*b]);
            image dc = collapse_image_layers(img3, 1);
            char buff3[100];
            sprintf(buff3, "a_excitation_act_collapsed_%d", b);
            show_image_cv(dc, buff3);
            */

            wait_key_cv(5);
        }
        wait_until_press_key_cv();
    }

    free(truth_cpu);
    free(gt);
    free(a_avg);
}
