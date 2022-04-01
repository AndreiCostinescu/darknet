//
// Created by andrei on 01.04.22.
//

#include <darknet/layers/layer.h>
#include <darknet/dark_cuda.h>
#include <darknet/images/image_opencv.h>

void visualizeGroundTruth(layer l, float alpha) {
    cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
    cudaStreamSynchronize(get_cuda_stream());
    CHECK_CUDA(cudaPeekAtLastError());

    for (int b = 0; b < l.batch; ++b) {
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