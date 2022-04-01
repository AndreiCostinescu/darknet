#include <darknet/darknet.h>
#include <darknet/network_opencv.h>
#include <darknet/images/image_opencv.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include <darknet/utils/parser.h>
#include <darknet/utils/utils.h>
#include <darknet/dark_cuda.h>

void visualize(char *cfgfile, char *weightfile) {
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    visualize_network(net);
    wait_until_press_key_cv();
}

int main(int argc, char **argv) {
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    printf(" _DEBUG is used \n");
#endif

#ifdef DEBUG
    printf(" DEBUG=1 \n");
#endif

    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        strip_args(argv[i]);
    }

    if (argc < 2) {
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if (find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
        printf("\n Currently Darknet doesn't support -nogpu flag. If you want to use CPU - please compile Darknet with GPU=0 in the Makefile, or compile darknet_no_gpu.sln on Windows.\n");
        exit(-1);
    }

#ifndef GPU
    gpu_index = -1;
    printf(" GPU isn't used \n");
    init_cpu();
#else   // GPU
    if (gpu_index >= 0) {
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }

    show_cuda_cudnn_info();
    cuda_debug_sync = find_arg(argc, argv, "-cuda_debug_sync");

#ifdef CUDNN_HALF
    printf(" CUDNN_HALF=1 \n");
#endif  // CUDNN_HALF
#endif  // GPU

    show_opencv_info();

    if (0 == strcmp(argv[1], "visualize")) {
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "imtest") || 0 == strcmp(argv[1], "test")) {
        test_resize_cv(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}
