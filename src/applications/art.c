#include <darknet/network.h>
#include <darknet/utils/utils.h>
#include <darknet/utils/parser.h>
#include <darknet/utils/option_list.h>
#include <darknet/utils/blas.h>
#include <darknet/applications/classifier.h>
#ifdef WIN32
#include <time.h>
#include <darknet/utils/gettimeofday.h>
#else
#include <sys/time.h>
#endif


void demo_art(char *cfgfile, char *weightfile, int cam_index)
{
#ifdef DARKNET_USE_OPENCV
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);
    cap_cv * cap;

    cap = get_capture_webcam(cam_index);

    char *window = "ArtJudgementBot9000!!!";
    if(!cap) error("Couldn't connect to webcam.\n");
    create_window_cv(window, 0, 512, 512);
    int i;
    int idx[] = {37, 401, 434};
    int n = sizeof(idx)/sizeof(idx[0]);

    while(1){
        image in = get_image_from_stream_cpp(cap);
        image in_s = resize_image(in, net.w, net.h);
        show_image(in, window);

        float *p = network_predict(net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");

        float score = 0;
        for(i = 0; i < n; ++i){
            float s = p[idx[i]];
            if (s > score) score = s;
        }
        score = score;
        printf("I APPRECIATE THIS ARTWORK: %10.7f%%\n", score*100);
        printf("[");
	int upper = 30;
        for(i = 0; i < upper; ++i){
            printf("%c", ((i+.5) < score*upper) ? 219 : ' ');
        }
        printf("]\n");

        free_image(in_s);
        free_image(in);

        wait_key_cv(1);
    }
#endif
}


void run_art(int argc, char **argv)
{
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    char *cfg = argv[2];
    char *weights = argv[3];
    demo_art(cfg, weights, cam_index);
}
