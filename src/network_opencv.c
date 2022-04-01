#include <darknet/network_opencv.h>
#include <darknet/darknet.h>
#include <darknet/images/image_opencv.h>
#include <darknet/layers/convolutional_layer_opencv.h>

void visualize_network(network net) {
    image *prev = 0;
    int i;
    char buff[256];
    for (i = 0; i < net.n; ++i) {
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    }
}

float train_network_waitkey(network net, data d, int wait_key) {
    assert(d.X.rows % net.batch == 0);
    int batch = net.batch;
    int n = d.X.rows / batch;
    float *X = (float *) xcalloc(batch * d.X.cols, sizeof(float));
    float *y = (float *) xcalloc(batch * d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for (i = 0; i < n; ++i) {
        get_next_batch(d, batch, i * batch, X, y);
        net.current_subdivision = i;
        float err = train_network_datum(net, X, y);
        sum += err;
        if (wait_key) wait_key_cv(5);
    }
    (*net.cur_iteration) += 1;
#ifdef GPU
    update_network_gpu(net);
#else   // GPU
    update_network(net);
#endif  // GPU
    free(X);
    free(y);
    return sum / (float) (n * batch);
}
