#ifndef PARSER_H
#define PARSER_H

#include <darknet/network.h>

#ifdef __cplusplus
extern "C" {
#endif
network parse_network_cfg(const char *filename);
network parse_network_cfg_verbose(const char *filename, int verbose);
network parse_network_cfg_custom(const char *filename, int batch, int time_steps);
network parse_network_cfg_custom_verbose(const char *filename, int batch, int time_steps, int verbose);
void save_network(network net, const char *filename);
void save_weights(network net, const char *filename);
void save_weights_upto(network net, const char *filename, int cutoff);
void save_weights_double(network net, const char *filename);
void load_weights(network *net, const char *filename);
void load_weights_upto_verbose(network *net, const char *filename, int cutoff, int verbose);
void load_weights_upto(network *net, const char *filename, int cutoff);

#ifdef __cplusplus
}
#endif
#endif
