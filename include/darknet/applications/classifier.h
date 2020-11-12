#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <darknet/utils/list.h>
#ifdef __cplusplus
extern "C" {
#endif
list *read_data_cfg(char *filename);
#ifdef __cplusplus
}
#endif
#endif
