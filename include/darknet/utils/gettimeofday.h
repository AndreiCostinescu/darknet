#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define CLOCK_REALTIME (1)
#endif

#include <winsock2.h>
#include <windows.h>
#include <winnt.h>
#include <stdint.h>
#include <time.h>
#include <darknet/darknet.h>

#define BILLION (1E9)

#ifndef timersub
#define timersub(a, b, result)                       \
  do {                                               \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;    \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec; \
    if ((result)->tv_usec < 0) {                     \
      --(result)->tv_sec;                            \
      (result)->tv_usec += 1000000;                  \
    }                                                \
  } while (0)
#endif // timersub

#ifdef __cplusplus
extern "C" {
#endif

struct timespec {
        time_t tv_sec;
        long tv_nsec;
};

static unsigned char g_first_time = 1;
static LARGE_INTEGER g_counts_per_sec;

int gettimeofday(struct timeval*, struct timezone*);
int clock_gettime(int, struct timespec*);

#ifdef __cplusplus
}
#endif
