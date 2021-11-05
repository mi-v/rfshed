#include "cuinit.h"

extern "C" {
    void cuinitInit() {
        cudaSetDevice(0);
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync | cudaDeviceMapHost);
    }
}
