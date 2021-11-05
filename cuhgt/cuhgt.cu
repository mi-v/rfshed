#include <stdint.h>
#include <unistd.h>
#include <algorithm>
#include "cuhgt.h"

#define cuErr(call)  {cudaError_t err; if (cudaSuccess != (err=(call))) throw cuErrX{err, cudaGetErrorString(err), __FILE__, __LINE__};}

__global__ void byteswap16(int32_t* Hgt) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    if (row < 1201 && col < 602) {
        int ofs = row * 640 + col;
        Hgt[ofs] = ((Hgt[ofs] >> 8) & 0x00ff00ff) | ((Hgt[ofs] << 8) & 0xff00ff00);
    }
}

void* Hgt_ha;
void* Hgt_hb;
cudaEvent_t *xfer_a = new cudaEvent_t;
cudaEvent_t *xfer_b = new cudaEvent_t;

extern "C" {
    UploadResult uploadHgt(int fd) {
        UploadResult Res = {0};
        short* Hgt_d;
        try {
            cuErr(cudaMalloc(&Hgt_d, 1280 * 1201 * sizeof(short)));
            cudaEventSynchronize(*xfer_a);
            read(fd, Hgt_ha, HGTSIZE);
            cuErr(cudaMemcpy2DAsync(
                Hgt_d,
                1280 * sizeof(short),
                Hgt_ha,
                1201 * sizeof(short),
                1201 * sizeof(short),
                1201,
                cudaMemcpyHostToDevice
            ));
            cuErr(cudaEventRecord(*xfer_a));
            byteswap16<<<dim3(19, 38), dim3(32, 32), 0>>>(reinterpret_cast<int32_t*>(Hgt_d));
            std::swap(xfer_a, xfer_b);
            std::swap(Hgt_ha, Hgt_hb);
            Res.ptr = (uint64_t)Hgt_d;
        } catch (cuErrX error) {
            Res.error = error;
        }
        return Res;
    }

    void freeHgt(uint64_t ptr) {
        cudaFree((short*)ptr);
    }

    cuErrX cuhgtInit() {
        cuErrX err = {0};
        try {
            cuErr(cudaHostAlloc(&Hgt_ha, HGTSIZE, cudaHostAllocWriteCombined));
            cuErr(cudaHostAlloc(&Hgt_hb, HGTSIZE, cudaHostAllocWriteCombined));
            cuErr(cudaEventCreateWithFlags(xfer_a, cudaEventDisableTiming));
            cuErr(cudaEventCreateWithFlags(xfer_b, cudaEventDisableTiming));
        } catch (cuErrX error) {
            err = error;
        }
        return err;
    }
}
