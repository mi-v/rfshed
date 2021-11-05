#include <cuda_profiler_api.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "cuvshed.h"

#define sindf(a) sinpif((a) / 180)
#define cosdf(a) cospif((a) / 180)

#define cuErr(call)  {cudaError_t err; if (cudaSuccess != (err=(call))) throw cuErrX{err, cudaGetErrorString(err), __FILE__, __LINE__};}
#define bswap32(i)  (((i & 0xFF) << 24) | ((i & 0xFF00) << 8) | ((i & 0xFF0000) >> 8) | ((i & 0xFF000000) >> 24))

__device__ float interp(float a, float b, float f)
{
    return a + f * (b - a);
}

__device__ float seaDistR(LL p0, LL p1)
{
    LL d = p1 - p0;

    return 2 * asinf(sqrtf(
        sinf(d.lat/2) * sinf(d.lat/2) + cosf(p0.lat) * cosf(p1.lat) * sinf(d.lon/2) * sinf(d.lon/2)
    ));
}

__device__ float abElev(float a, float b, float d, Refract refract)
{
    float ar = a / ERAD;
    float br = b / ERAD;

    float k = 1;

    switch (refract.mode) {
        case refract.RADIUS:
        k = 1 / refract.param;
        break;

        case refract.TEMP:
        float L = 0.0098f;
        float P0 = 1013.0f;
        float h = (a + b) / 2;
        float T0 = refract.param + 273 + L * a;
        float Th = T0 - L * h;
        //float P = P0 * powf(1 - L * h / T0, 3.5f);
        float P = 1 - L * h / T0;
        P = P0 * P * P * P * sqrtf(P);
        k = 503.0f * (0.0343f - L) * P / (Th * Th);
        k = 1 - k;
        break;
    }

    return (ar + d * sinf(k * d/2) - br * cos(k * d)) /
        (d * cos(k * d/2) + br * sin(k * d));
}

__device__ float hgtQuery(const short** __restrict__ HgtMap, Recti rect, LL ll)
{
    LLi lliof = LLi(ll) - rect.ll;
    const short* hgtCell = HgtMap[lliof.lat * rect.width + lliof.lon];
    if (!hgtCell) return 0;

    ll -= ll.floor();

    float Y = ll.lat * 1200;
    float X = ll.lon * 1200;

    int Xi = floorf(X);
    int Yi = floorf(Y);

    float Xf = X - Xi;
    float Yf = Y - Yi;

    int ofs = 1200 * 1280 - Yi * 1280 + Xi;
    float a = hgtCell[ofs];
    float b = hgtCell[ofs + 1];
    float c = hgtCell[ofs - 1280];
    float d = hgtCell[ofs - 1280 + 1];

    return (a * (1 - Xf) + b * Xf) * (1 - Yf) + (c * (1 - Xf) + d * Xf) * Yf;
}

__global__ void Query(const short** __restrict__ HgtMap, Recti rect, LL ll, float* result) {
    if (blockIdx.x || blockIdx.y || threadIdx.x || threadIdx.y) return;
    *result = hgtQuery(HgtMap, rect, ll);
}

__global__ void altQuery(const short** __restrict__ HgtMap, Recti rect, LL ll, float myH, float* result) {
    if (blockIdx.x || blockIdx.y || threadIdx.x || threadIdx.y) return;
    *result = myH + hgtQuery(HgtMap, rect, ll);
}

__global__ void doScape(const short** __restrict__ HgtMap, Recti hgtRect, float* __restrict__ AzEleD, const float* __restrict__ myAlt, LL myL, Refract refract, float cuton, float dstep)
{
    int az = blockIdx.x * blockDim.x + threadIdx.x;
    int distN = blockIdx.y * blockDim.y + threadIdx.y;
    float dist = cuton + dstep * distN * (distN + 1) / 2;
    float rDist = dist / ERAD;

    float azR = 2 * PI * az / ANGSTEPS;

    LL myR = myL.toRad();
    LL ptR = {asinf(sindf(myL.lat) * cosf(rDist) + cosdf(myL.lat) * sinf(rDist) * cosf(azR))}; // <- lat only! lon follows
    ptR.lon = myR.lon + atan2f(sinf(azR) * sinf(rDist) * cosdf(myL.lat), cosf(rDist) - sindf(myL.lat) * sinf(ptR.lat));

    LL ptL = ptR.fromRad();

    float hgt = hgtQuery(HgtMap, hgtRect, ptL);

    float elev = abElev(*myAlt, hgt, rDist, refract);

    int ofs = distN * ANGSTEPS + az;
    AzEleD[ofs] = elev;
}

__global__ void elevProject(float* AzEleD)
{
    int az = blockIdx.x * blockDim.x + threadIdx.x;
    float elev = AzEleD[ANGSTEPS + az];
    for (int distN = 1; distN < DSTEPS; distN++) {
        int ofs = distN * ANGSTEPS + az;
        if (AzEleD[ofs] > elev) {
            AzEleD[ofs] = elev;
        } else {
            elev = AzEleD[ofs];
        }
    }
}

__global__ void doVisMap(
    const short** __restrict__ HgtMap,
    Recti hgtRect,
    const float* __restrict__ AzEleD,
    LL myL,
    const float* __restrict__ myAlt,
    float theirH,
    Refract refract,
    Px2 pxBase,
    unsigned char* __restrict__ visMap,
    int zoom,
    float cuton,
    float dstep
)
{
    Px2 imgPx = {
        int(blockIdx.x * blockDim.x + threadIdx.x),
        int(blockIdx.y * blockDim.y + threadIdx.y)
    };
    int visMapWidth = blockDim.x * gridDim.x;

    Px2 ptPx = pxBase + imgPx;

    LL ptR = ptPx.toLL(zoom);
    LL ptL = ptR.fromRad();

    LL myR = myL.toRad();
    float distR = seaDistR(myR, ptR);

    float hgt = hgtQuery(HgtMap, hgtRect, ptL) + theirH;

    float elev = abElev(*myAlt, hgt, distR, refract);

    float dist = ERAD * distR;
    int distN = floorf((sqrtf(1 + 8 * (dist - cuton) / dstep) - 1) / 2);
    float distNdist = cuton + dstep * distN * (distN + 1) / 2;

    float azR = atan2f(sinf(ptR.lon - myR.lon) * cosf(ptR.lat), cosf(myR.lat) * sinf(ptR.lat) - sinf(myR.lat) * cosf(ptR.lat) * cosf(ptR.lon - myR.lon));
    while (azR < 0) {
        azR += 2 * PI;
    }
    float azi;
    float azf = modff(ANGSTEPS * azR / (2 * PI), &azi);
    int az = azi;

    bool visible = false;

    Px2 myPx = myR.toPx2(zoom);

    if (dist < cuton || ptPx == myPx) {
        visible = true;
    } else if (distN >= 0 && distN < DSTEPS && elev - 0.00005 <= interp(AzEleD[distN * ANGSTEPS + az % ANGSTEPS], AzEleD[distN * ANGSTEPS + (az+1) % ANGSTEPS], azf)) {

        float pxDist = float(ptPx - myPx);

        LL llStep = (ptL - myL) / pxDist;

        float distStep = dist / pxDist;
        float distRStep = distR / pxDist;

        visible = true;
        int i = 10;
        while (dist > distNdist && i--) {
            dist -= distStep;
            distR -= distRStep;
            ptL -= llStep;

            hgt = hgtQuery(HgtMap, hgtRect, ptL);

            float stepElev = abElev(*myAlt, hgt, distR, refract);

            if (stepElev < elev) {
                visible = false;
                break;
            }
        }
    }

    int visMapOffset = visMapWidth * imgPx.y + imgPx.x;

    unsigned bb = __brev(__ballot_sync(~0, visible));
    if (threadIdx.x % 8 == 0) {
        unsigned char b = bb >> (24 - threadIdx.x % 32); // warp is 32 threads, get the 8 bits we need into b
        visMap[visMapOffset / 8] = b;
    }
}

extern "C" {
    Image makeImage(LL myL, int zoom, int myH, int theirH, float cuton, float cutoff, Refract refract, const uint64_t* HgtMapIn, Recti hgtRect) {
        const short** HgtMap_d;
        float* AzEleD_d;
        float* myAlt_d;
        unsigned char* Imbuf_d;
        Image Img = {nullptr};
        float dstep = 2 * (cutoff - cuton) / (DSTEPS * (DSTEPS - 1));

        try {
            cuErr(cudaMalloc(&HgtMap_d, hgtRect.width * hgtRect.height * sizeof(uint64_t)));
            cuErr(cudaMalloc(&AzEleD_d, ANGSTEPS * DSTEPS * sizeof(float)));
            cuErr(cudaMalloc(&myAlt_d, sizeof(float)));

            cuErr(cudaMemcpyAsync(HgtMap_d, HgtMapIn, hgtRect.width * hgtRect.height * sizeof(uint64_t), cudaMemcpyHostToDevice));

            altQuery<<<1, 1>>>(HgtMap_d, hgtRect, myL, myH, myAlt_d);

            doScape<<<dim3(ANGSTEPS/32, DSTEPS/32), dim3(32, 32)>>>(
                HgtMap_d,
                hgtRect,
                AzEleD_d,
                myAlt_d,
                myL,
                refract,
                cuton,
                dstep
            );
            cuErr(cudaGetLastError());

            elevProject<<<dim3(ANGSTEPS/256), dim3(256)>>>(AzEleD_d);
            cuErr(cudaGetLastError());

            LL myR = myL.toRad();
            LL rngR = {cutoff / ERAD};
            rngR.lon = -rngR.lat / cosf(myR.lat);

            Img.rect.P = (myR + rngR).toPx2(zoom);
            Img.rect.P.x &= ~255;
            Img.rect.P.y &= ~255;
            Img.rect.Q = (myR - rngR).toPx2(zoom);
            Img.rect.Q.x |= 255;
            Img.rect.Q.y |= 255;
            Img.rect.Q ++;
            //Img.rect = Img.rect.cropY(256 << zoom);
            //printf("Image: %d x %d, %d bytes, z: %d  lat=%f  lon=%f\n", Img.rect.w(), Img.rect.h(), (Img.rect.wh() + 7) / 8, zoom, myL.lat, myL.lon);

            cuErr(cudaMalloc(&Imbuf_d, Img.bytes()));
            cuErr(cudaMemset(Imbuf_d, 0, Img.bytes()));

            doVisMap<<<dim3(Img.rect.w()/32, Img.rect.h()/32), dim3(32, 32)>>>(
                HgtMap_d,
                hgtRect,
                AzEleD_d,
                myL,
                myAlt_d,
                theirH,
                refract,
                Img.rect.P,
                Imbuf_d,
                zoom,
                cuton,
                dstep
            );
            cuErr(cudaGetLastError());

            cudaFree(HgtMap_d);
            cudaFree(AzEleD_d);

            cuErr(cudaMallocHost(&Img.buf, Img.bytes()));
            Img.ptr = (uint64_t)Img.buf;

            cuErr(cudaMemcpy(Img.buf, Imbuf_d, Img.bytes(), cudaMemcpyDeviceToHost));
            cudaFree(Imbuf_d);
        } catch (cuErrX error) {
            Img.error = error;
        }

        return Img;
    }

    void freeImage(uint64_t ptr) {
        cudaFreeHost((void*)ptr);
    }
}
