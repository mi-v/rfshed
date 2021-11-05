#define PI 3.141592653589793f
#define ERAD 6371000.0f
#define TORAD (PI / 180.0f)
#define MRDLEN (PI * ERAD)
#define CSLAT (MRDLEN / 180.0f) // cell size along latitude

#define DSTEPS 2048
#define ANGSTEPS 4096

#ifdef __cplusplus
#include <algorithm>
#endif

typedef struct cuErrX {
    int code;
    const char* msg;
    const char* file;
    int line;
} cuErrX;

typedef struct LLi {
    int lat, lon;
    #ifdef __cplusplus
    __host__ __device__ LLi operator+ (LLi b) {return LLi{lat + b.lat, lon + b.lon};};
    __host__ __device__ LLi operator- (LLi b) {return LLi{lat - b.lat, lon - b.lon};};
    #endif
} LLi;

typedef struct {LLi ll; int width, height;} Recti;

typedef struct Refract {
    enum {
        NONE,
        RADIUS,
        TEMP,
    } mode;
    float param;
} Refract;

struct Px2;

typedef struct LL {
    float lat, lon;
    #ifdef __cplusplus
    __host__ __device__ LL toRad() {return LL{lat * TORAD, lon * TORAD};};
    __host__ __device__ LL fromRad() {return LL{lat / TORAD, lon / TORAD};};
    __host__ __device__ LL floor() {return LL{floorf(lat), floorf(lon)};};
    __host__ __device__ LL operator+ (LL b) {return LL{lat + b.lat, lon + b.lon};};
    __host__ __device__ LL operator- (LL b) {return LL{lat - b.lat, lon - b.lon};};
    __host__ __device__ LL operator/ (float b) {return LL{lat / b, lon / b};};
    __host__ __device__ LL& operator+= (LL b) {lat+=b.lat; lon+=b.lon; return *this;};
    __host__ __device__ LL& operator-= (LL b) {lat-=b.lat; lon-=b.lon; return *this;};
    __host__ __device__ operator LLi() {return LLi{int(floorf(lat)), int(floorf(lon))};};
    __host__ __device__ Px2 toPx2(int z);
    #endif
} LL;

typedef struct Px2 {
    int x, y;
    #ifdef __cplusplus
    __host__ __device__ bool operator== (Px2 b) {return x == b.x && y == b.y;};
    __host__ __device__ Px2 operator+ (Px2 b) {return Px2{x+b.x, y+b.y};};
    __host__ __device__ Px2& operator++ () {x++; y++; return *this;};
    __host__ __device__ Px2 operator++ (int) {Px2 t(*this); x++; y++; return t;};
    __host__ __device__ Px2 operator- (Px2 b) {return Px2{x-b.x, y-b.y};};
    __host__ __device__ Px2 operator% (int b) {return Px2{x%b, y%b};};
    __host__ __device__ Px2& operator*= (int b) {x *= b; y *= b; return *this;};
    __host__ __device__ Px2& operator/= (int b) {x /= b; y /= b; return *this;};
    __host__ __device__ Px2& operator>>= (int b) {x >>= b; y >>= b; return *this;};
    __host__ __device__ Px2& operator&= (int b) {x &= b; y &= b; return *this;};
    __host__ __device__ Px2 friend operator* (Px2 a, int b) {return a *= b;};
    __host__ __device__ Px2 friend operator/ (Px2 a, int b) {return a /= b;};
    __host__ __device__ Px2 friend operator>> (Px2 a, int b) {return a >>= b;};
    __host__ __device__ Px2 friend operator& (Px2 a, int b) {return a &= b;};
    __host__ __device__ operator float() {return hypotf(x, y);};
    __host__ __device__ Px2 cropY(int height) {return Px2{x, y <= 0 ? 0 : y <= height ? y : height};};
    __host__ __device__ LL toLL(int zoom) {
        return LL{
            2 * atanf(expf(PI * (1 - float(y) / (128 << zoom)))) - PI / 2,
            (2 * PI * x) / (256 << zoom) - PI
        };
    };
    #endif
} Px2;

typedef struct PxRect {
    Px2 P, Q;
    #ifdef __cplusplus
        __host__ __device__ int w() {return Q.x - P.x;};
        __host__ __device__ int h() {return Q.y - P.y;};
        __host__ __device__ int wh() {return w() * h();};
        __host__ __device__ int indexOf(Px2 pt) {return w() * (pt.y - P.y) + (pt.x - P.x);};
        __host__ __device__ int contains(Px2 pt) {return pt.x >= P.x && pt.y >= P.y && pt.x < Q.x && pt.y < Q.y;};
        __host__ __device__ PxRect& operator*= (int b) {P*=b; Q*=b; return *this;};
        __host__ __device__ PxRect& operator/= (int b) {P/=b; Q/=b; return *this;};
        __host__ __device__ Px2 operator[] (int i) {return Px2{P.x + i%w(), P.y + i/w()};};
        __host__ __device__ PxRect friend operator* (PxRect a, int b) {return a *= b;};
        __host__ __device__ PxRect friend operator/ (PxRect a, int b) {return a /= b;};
        __host__ __device__ PxRect cropY(int height) {return PxRect{P.cropY(height), Q.cropY(height)};};
    #endif
} PxRect;

typedef struct Image {
    void* buf;
    uint64_t ptr;
    PxRect rect;
    cuErrX error;
    #ifdef __cplusplus
        int w() {return rect.Q.x - rect.P.x;};
        int h() {return rect.Q.y - rect.P.y;};
        int wh() {return w() * h();};
        int bytes() {return ((w()+7) / 8) * h();};
    #endif
} Image;

#ifdef __cplusplus
__host__ __device__ Px2 LL::toPx2(int zoom) {
    return Px2{
        int((256 << zoom) * ((lon + PI) / (2 * PI))),
        int((128 << zoom) * (1 - logf(tanf(PI / 4 + lat / 2)) / PI))
    };
};
#endif
