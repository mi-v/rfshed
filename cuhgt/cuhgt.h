#define HGTSIZE (1201 * 1201 * 2)

typedef struct cuErrX {
    int code;
    const char* msg;
    const char* file;
    int line;
} cuErrX;

typedef struct UploadResult {
    uint64_t ptr;
    cuErrX error;
} UploadResult;
