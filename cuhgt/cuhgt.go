package cuhgt

import (
    "os"
    "fmt"
    _ "github.com/mi-v/rfshed/cuinit"
    "github.com/mi-v/rfshed/latlon"
    "log"
)

// #include <stdint.h>
// #include <stdlib.h>
// #include "cuhgt.h"
// UploadResult uploadHgt(int fd);
// void freeHgt(uint64_t ptr);
// cuErrX cuhgtInit();
// #cgo LDFLAGS: -L../ -lcuhgt
import "C"

const hgtFileSize = 1201 * 1201 * 2

func Fetch(ll latlon.LLi, dir string) (ptr uint64) {
    hgtName := dir + "/" + mkHgtName(ll)
    hf, err := os.Open(hgtName)
    if err != nil {
        log.Printf("%s not found", hgtName)
        return
    }
    defer hf.Close()
    hfs, err := hf.Stat()
    if (err != nil || hfs.Size() != hgtFileSize) {
        return
    }

    cUR := C.uploadHgt(C.int(hf.Fd()))
    if cUR.error.msg != nil {
        log.Fatalf("CUDA error: %d %s in %s:%d", cUR.error.code, C.GoString(cUR.error.msg), C.GoString(cUR.error.file), cUR.error.line)
    }
    return uint64(cUR.ptr)
}

func Free(ptr uint64) {
    C.freeHgt(C.ulong(ptr));
}

func mkHgtName(ll latlon.LLi) string {
    ns := 'N'
    if ll.Lat < 0 {
        ns = 'S'
        ll.Lat = -ll.Lat
    }

    ew := 'E'
    if ll.Lon < 0 {
        ew = 'W'
        ll.Lon = -ll.Lon
    }

    return fmt.Sprintf("%c%02d%c%03d.hgt", ns, ll.Lat, ew, ll.Lon)
}

func init() {
    cErr := C.cuhgtInit()
    if cErr.code != 0 {
        log.Fatal("could not init cuhgt")
    }
}
