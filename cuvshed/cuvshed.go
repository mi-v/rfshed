package cuvshed

import (
    _ "rfshed/cuinit"
    "rfshed/hgtmgr"
    "rfshed/latlon"
    "fmt"
    "errors"
    "log"
)

/*
#include <stdint.h>
#include <stdlib.h>
#include "cuvshed.h"
#cgo LDFLAGS: -L../ -lcuvshed

Image makeImage(LL myL, int zoom, int myH, int theirH, float cuton, float cutoff, Refract rfr, const uint64_t* HgtMapIn, Recti hgtRect);
void freeImage(uint64_t ptr);
*/
import "C"

type Image struct {
    Pix []byte
    ptr uint64
    W, H int
}

func (i *Image) Bytes() int {
    return ((i.W+7) / 8) * i.H;
}

func (i *Image) Free() {
    i.Pix = nil
    C.freeImage(C.ulong(i.ptr))
    i.ptr = 0
}

func MakeImage(ll latlon.LL, zoom int, myH int, theirH int, cuton float64, cutoff float64, rfrMode int, rfrParam float64, g *hgtmgr.Grid) (*Image, error) {
    hgtmap := g.PtrMap()
    rect := g.Rect()
    cImg := C.makeImage(
        C.LL{C.float(ll.Lat), C.float(ll.Lon)},
        C.int(zoom),
        C.int(myH),
        C.int(theirH),
        C.float(cuton),
        C.float(cutoff),
        C.Refract{uint32(rfrMode), C.float(rfrParam)},
        (*C.ulong)(&hgtmap[0]),
        C.Recti{
            C.LLi{C.int(rect.Lat), C.int(rect.Lon)},
            C.int(rect.Width),
            C.int(rect.Height),
        },
    )
    if (cImg.error.msg != nil) {
        log.Fatalf("CUDA error: %d %s in %s:%d", cImg.error.code, C.GoString(cImg.error.msg), C.GoString(cImg.error.file), cImg.error.line)
        return nil, errors.New(fmt.Sprintf("CUDA error: %d %s in %s:%d", cImg.error.code, C.GoString(cImg.error.msg), C.GoString(cImg.error.file), cImg.error.line))
    }

    img := &Image{
        W: int(cImg.rect.Q.x - cImg.rect.P.x),
        H: int(cImg.rect.Q.y - cImg.rect.P.y),
        ptr: uint64(cImg.ptr),
    }

    img.Pix = (*[1<<30]byte)(cImg.buf)[:img.Bytes():img.Bytes()]

    return img, nil
}
