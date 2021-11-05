package cuinit

// #include "cuinit.h"
// void cuinitInit();
// #cgo LDFLAGS: -L../ -lcuinit
import "C"

func init() {
    C.cuinitInit()
}
