package hgtmgr

import (
    "os"
    "math"
    "errors"
    "github.com/mi-v/rfshed/cuhgt"
    "github.com/mi-v/rfshed/latlon"
    . "github.com/mi-v/rfshed/conf"
)

type HgtMgr struct {
    hgtdir string
}

type Grid struct {
    ptrMap []uint64
    evtReady uint64
    rect latlon.Recti
    mgr *HgtMgr
    mask []bool
}

func New(dir string) (m *HgtMgr, err error) {
    fi, err := os.Stat(dir)
    if err != nil {
        return
    }

    if !fi.IsDir() {
        err = errors.New("not a dir: " + dir)
        return
    }

    m = &HgtMgr{
        hgtdir: dir,
    }

    return
}

func (m *HgtMgr) GetGrid(rect latlon.Recti, mask []bool) *Grid {
    g := &Grid{
        ptrMap: make([]uint64, 0, rect.Width * rect.Height),
        rect: rect,
        mgr: m,
        mask: mask,
    }
    g.rect.Apply(func (ll latlon.LLi) {
        if mask[len(g.ptrMap)] == false {
            g.ptrMap = append(g.ptrMap, 0)
            return
        }
        ll = ll.Wrap()
        ptr := cuhgt.Fetch(ll, m.hgtdir)
        if ptr == 0 {
            g.mask[len(g.ptrMap)] = false;
            g.ptrMap = append(g.ptrMap, 0)
            return
        } else {
            g.ptrMap = append(g.ptrMap, ptr)
        }
    })
    return g
}

func (m *HgtMgr) GetGridAround(ll latlon.LL, cutoff float64) (grid *Grid) {
    cutoffD := cutoff / CSLAT + 0.1
    r := latlon.RectiFromRadius(ll, cutoffD)
    mask := make([]bool, 0, r.Width * r.Height)
    r.Apply(func (cll latlon.LLi) {
        cllf := cll.Float()
        dY := clamp(ll.Lat, cllf.Lat, cllf.Lat + 1) - ll.Lat
        dX := (clamp(ll.Lon, cllf.Lon, cllf.Lon + 1) - ll.Lon) * math.Cos((ll.LatR() + cllf.LatR()) / 2)
        mask = append(mask, dX * dX + dY * dY < cutoffD * cutoffD)
    })

    return m.GetGrid(r, mask)
}

func (g *Grid) PtrMap() []uint64 {
    return g.ptrMap
}

func (g *Grid) EvtReady() uint64 {
    return g.evtReady
}

func (g *Grid) Rect() latlon.Recti {
    return g.rect
}

func (g *Grid) Free() {
    for _, ptr := range g.ptrMap {
        cuhgt.Free(ptr);
    }
}

func clamp(v, min, max float64) float64 {
    return math.Min(math.Max(v, min), max)
}
