package main

import (
    "os"
    "log"
    "rfshed/latlon"
    "rfshed/hgtmgr"
    "rfshed/cuvshed"
    "strconv"
    "flag"
    "image"
    "image/color"
    "github.com/mi-v/img1b"
    "github.com/mi-v/img1b/png"
)

const (
    refractNone = iota
    refractRadius = iota
    refractTemperature = iota
)

var point latlon.LL
var outfile string
var hgtdir string
var obsAh, obsBh int
var rfrMode int
var rfrParam float64
var cuton int
var cutoff int
var zoom int

func init() {
    flag.Float64Var(&point.Lat, "lat", 27.9886, "latitude")
    flag.Float64Var(&point.Lon, "lon", 86.9253, "longitude")
    flag.IntVar(&zoom, "z", 11, "Web Mercator zoom level")
    flag.StringVar(&outfile, "outfile", "out.png", "resulting png name")
    flag.StringVar(&hgtdir, "hgtdir", "./hgt", "directory with 3″ SRTM .hgt files")
    flag.IntVar(&obsAh, "ah", 2, "observer height above terrain, meters")
    flag.IntVar(&obsBh, "bh", 0, "target height above terrain, meters")

    flag.Func("rr", "simulate refraction by adjusting Earth curvature by the factor specified", func(s string) error {
        if rfrMode != refractNone && rfrMode != refractRadius {
            log.Fatal("Please use only one refraction mode at a time")
        }
        rfrMode = refractRadius
        var err error
        rfrParam, err = strconv.ParseFloat(s, 64)
        return err
    })

    flag.Func("rt", "estimate refraction using a specified temperature at the observation point (degrees Celsius)", func(s string) error {
        if rfrMode != refractNone && rfrMode != refractTemperature {
            log.Fatal("Please use only one refraction mode at a time")
        }
        rfrMode = refractTemperature
        var err error
        rfrParam, err = strconv.ParseFloat(s, 64)
        return err
    })

    flag.IntVar(&cuton, "cuton", 80, "cuton distance, meters")
    flag.IntVar(&cutoff, "cutoff", 200000, "cutoff distance, meters")
}

func main() {
    flag.Parse()

    if point.Lat > 85 || point.Lat < -85 {
        log.Fatal("latitude is out of the supported range")
    }

    hm, err := hgtmgr.New(hgtdir)
    if err != nil {
        log.Fatal(err)
    }

    grid := hm.GetGridAround(point, float64(cutoff));
    defer grid.Free()

    bmp, err := cuvshed.MakeImage(
        point,
        zoom,
        obsAh,
        obsBh,
        float64(cuton),
        float64(cutoff),
        rfrMode,
        rfrParam,
        grid,
    )

    fd, err := os.Create(outfile)
    if err != nil {
        log.Fatal(err)
    }
    png.Encode(fd, &img1b.Image{
        Pix: bmp.Pix,
        Stride: (bmp.W+7) / 8,
        Rect: image.Rect(0, 0, bmp.W, bmp.H),
        Palette: color.Palette{
            color.Black,
            color.White,
        },
    })
    fd.Close()

    bmp.Free()
}
