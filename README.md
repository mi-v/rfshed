# rfshed
A reasonably fast CUDA accelerated viewshed generator. Creates a 12k×12k pixels viewshed image in well under a second on a GTX 1660 GPU.

## Requirements
Install Go and CUDA to build.

To run it you will have to provide the 3″ ("90m") SRTM .hgt files.

## Build
`make`

## Run
`LD_LIBRARY_PATH=. ./rfshed`

### Arguments
-h prints help

-lat *latitude* (decimal degrees)

-lon *longitude* (decimal degrees)

-hgtdir *path to .hgt files*

-ah *observer height* (meters above terrain)

-bh *target/receiver height* (meters above terrain)

-cutoff *radius* (meters)

-z *Web Mercator zoom level*

-rr *factor* to simulate refractions by adjusting Earth curvature by the factor specified

-rt *degrees Celsius* to estimate refraction using a specified temperature at the observation point

-outfile *name* where to save the resulting png file

All arguments are optional, with none provided a 200km Everest viewshed is generated to out.png.

### Result
Viewshed is generated in a Web Mercator projected monochrome png file.

## Try online
The same algorithm is used at [sauropod.xyz](https://sauropod.xyz/#37.9236,-121.8068,10z,tb,37.881726,-121.914661y).
