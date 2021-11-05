package latlon

import m "math"

type LL struct {
    Lat float64
    Lon float64
}

type LLi struct {
    Lat int
    Lon int
}

type DifLL LL
type DifLLi LLi

type Recti struct {
    LLi
    Width, Height int
}

type Path []LL

var toRad = m.Pi / 180

func (ll LL) LatR() float64 {
    return ll.Lat * toRad
}

func (ll LL) LonR() float64 {
    return ll.Lon * toRad
}

func (ll LL) MerLat() float64 {
    return MerLatR(ll.LatR())
}

func (ll LL) Floor() LL {
    return LL{m.Floor(ll.Lat), m.Floor(ll.Lon)}
}

func (ll LL) Wrap() LL {
    if ll.Lon >= -180 && ll.Lon < 180 {
        return ll
    }
    ll.Lon = m.Remainder(ll.Lon, 360)
    if ll.Lon == 180 {
        ll.Lon = -180
    }
    return ll
}

func (ll LLi) Wrap() LLi {
    for ll.Lon < -180 {
        ll.Lon += 360
    }
    for ll.Lon >= 180 {
        ll.Lon -= 360
    }
    return ll
}

func (r Recti) Apply(f func(LLi)) {
    for ll := r.LLi; ll.Lat < r.Lat + r.Height; ll.Lat++ {
        for ll2 := ll; ll2.Lon < r.Lon + r.Width; ll2.Lon++ {
            f(ll2)
        }
    }
}

func MerLatR(lat float64) float64 {
    return m.Log(m.Tan(m.Pi / 4 + lat / 2))
}

func NewR(lat, lon float64) LL {
    return LL{lat / toRad, lon / toRad}
}

func (ll LL) Int() LLi {
    return LLi{int(ll.Lat), int(ll.Lon)}
}

func (ll LLi) Float() LL {
    return LL{float64(ll.Lat), float64(ll.Lon)}
}

func (ll LL) Add(d DifLL) LL {
    ll.Lat += d.Lat
    ll.Lon += d.Lon
    return ll
}

func (ll LL) Sub(d DifLL) LL {
    ll.Lat -= d.Lat
    ll.Lon -= d.Lon
    return ll
}

func RectiFromRadius(ll LL, latRadius float64) Recti {
    ofs := DifLL{Lat: latRadius}
    ofs.Lon = ofs.Lat / m.Cos(ll.LatR())
    swCorner := ll.Sub(ofs).Floor().Int()
    neCorner := ll.Add(ofs).Floor().Int()
    r := Recti{
        LLi: swCorner,
        Width: neCorner.Lon - swCorner.Lon + 1,
        Height: neCorner.Lat - swCorner.Lat + 1,
    }
    return r
}
