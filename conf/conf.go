// +build !dev

package conf

import "math"

const ERAD = 6371000.0
const MRDLEN = math.Pi * ERAD
const CSLAT = MRDLEN / 180 // cell size along latitude
