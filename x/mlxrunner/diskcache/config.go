package diskcache

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// DefaultBudget is the SSD budget when OLLAMA_MLX_DISK_CACHE is unset.
const DefaultBudget int64 = 32 << 30

// FromEnv opens the store configured by the environment, or returns nil when
// the tier is disabled:
//
//	OLLAMA_MLX_DISK_CACHE      budget such as 32GiB or 500GB; 0 or off disables
//	OLLAMA_MLX_DISK_CACHE_DIR  directory (default defaultDir)
func FromEnv(defaultDir string) (*Store, error) {
	budget := DefaultBudget
	if v := strings.TrimSpace(os.Getenv("OLLAMA_MLX_DISK_CACHE")); v != "" {
		if strings.EqualFold(v, "off") || strings.EqualFold(v, "false") {
			return nil, nil
		}
		b, err := ParseSize(v)
		if err != nil {
			return nil, fmt.Errorf("OLLAMA_MLX_DISK_CACHE: %w", err)
		}
		if b == 0 {
			return nil, nil
		}
		budget = b
	}
	dir := defaultDir
	if v := strings.TrimSpace(os.Getenv("OLLAMA_MLX_DISK_CACHE_DIR")); v != "" {
		dir = v
	}
	return Open(dir, budget)
}

// ParseSize reads a byte count: a bare number, or a number with a unit.
// KiB, MiB, GiB, TiB and the single letters K, M, G, T are powers of 1024;
// KB, MB, GB and TB are powers of 1000.
func ParseSize(s string) (int64, error) {
	s = strings.TrimSpace(s)
	i := strings.IndexFunc(s, func(r rune) bool { return (r < '0' || r > '9') && r != '.' })
	num, unit := s, ""
	if i >= 0 {
		num, unit = strings.TrimSpace(s[:i]), strings.TrimSpace(s[i:])
	}
	n, err := strconv.ParseFloat(num, 64)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	mult, ok := map[string]float64{
		"": 1, "B": 1,
		"K": 1 << 10, "KIB": 1 << 10, "KB": 1e3,
		"M": 1 << 20, "MIB": 1 << 20, "MB": 1e6,
		"G": 1 << 30, "GIB": 1 << 30, "GB": 1e9,
		"T": 1 << 40, "TIB": 1 << 40, "TB": 1e12,
	}[strings.ToUpper(unit)]
	if !ok {
		return 0, fmt.Errorf("invalid size unit %q in %q", unit, s)
	}
	b := n * mult
	if b > math.MaxInt64 {
		return 0, fmt.Errorf("size %q is too large", s)
	}
	return int64(b), nil
}
