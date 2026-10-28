package diskcache

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func tokens(n int) []int32 {
	t := make([]int32, n)
	for i := range t {
		t[i] = int32(i * 7)
	}
	return t
}

func writeBytes(n int) func(string) error {
	return func(path string) error { return os.WriteFile(path, make([]byte, n), 0o644) }
}

func open(t *testing.T, budget int64) *Store {
	t.Helper()
	s, err := Open(t.TempDir(), budget)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// age sets a chunk's LRU clock, since back-to-back writes can share an mtime.
func age(t *testing.T, s *Store, k Key, ago time.Duration) {
	t.Helper()
	when := time.Now().Add(-ago)
	if err := os.Chtimes(s.path(k), when, when); err != nil {
		t.Fatal(err)
	}
}

func TestChainNamesWholePrefixes(t *testing.T) {
	root := Root("qwen3.8:27b@sha256:abc")
	all := Chain(root, tokens(10000), 4096)
	if len(all) != 2 {
		t.Fatalf("got %d keys for 10000 tokens in 4096-token chunks, want 2 (partial chunk has none)", len(all))
	}
	if short := Chain(root, tokens(8192), 4096); short[0] != all[0] || short[1] != all[1] {
		t.Fatal("a prompt's chunk keys must not depend on what follows them")
	}

	other := tokens(8192)
	other[0] = 99
	if Chain(root, other, 4096)[1] == all[1] {
		t.Fatal("changing an early token must change every later key")
	}
	if Chain(Root("bonsai-2:27b@sha256:def"), tokens(8192), 4096)[0] == all[0] {
		t.Fatal("different models must never share keys")
	}
	if Chain(root, tokens(100), 0) != nil {
		t.Fatal("chunk size 0 yields no keys")
	}
}

func TestPutGetAndDedup(t *testing.T) {
	s := open(t, 1<<20)
	k := Next(Root("m"), tokens(4))
	if s.Has(k) {
		t.Fatal("empty store has a key")
	}
	if err := s.Put(k, writeBytes(100)); err != nil {
		t.Fatal(err)
	}
	p, ok := s.Get(k)
	if !ok || !strings.HasSuffix(p, ext) {
		t.Fatalf("Get = %q, %v", p, ok)
	}
	if err := s.Put(k, func(string) error { t.Fatal("rewrote an existing chunk"); return nil }); err != nil {
		t.Fatal(err)
	}
	if b, n, _ := s.Usage(); b != 100 || n != 1 {
		t.Fatalf("usage = %d bytes in %d chunks, want 100 in 1", b, n)
	}
}

func TestFailedWriteLeavesNothing(t *testing.T) {
	s := open(t, 1<<20)
	k := Next(Root("m"), tokens(4))
	boom := errors.New("boom")
	err := s.Put(k, func(path string) error {
		_ = os.WriteFile(path, []byte("partial"), 0o644)
		return boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("Put error = %v, want boom", err)
	}
	if s.Has(k) {
		t.Fatal("a failed write became visible")
	}
	leftovers, _ := filepath.Glob(filepath.Join(s.Dir(), tmpPrefix+"*"))
	if len(leftovers) != 0 {
		t.Fatalf("temporary files left behind: %v", leftovers)
	}
}

func TestEvictsLeastRecentlyUsed(t *testing.T) {
	s := open(t, 3000)
	root := Root("m")
	keys := Chain(root, tokens(5*4), 4)
	for i, k := range keys[:3] {
		if err := s.Put(k, writeBytes(1000)); err != nil {
			t.Fatal(err)
		}
		age(t, s, k, time.Duration(10-i)*time.Minute)
	}
	// Reading the oldest chunk makes it the most recently used.
	if _, ok := s.Get(keys[0]); !ok {
		t.Fatal("missing chunk 0")
	}
	if err := s.Put(keys[3], writeBytes(1000)); err != nil {
		t.Fatal(err)
	}
	if !s.Has(keys[0]) || s.Has(keys[1]) || !s.Has(keys[2]) || !s.Has(keys[3]) {
		t.Fatalf("after eviction have %v %v %v %v, want true false true true",
			s.Has(keys[0]), s.Has(keys[1]), s.Has(keys[2]), s.Has(keys[3]))
	}
	if b, _, _ := s.Usage(); b > s.Budget() {
		t.Fatalf("usage %d exceeds budget %d", b, s.Budget())
	}
}

func TestChunkLargerThanBudget(t *testing.T) {
	s := open(t, 500)
	k := Next(Root("m"), tokens(4))
	if err := s.Put(k, writeBytes(501)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("Put error = %v, want ErrTooLarge", err)
	}
	if s.Has(k) {
		t.Fatal("oversized chunk was stored")
	}
}

func TestLongestPrefix(t *testing.T) {
	s := open(t, 1<<20)
	keys := Chain(Root("m"), tokens(4*4), 4)
	for _, k := range keys[:2] {
		if err := s.Put(k, writeBytes(10)); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Put(keys[3], writeBytes(10)); err != nil {
		t.Fatal(err)
	}
	if n := s.Longest(keys); n != 2 {
		t.Fatalf("Longest = %d, want 2: a gap ends the restorable prefix", n)
	}
	if err := s.Put(keys[2], writeBytes(10)); err != nil {
		t.Fatal(err)
	}
	if n := s.Longest(keys); n != len(keys) {
		t.Fatalf("Longest = %d with every chunk stored, want %d", n, len(keys))
	}
}

func TestOpenRejectsEmptyBudget(t *testing.T) {
	if _, err := Open(t.TempDir(), 0); err == nil {
		t.Fatal("a zero budget must be rejected; FromEnv maps 0 to a disabled tier instead")
	}
}

func TestStoresSharingADirectoryShareTheBudget(t *testing.T) {
	dir := t.TempDir()
	a, err := Open(dir, 2000)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Open(dir, 2000)
	if err != nil {
		t.Fatal(err)
	}
	keys := Chain(Root("m"), tokens(3*4), 4)
	if err := a.Put(keys[0], writeBytes(1000)); err != nil {
		t.Fatal(err)
	}
	age(t, a, keys[0], time.Hour)
	if err := b.Put(keys[1], writeBytes(1000)); err != nil {
		t.Fatal(err)
	}
	if err := a.Put(keys[2], writeBytes(1000)); err != nil {
		t.Fatal(err)
	}
	if used, _, _ := a.Usage(); used > 2000 {
		t.Fatalf("shared directory holds %d bytes, budget 2000", used)
	}
	if b.Has(keys[0]) {
		t.Fatal("the oldest chunk across both stores should have gone first")
	}
}

func TestOpenRemovesCrashedWrites(t *testing.T) {
	dir := t.TempDir()
	stale := filepath.Join(dir, tmpPrefix+"dead"+ext)
	if err := os.WriteFile(stale, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(dir, 1<<20); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("stale temporary file survived Open")
	}
}

func TestRemoveAndNilStore(t *testing.T) {
	s := open(t, 1<<20)
	k := Next(Root("m"), tokens(4))
	if err := s.Put(k, writeBytes(10)); err != nil {
		t.Fatal(err)
	}
	if err := s.Remove(k); err != nil || s.Has(k) {
		t.Fatalf("Remove: err %v, still present %v", err, s.Has(k))
	}
	if err := s.Remove(k); err != nil {
		t.Fatalf("removing a missing key: %v", err)
	}

	var off *Store
	if off.Has(k) || off.Longest([]Key{k}) != 0 {
		t.Fatal("a disabled store has chunks")
	}
	if err := off.Put(k, func(string) error { t.Fatal("disabled store wrote"); return nil }); err != nil {
		t.Fatal(err)
	}
	if _, ok := off.Get(k); ok {
		t.Fatal("a disabled store returned a chunk")
	}
}

func TestParseSize(t *testing.T) {
	for in, want := range map[string]int64{
		"32GiB": 32 << 30, "32G": 32 << 30, "32 gib": 32 << 30, "500GB": 500e9,
		"1.5GiB": 3 << 29, "4096": 4096, "512MiB": 512 << 20, "0": 0, "2TB": 2e12,
	} {
		if got, err := ParseSize(in); err != nil || got != want {
			t.Errorf("ParseSize(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, in := range []string{"", "GiB", "-1G", "12 parsecs", "1e9"} {
		if _, err := ParseSize(in); err == nil {
			t.Errorf("ParseSize(%q) succeeded, want an error", in)
		}
	}
}

func TestFromEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_MLX_DISK_CACHE_DIR", "")

	t.Setenv("OLLAMA_MLX_DISK_CACHE", "")
	s, err := FromEnv(dir)
	if err != nil || s == nil || s.Budget() != DefaultBudget || s.Dir() != dir {
		t.Fatalf("default: %v %+v", err, s)
	}

	for _, off := range []string{"off", "0", "FALSE"} {
		t.Setenv("OLLAMA_MLX_DISK_CACHE", off)
		if s, err := FromEnv(dir); err != nil || s != nil {
			t.Fatalf("%q should disable the tier: %v %+v", off, err, s)
		}
	}

	other := filepath.Join(dir, "elsewhere")
	t.Setenv("OLLAMA_MLX_DISK_CACHE", "8GiB")
	t.Setenv("OLLAMA_MLX_DISK_CACHE_DIR", other)
	if s, err := FromEnv(dir); err != nil || s.Budget() != 8<<30 || s.Dir() != other {
		t.Fatalf("configured: %v %+v", err, s)
	}

	t.Setenv("OLLAMA_MLX_DISK_CACHE", "lots")
	if _, err := FromEnv(dir); err == nil {
		t.Fatal("an invalid budget must be an error, not a silent default")
	}
}
