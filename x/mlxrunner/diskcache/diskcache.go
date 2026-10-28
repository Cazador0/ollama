// Package diskcache is the SSD tier of the MLX runner's prompt cache.
//
// A prompt is cut into fixed-size token chunks, and each chunk's cached state
// is filed under a key that hashes the model identity and every token up to
// the chunk's end. A key therefore names a whole prefix rather than one
// chunk's tokens: sessions that share a system prompt and tool list share
// those chunks, and a prompt's longest cached prefix is found by walking its
// keys until the first miss.
//
// The store is a directory of files with least-recently-used eviction under a
// byte budget. It does not interpret a chunk; the runner writes and reads the
// payload itself (safetensors, through MLX). Several runner processes may
// share one directory: files appear by atomic rename and eviction runs under
// a lock file, so a reader sees either a complete chunk or none.
package diskcache

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

// ext is the file extension of every chunk. MLX appends ".safetensors" to a
// save path that lacks it, so temporary and final names both carry it.
const ext = ".safetensors"

const tmpPrefix = ".tmp-"

// Key names a cached prefix: the model identity plus every token up to the
// end of one chunk.
type Key [sha256.Size]byte

func (k Key) String() string { return hex.EncodeToString(k[:]) }

// Root is the key every chain for a model starts from. identity must change
// whenever the cached state would: model digest, cache layout, dtype.
func Root(identity string) Key {
	return sha256.Sum256([]byte("ollama-mlx-prompt-cache-v1\x00" + identity))
}

// Next returns the key of parent's prefix extended by tokens.
func Next(parent Key, tokens []int32) Key {
	h := sha256.New()
	h.Write(parent[:])
	var b [4]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint32(b[:], uint32(t))
		h.Write(b[:])
	}
	var k Key
	h.Sum(k[:0])
	return k
}

// Chain returns one key per complete chunk of tokens: keys[i] names
// tokens[:(i+1)*chunk]. A trailing partial chunk has no key.
func Chain(root Key, tokens []int32, chunk int) []Key {
	if chunk <= 0 {
		return nil
	}
	keys := make([]Key, 0, len(tokens)/chunk)
	k := root
	for end := chunk; end <= len(tokens); end += chunk {
		k = Next(k, tokens[end-chunk:end])
		keys = append(keys, k)
	}
	return keys
}

// ErrTooLarge reports a chunk that cannot fit in the budget even alone.
var ErrTooLarge = errors.New("diskcache: chunk is larger than the budget")

// Store is a budgeted directory of chunks. A nil *Store is a disabled tier:
// it holds nothing and ignores writes.
type Store struct {
	dir    string
	budget int64

	// mu serializes eviction within this process; the lock file does the
	// same across processes.
	mu sync.Mutex
}

// Open prepares dir as a store limited to budget bytes and removes temporary
// files left by writers that crashed.
func Open(dir string, budget int64) (*Store, error) {
	if budget <= 0 {
		return nil, fmt.Errorf("diskcache: budget must be positive, got %d", budget)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	s := &Store{dir: dir, budget: budget}
	if err := s.walk(func(path string, d fs.DirEntry) error {
		if strings.HasPrefix(d.Name(), tmpPrefix) {
			return os.Remove(path)
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return s, nil
}

// Dir returns the store's directory.
func (s *Store) Dir() string { return s.dir }

// Budget returns the byte budget.
func (s *Store) Budget() int64 { return s.budget }

func (s *Store) path(k Key) string {
	name := k.String()
	return filepath.Join(s.dir, name[:2], name+ext)
}

// Has reports whether k is stored, without counting as a use.
func (s *Store) Has(k Key) bool {
	if s == nil {
		return false
	}
	_, err := os.Stat(s.path(k))
	return err == nil
}

// Get returns the path of k's chunk and marks it recently used. The file can
// be evicted once Get returns; a caller that fails to open or parse it should
// treat that as a miss and Remove the key.
func (s *Store) Get(k Key) (string, bool) {
	if s == nil {
		return "", false
	}
	p := s.path(k)
	if _, err := os.Stat(p); err != nil {
		return "", false
	}
	now := time.Now()
	_ = os.Chtimes(p, now, now) // the modification time is the LRU clock
	return p, true
}

// Longest returns how many leading keys are stored, which is how many chunks
// of the prefix the keys describe can be restored from disk.
func (s *Store) Longest(keys []Key) int {
	for i, k := range keys {
		if !s.Has(k) {
			return i
		}
	}
	return len(keys)
}

// Put stores k's chunk. write receives a temporary path to create the
// payload at; the chunk becomes visible only once write succeeds. An existing
// key is marked used and write is not called.
func (s *Store) Put(k Key, write func(path string) error) error {
	if s == nil {
		return nil
	}
	if _, ok := s.Get(k); ok {
		return nil
	}

	final := s.path(k)
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return err
	}
	var r [8]byte
	if _, err := rand.Read(r[:]); err != nil {
		return err
	}
	tmp := filepath.Join(s.dir, tmpPrefix+k.String()+"-"+hex.EncodeToString(r[:])+ext)
	defer os.Remove(tmp) // no-op after the rename

	if err := write(tmp); err != nil {
		return err
	}
	size, err := syncFile(tmp)
	if err != nil {
		return err
	}
	if size > s.budget {
		return fmt.Errorf("%w: %d bytes, budget %d", ErrTooLarge, size, s.budget)
	}
	if err := os.Rename(tmp, final); err != nil {
		return err
	}
	return s.Evict()
}

// syncFile flushes path to stable storage, so a crash cannot leave a
// truncated chunk under its final name, and returns its size.
func syncFile(path string) (int64, error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	if err := f.Sync(); err != nil {
		return 0, err
	}
	fi, err := f.Stat()
	if err != nil {
		return 0, err
	}
	return fi.Size(), nil
}

// Remove deletes k, for chunks a reader found unreadable.
func (s *Store) Remove(k Key) error {
	if s == nil {
		return nil
	}
	if err := os.Remove(s.path(k)); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return nil
}

type entry struct {
	path  string
	size  int64
	mtime time.Time
}

// Usage returns the bytes and number of chunks stored.
func (s *Store) Usage() (int64, int, error) {
	if s == nil {
		return 0, 0, nil
	}
	entries, err := s.entries()
	if err != nil {
		return 0, 0, err
	}
	var total int64
	for _, e := range entries {
		total += e.size
	}
	return total, len(entries), nil
}

// Evict deletes least-recently-used chunks until the store fits its budget.
// It rescans the directory, so chunks written by other processes count.
func (s *Store) Evict() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	unlock, err := lockFile(filepath.Join(s.dir, ".lock"))
	if err != nil {
		return err
	}
	defer unlock()

	entries, err := s.entries()
	if err != nil {
		return err
	}
	var total int64
	for _, e := range entries {
		total += e.size
	}
	slices.SortFunc(entries, func(a, b entry) int { return a.mtime.Compare(b.mtime) })
	for _, e := range entries {
		if total <= s.budget {
			break
		}
		if err := os.Remove(e.path); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		total -= e.size
	}
	return nil
}

func (s *Store) entries() ([]entry, error) {
	var out []entry
	err := s.walk(func(path string, d fs.DirEntry) error {
		name := d.Name()
		if strings.HasPrefix(name, tmpPrefix) || !strings.HasSuffix(name, ext) {
			return nil
		}
		fi, err := d.Info()
		if errors.Is(err, fs.ErrNotExist) {
			return nil // evicted by another process mid-scan
		} else if err != nil {
			return err
		}
		out = append(out, entry{path: path, size: fi.Size(), mtime: fi.ModTime()})
		return nil
	})
	return out, err
}

// walk visits the files in the store root and its shard directories.
func (s *Store) walk(fn func(path string, d fs.DirEntry) error) error {
	return filepath.WalkDir(s.dir, func(path string, d fs.DirEntry, err error) error {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		} else if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		return fn(path, d)
	})
}
