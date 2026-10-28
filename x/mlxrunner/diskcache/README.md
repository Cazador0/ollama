# diskcache: the SSD tier of the MLX prompt cache

The MLX runner keeps computed prompt state (attention KV and linear-attention
recurrent state) in a prefix trie in unified memory, paging out diverged
branches until `maxPagedOutBytes` (8 GiB) and evicting after that. Evicted
state, and everything when the model unloads, must be recomputed. This
package stores that state on disk in fixed-size token chunks so a session can
resume without recomputation after eviction, unload or restart, and so
sessions sharing a prefix (system prompt, tool list) share chunks.

## Store (done)

- Keys: `Root(identity)` then `Next(parent, tokens)` per chunk; `Chain` keys a
  whole prompt. A key names the prefix up to its chunk's end.
- `Put` writes through a temporary path, fsyncs, and renames, then evicts
  least-recently-used chunks (by modification time) to fit the budget.
  Eviction rescans the directory under a lock file, so runner processes that
  share the directory share the budget.
- `Longest(keys)` is how many leading chunks of a prompt are restorable.

| Variable | Meaning | Default |
|---|---|---|
| `OLLAMA_MLX_DISK_CACHE` | budget (`32GiB`, `500GB`); `0`/`off` disables | `32GiB` |
| `OLLAMA_MLX_DISK_CACHE_DIR` | directory | chosen by the runner |

## Runner integration (next, against the synced upstream runner)

1. Chunk boundaries: schedule prefill snapshots every chunk (default 4096
   tokens, `OLLAMA_MLX_DISK_CACHE_CHUNK`) instead of every 8192.
2. Write-behind: at each boundary, save the chunk with
   `mlx.SaveSafetensorsWithMetadata` from the MLX worker: per attention layer
   the K/V rows for `[start, end)`, per recurrent layer the conv and delta
   state at `end`, metadata with offsets and identity. Keep at most two chunks
   in flight; drop rather than block prefill when the queue is full.
3. Restore: in the cache's `begin`, compute `Chain` over the request tokens.
   When `Longest` covers more than the trie match, load those chunks
   (memory-mapped through `mlx.LoadSafetensorsNative`), concatenate KV rows per
   layer, take recurrent state from the last chunk, insert the branch into the
   trie, and prefill only the remainder. An unreadable chunk is a miss and is
   removed.
4. RAM tier: make `maxPagedOutBytes` configurable with a device-aware default,
   since with the disk tier behind it eviction loses nothing.
5. Identity: model digest, cache layout (layer kinds, KV heads, head dims,
   dtypes) and a format version, so a changed model never reads stale state.
6. Logging: a `disk cache` line per request (chunks restored and written,
   bytes, milliseconds) for `scripts/local-model-lab.sh cache` to report.

## Sizing

For the 64-layer hybrid Qwen layout (16 attention layers at 4 KV heads x 256,
48 linear layers), one 4096-token chunk is about 256 MiB of KV plus about
147 MiB of float32 recurrent state, so 32 GiB holds roughly 80 chunks, or
330k tokens of distinct prefixes. Only new chunks are written: a turn that
extends a cached session writes just the chunks it adds.
