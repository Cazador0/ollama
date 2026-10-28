package diskcache

// lockFile is a no-op on Windows: eviction is serialized within the process
// only, and stores shared across processes may briefly exceed the budget.
func lockFile(string) (func(), error) {
	return func() {}, nil
}
