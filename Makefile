###############################################
#
# Makefile — zignanogpt
#
# `make validate` is the pre-commit gate; `make build` is the entry point.
#
###############################################

.DEFAULT_GOAL := all

.PHONY: all validate build dist run cli web test-cpu cuda train-bench demo linux windows targets test bench format lint fixtures docs tag st open github clone claude clean

# ---------------------------------------------
# Configuration
# ---------------------------------------------

# The released version, read from build.zig.zon (the source of truth).
VERSION := $(shell sed -n -E 's/^[[:space:]]*\.version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' build.zig.zon)

# The Zig sources to format and lint. Named explicitly rather than `.`: the
# nanochat/ submodule and .zig-cache/ are not ours to rewrite or gate on.
ZIG_SOURCES := build.zig $(wildcard src/*.zig) $(wildcard src/*/*.zig) $(wildcard cli/*.zig) $(wildcard web/*.zig) $(wildcard bench/*.zig) $(wildcard tools/*.zig)

# The styleguide checkout lint reads its zlint.json from. Override to lint
# against another checkout: `make lint STYLEGUIDE=../zigmicrogpt/styleguide`.
STYLEGUIDE ?= styleguide

# The Python reference's environment, kept out of the nanochat/ submodule so
# the submodule's working tree stays clean.
PY_ENV := $(CURDIR)/.venv

# uv, invoked by absolute path like the other inferise tools.
UV ?= /usr/local/inferise/uv/bin/uv

# On macOS the gate also cross-compiles the DGX OS (Linux ARM64) CUDA build and
# its tests, so both supported platforms stay compiling; on the Spark itself
# `make test` already runs the suite on CUDA.
VALIDATE_EXTRA := $(if $(filter Darwin,$(shell uname -s)),cuda,)

# ---------------------------------------------
# Primary workflows
# ---------------------------------------------

# Build, format, and test.
all: build format test
	@echo "done"

# Full pre-commit gate: clean, format, lint, build, then test.
validate: clean format lint build test test-cpu $(VALIDATE_EXTRA)
	@echo "validate done"

# ---------------------------------------------
# Build
# ---------------------------------------------

# Build the library and CLI.
build:
	zig build

# Build optimized for release (backend detected; override with -Dbackend=).
dist:
	zig build --release=fast

# Build and run the CLI. Pass arguments with `make run ARGS="Zig"`.
run:
	zig build run -- $(ARGS)

# Open the console (the TUI) on this terminal, from a ReleaseFast build: training
# in Debug is many times slower. Point it elsewhere with ZIGNANOGPT_BASE_DIR=<dir>.
cli:
	zig build --release=fast
	./zig-out/bin/zignanogpt tui

web:
	zig build --release=fast
	./zig-out/bin/zignanogpt-web $(ARGS)

# Build and run the web console on a random port; it opens a browser itself.
demo:
	zig build serve

# ---------------------------------------------
# Cross-compilation
# ---------------------------------------------

# Compile for Linux: x86_64, and aarch64 for the DGX Spark.
linux:
	zig build -Dtarget=x86_64-linux
	zig build -Dtarget=aarch64-linux

# Compile for Windows.
windows:
	zig build -Dtarget=x86_64-windows

# List every target Zig can build for.
targets:
	zig targets

# ---------------------------------------------
# Test
# ---------------------------------------------

# Run the unit test suite on the detected backend (Metal, CUDA or CPU).
test:
	zig build test --summary all

# The test suite on the CPU backend (the reference the GPU backends are checked against).
test-cpu:
	zig build test -Dbackend=cpu --summary all

# Cross-compile the CUDA build for DGX OS (Linux ARM64) from macOS: the
# binaries, plus every test binary compiled (not run: no GPU or Linux here).
cuda:
	zig build -Dbackend=cuda -Dtarget=aarch64-linux-gnu
	zig build test -Dbackend=cuda -Dtarget=aarch64-linux-gnu --summary all

# Time backend matmul throughput (always ReleaseFast).
bench:
	zig build bench

# Time 40 base-training steps of the d6 run (32 x 512, eval and samples off) on
# the detected backend; the same command on the Mac (Metal) and the Spark (CUDA).
# Needs the tokenizer and data shards in the base dir; the checkpoint is removed.
train-bench:
	zig build --release=fast
	./zig-out/bin/zignanogpt train --no-tui --depth 6 --head-dim 64 --max-seq-len 512 --window-pattern L \
		--device-batch-size 32 --total-batch-size 16384 --model-tag train-bench --num-iterations 40 \
		--eval-every 0 --sample-every 0 --save-every 1000 | grep -E "^step 000[123]"
	rm -rf "$${ZIGNANOGPT_BASE_DIR:-$$HOME/.cache/zignanogpt}/base_checkpoints/train-bench"

# ---------------------------------------------
# Format & lint
# ---------------------------------------------

# Format the Zig sources.
format:
	zig fmt $(ZIG_SOURCES)

# Token-level style, then the rule set in the styleguide. zlintpre exits 0
# whatever it finds, so its summary line is what gates. zlint gets its files on
# stdin: walking the tree on its own, it would also lint nanochat/ and caches.
lint:
	@test -f $(STYLEGUIDE)/zlint.json || { echo "lint: $(STYLEGUIDE)/zlint.json missing; add the styleguide or pass STYLEGUIDE=<path>"; exit 1; }
	@output=$$(zlintpre $(ZIG_SOURCES) 2>&1); echo "$$output"; \
	echo "$$output" | grep -q 'found 0 failures' || { echo "lint: zlintpre findings above"; exit 1; }
	printf '%s\n' $(ZIG_SOURCES) | zlint -c $(STYLEGUIDE) --deny-warnings --stdin

# ---------------------------------------------
# Parity fixtures
# ---------------------------------------------

# Regenerate testdata/ from the Python reference (dev only; needs uv). `--frozen`
# keeps uv from rewriting the submodule's uv.lock. One group: `make fixtures ONLY=gpt`.
fixtures:
	cd nanochat && PYTHONPATH=$(CURDIR)/nanochat UV_PROJECT_ENVIRONMENT=$(PY_ENV) $(UV) run --frozen --extra cpu python $(CURDIR)/dev/fixtures.py --out $(CURDIR)/testdata $(if $(ONLY),--only $(ONLY))

# ---------------------------------------------
# Documentation
# ---------------------------------------------

# Build the API docs and serve them locally.
docs:
	zig build docs
	open "http://127.0.0.1:8080" &
	python3 -m http.server -b 127.0.0.1 8080 -d zig-out/docs

# ---------------------------------------------
# Release
# ---------------------------------------------

# Tag the build.zig.zon version (e.g. 1.0.0) and push it.
tag:
	@test -n "$(VERSION)" || { echo "❌ could not read .version from build.zig.zon"; exit 1; }
	git tag -a "$(VERSION)" -m "$(VERSION)"
	git push
	git push --tags

# ---------------------------------------------
# Environment
# ---------------------------------------------

# Open the working copy in SourceTree.
st:
	open -a SourceTree .

# Open the project in the editor.
open:
	code .

# Open the repository on GitHub.
github:
	open "https://github.com/inferise/zignanogpt"

# Clone a fresh working copy.
clone:
	git clone git@github.com:inferise/zignanogpt.git

# Start Claude Code here.
claude:
	claude

# ---------------------------------------------
# Housekeeping
# ---------------------------------------------

# Remove every build artifact.
clean:
	rm -rf .zig-cache
	rm -rf zig-out
