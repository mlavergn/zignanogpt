const std = @import("std");
const builtin = @import("builtin");

const zon = @import("build.zig.zon");

const Git = struct {
    const Self = @This();

    /// Clones every missing cloneable dependency and exits 1; returns when none are missing.
    ///
    /// Parameters:
    /// - `b`: the build graph.
    ///
    /// Return: nothing when all are present; otherwise never.
    pub fn cloneDeps(b: *std.Build) if (Self.hasAllDeps()) void else noreturn {
        if (comptime Self.hasAllDeps()) return;
        const io = b.graph.io;
        var missing: usize = 0;
        var cloned: usize = 0;
        inline for (@typeInfo(@TypeOf(zon.dependencies)).@"struct".fields) |field| {
            if (comptime Self.isCloneable(field.name) and !Self.isCloned(field.name)) {
                const dep = @field(zon.dependencies, field.name);
                const dest = b.pathFromRoot(dep.path);
                missing += 1;
                if (std.Io.Dir.cwd().access(io, dest, .{})) |_| {
                    std.debug.print("{s} exists but is not a Zig package\n", .{dep.path});
                } else |err| switch (err) {
                    error.FileNotFound => if (Self.clone(io, dep.clone, dest)) {
                        cloned += 1;
                    } else |clone_err| {
                        std.debug.print("git clone {s} {s} failed [{any}]\n", .{ dep.clone, dep.path, clone_err });
                    },
                    else => std.debug.print("cannot check {s} [{any}]\n", .{ dep.path, err }),
                }
            }
        }
        if (cloned > 0) std.debug.print("cloned {d} of {d} missing dependencies; re-run zig build\n", .{ cloned, missing });
        std.process.exit(1);
    }

    /// Reports whether every cloneable dependency is present.
    ///
    /// Return: `true` when none is missing.
    fn hasAllDeps() bool {
        inline for (@typeInfo(@TypeOf(zon.dependencies)).@"struct".fields) |field| {
            if (Self.isCloneable(field.name) and !Self.isCloned(field.name)) return false;
        }
        return true;
    }

    /// Reports whether dependency `name` was present when the build runner was compiled.
    ///
    /// Parameters:
    /// - `name`: the dependency's field name in `build.zig.zon`.
    ///
    /// Return: `true` when present, or when this package is not the root.
    fn isCloned(comptime name: []const u8) bool {
        const deps = @import("root").dependencies;
        for (deps.root_deps) |dep| {
            if (std.mem.eql(u8, dep[0], name)) return @hasDecl(@field(deps.packages, dep[1]), "build_zig");
        }
        return true;
    }

    /// Reports whether dependency `name` declares both a `.path` and a `.clone`.
    ///
    /// Parameters:
    /// - `name`: the dependency's field name in `build.zig.zon`.
    ///
    /// Return: `true` when both fields are present.
    fn isCloneable(comptime name: []const u8) bool {
        const Dep = @TypeOf(@field(zon.dependencies, name));
        return @hasField(Dep, "path") and @hasField(Dep, "clone");
    }

    /// Clones the repository at `url` into `dest`.
    ///
    /// Parameters:
    /// - `io`: IO the `git` child is spawned on.
    /// - `url`: the repository to clone.
    /// - `dest`: the directory to clone into.
    ///
    /// Return: nothing on success; `error.GitCloneFailed` when `git` fails.
    fn clone(io: std.Io, url: []const u8, dest: []const u8) !void {
        var child = try std.process.spawn(io, .{ .argv = &.{ "git", "clone", url, dest } });
        switch (try child.wait(io)) {
            .exited => |code| if (code != 0) return error.GitCloneFailed,
            else => return error.GitCloneFailed,
        }
    }
};

const Xcode = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    io: std.Io,
    target: std.Target,
    sdk: []const u8,

    /// Creates an unresolved `Xcode` handle.
    ///
    /// Parameters:
    /// - `allocator`: allocator for the SDK lookup.
    /// - `io`: IO for the SDK lookup.
    ///
    /// Return: the handle; `target` and `sdk` are unset until `resolve`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Self {
        return Self{
            .allocator = allocator,
            .io = io,
            // SAFETY: populated by resolve() before either field is read.
            .target = undefined,
            // SAFETY: populated by resolve() before either field is read.
            .sdk = undefined,
        };
    }

    /// Resolves the Apple Silicon macOS target and SDK path into the handle.
    ///
    /// Parameters:
    /// - `self`: the handle to populate.
    ///
    /// Return: nothing on success; `error.FailedToResolveSDK` when no SDK is found.
    pub fn resolve(self: *Self) !void {
        const query = std.Target.Query{
            .cpu_arch = .aarch64,
            .os_tag = .macos,
        };
        self.target = try std.zig.system.resolveTargetQuery(self.io, query);
        self.sdk = std.zig.system.darwin.getSdk(
            self.allocator,
            self.io,
            &self.target,
        ) orelse return error.FailedToResolveSDK;
    }
};

/// The compute backend the library is built for: its choice (`-Dbackend=`, or
/// detected from the target and the host), what it links and the kernels it
/// embeds. One per build: model code is generic over it, so the choice is a
/// comptime constant.
const Backend = struct {
    pub const Kind = enum { cpu, metal, cuda };

    /// Where Linux installs the NVIDIA driver library (`CudaBackend` loads it at run time).
    const cuda_driver_paths = [_][]const u8{
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
    };

    /// The backend for this build: `-Dbackend` when given, else `detect`.
    ///
    /// Parameters:
    /// - `b`: the build.
    /// - `target`: the build target.
    ///
    /// Return: the backend.
    pub fn choose(b: *std.Build, target: std.Build.ResolvedTarget) Kind {
        return b.option(Kind, "backend", "Compute backend: cpu, metal, cuda (default: detected)") orelse detect(b, target);
    }

    /// Metal on Apple-silicon macOS, CUDA on Linux with the NVIDIA driver
    /// installed, otherwise the CPU. Detection looks at this machine, so a cross
    /// build (another OS or CPU) gets the CPU unless `-Dbackend` says otherwise.
    ///
    /// Parameters:
    /// - `b`: the build.
    /// - `target`: the build target.
    ///
    /// Return: the detected backend.
    pub fn detect(b: *std.Build, target: std.Build.ResolvedTarget) Kind {
        const t = target.result;
        const host = b.graph.host.result;
        if (t.os.tag != host.os.tag or t.cpu.arch != host.cpu.arch) return .cpu;
        switch (t.os.tag) {
            .macos => if (t.cpu.arch == .aarch64 and exists(b, "/System/Library/Frameworks/Metal.framework")) return .metal,
            .linux => for (cuda_driver_paths) |path| {
                if (exists(b, path)) return .cuda;
            },
            else => {},
        }
        return .cpu;
    }

    /// The kernels the backend embeds: the CUDA PTX, built from Zig sources.
    ///
    /// Parameters:
    /// - `b`: the build.
    /// - `kind`: the backend.
    ///
    /// Return: the generated file, or null for backends that compile at run time (Metal) or have none.
    pub fn kernels(b: *std.Build, kind: Kind) ?std.Build.LazyPath {
        return if (kind == .cuda) cudaKernels(b) else null;
    }

    /// Wires the backend into a library module: Metal's frameworks, or the CUDA PTX.
    ///
    /// Parameters:
    /// - `module`: a module rooted in `src/`.
    /// - `kind`: the backend.
    /// - `ptx`: `kernels(b, kind)`.
    ///
    /// Return: nothing.
    pub fn link(module: *std.Build.Module, kind: Kind, ptx: ?std.Build.LazyPath) void {
        switch (kind) {
            .cpu => {},
            // Metal is reached through the Objective-C runtime; its MSL sources are compiled at start-up.
            .metal => {
                module.linkFramework("Metal", .{});
                module.linkFramework("MetalPerformanceShaders", .{});
                module.linkFramework("Foundation", .{});
                module.linkSystemLibrary("objc", .{});
            },
            .cuda => module.addAnonymousImport("cuda_kernels.ptx", .{ .root_source_file = ptx.? }),
        }
    }

    fn exists(b: *std.Build, path: []const u8) bool {
        std.Io.Dir.cwd().access(b.graph.io, path, .{}) catch return false;
        return true;
    }

    /// Builds the CUDA kernels' PTX: `src/cuda/kernels.zig` to LLVM IR for
    /// nvptx64-cuda (sm_80, JIT-compiled forward by the driver), the kernel
    /// aliases Zig emits rewritten (`tools/nvptx_fixup.zig`), then `zig cc` to PTX.
    ///
    /// Parameters:
    /// - `b`: the build.
    ///
    /// Return: the generated `.ptx` file.
    fn cudaKernels(b: *std.Build) std.Build.LazyPath {
        const nvptx = b.resolveTargetQuery(.{
            .cpu_arch = .nvptx64,
            .os_tag = .cuda,
            .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_80 },
        });
        const object = b.addObject(.{
            .name = "cuda_kernels",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/cuda/kernels.zig"),
                .target = nvptx,
                .optimize = .ReleaseFast,
            }),
        });
        const fixup = b.addExecutable(.{
            .name = "nvptx_fixup",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/nvptx_fixup.zig"),
                .target = b.graph.host,
            }),
        });
        const run_fixup = b.addRunArtifact(fixup);
        run_fixup.addFileArg(object.getEmittedLlvmIr());
        const fixed = run_fixup.addOutputFileArg("cuda_kernels.ll");
        const to_ptx = b.addSystemCommand(&.{ b.graph.zig_exe, "cc", "-target", "nvptx64-cuda", "-march=sm_80", "-S", "-Wno-unused-command-line-argument" });
        to_ptx.addFileArg(fixed);
        to_ptx.addArg("-o");
        return to_ptx.addOutputFileArg("cuda_kernels.ptx");
    }
};

const Config = struct {
    name: []const u8,
    mod_name: []const u8,
    web_name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    module_source_file: std.Build.LazyPath,
    cli_source_file: std.Build.LazyPath,
    version: std.SemanticVersion,
};

/// The dependency modules, resolved once and shared by every root that needs them.
const Deps = struct {
    storage: *std.Build.Module,
    uucode: *std.Build.Module,
    /// The console's TUI components (and the vaxis API they re-export).
    tui: *std.Build.Module,
    /// The backend and its embedded kernels (`Backend.kernels`).
    backend: Backend.Kind = .cpu,
    kernels: ?std.Build.LazyPath = null,
};

/// Wires the options and library-side dependencies into a module rooted in `src/`.
///
/// Parameters:
/// - `module`: the module to wire.
/// - `options`: the `config` build options.
/// - `deps`: the resolved dependency modules.
///
/// Return: nothing.
fn addLibImports(module: *std.Build.Module, options: *std.Build.Step.Options, deps: Deps) void {
    module.addOptions("config", options);
    module.addImport("zigstorage", deps.storage);
    module.addImport("uucode", deps.uucode);
    Backend.link(module, deps.backend, deps.kernels);
}

/// Wires an executable-side module (CLI or web) onto the library.
///
/// Parameters:
/// - `module`: the module to wire.
/// - `options`: the `config` build options.
/// - `lib_name`: the name the library is imported under.
/// - `lib_module`: the library module.
///
/// Return: nothing.
fn addAppImports(module: *std.Build.Module, options: *std.Build.Step.Options, lib_name: []const u8, lib_module: *std.Build.Module) void {
    module.addOptions("config", options);
    module.addImport(lib_name, lib_module);
}

/// Declares the build graph for the zignanogpt library, CLI, web console, tests, and docs.
///
/// Serves as the entry point the Zig build runner invokes to wire up every
/// build step consumers rely on — `lib`, `cli`, `run`, `web`, `serve`, `test`,
/// and `docs` — plus the standard target and optimize options, `-Dbackend`, and
/// `-Dtest-filter`. On macOS it adds the resolved SDK framework path so linking
/// succeeds. Called once per build invocation.
///
/// Parameters:
/// - `b`: the build graph the steps and options are registered on.
///
/// Return: nothing on success; propagates errors from option parsing, version
/// parsing, and macOS SDK resolution.
pub fn build(b: *std.Build) !void {
    // Pre-flight: ensure any local dependencies are cloned
    Git.cloneDeps(b);

    // Build config
    const cfg = Config{
        .name = @tagName(zon.name),
        .mod_name = @tagName(zon.name),
        .web_name = @tagName(zon.name) ++ "-web",
        .target = b.standardTargetOptions(.{}),
        .optimize = b.standardOptimizeOption(.{}),
        .module_source_file = b.path("src/module.zig"),
        .cli_source_file = b.path("cli/main.zig"),
        .version = try std.SemanticVersion.parse(zon.version),
    };

    // Build options
    const options = b.addOptions();

    const backend = Backend.choose(b, cfg.target);
    options.addOption(Backend.Kind, "backend", backend);

    const test_filter = b.option([]const u8, "test-filter", "Run unit tests that match filter") orelse "";
    options.addOption([]const u8, "test_filter", test_filter);
    const test_filters: []const []const u8 = if (test_filter.len > 0) &.{test_filter} else &.{};

    options.addOption([]const u8, "version", zon.version);
    // Absolute build root, so tests find committed fixtures under testdata/
    // whatever directory the test runner starts in.
    options.addOption([]const u8, "source_root", b.build_root.path orelse ".");

    // -------------------------------------------------------------------------
    // Dependencies

    const storage_dep = b.dependency("zigstorage", .{
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    // zigtui: the console's components, and the whole vaxis API re-exported
    // (the CLI never imports zigvaxis itself). It also owns the process's
    // uucode module (zigvaxis's tables, `general_category` included): Zig puts
    // a package's files in one module only, so the tokenizer uses that one.
    const tui_dep = b.dependency("zigtui", .{
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    const deps = Deps{
        .storage = storage_dep.module("zigstorage"),
        .uucode = tui_dep.module("uucode"),
        .tui = tui_dep.module("zigtui"),
        .backend = backend,
        .kernels = Backend.kernels(b, backend),
    };

    // -------------------------------------------------------------------------
    // Module

    const module = b.addModule(cfg.name, .{
        .root_source_file = cfg.module_source_file,
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    addLibImports(module, options, deps);

    // -------------------------------------------------------------------------
    // Lib

    const lib = b.addLibrary(.{ .name = cfg.name, .root_module = module, .linkage = .static });

    if (builtin.os.tag == .macos) {
        var xcode = try Xcode.init(b.allocator, b.graph.io);
        try xcode.resolve();
        lib.root_module.addFrameworkPath(.{ .cwd_relative = xcode.sdk });
    }

    const lib_install = b.addInstallArtifact(lib, .{});
    const lib_step = b.step("lib", "Build static library");
    lib_step.dependOn(&lib_install.step);

    // -------------------------------------------------------------------------
    // CLI

    const cli_module = b.createModule(.{
        .root_source_file = cfg.cli_source_file,
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    addAppImports(cli_module, options, cfg.mod_name, module);
    cli_module.addImport("zigtui", deps.tui);

    const cli = b.addExecutable(.{
        .name = cfg.name,
        .root_module = cli_module,
    });
    b.installArtifact(cli);

    const cli_install = b.addInstallArtifact(cli, .{});
    const cli_step = b.step("cli", "Build the CLI app");
    cli_step.dependOn(&cli_install.step);

    // -------------------------------------------------------------------------
    // Web console

    const web_module = b.createModule(.{
        .root_source_file = b.path("web/main.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    addAppImports(web_module, options, cfg.mod_name, module);

    const web = b.addExecutable(.{
        .name = cfg.web_name,
        .root_module = web_module,
    });
    b.installArtifact(web);

    const web_install = b.addInstallArtifact(web, .{});
    const web_step = b.step("web", "Build the web console");
    web_step.dependOn(&web_install.step);

    const web_run = b.addRunArtifact(web);
    web_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        web_run.addArgs(args);
    }

    const web_run_step = b.step("serve", "Run the web console");
    web_run_step.dependOn(&web_run.step);

    // -------------------------------------------------------------------------
    // Run

    const cli_run = b.addRunArtifact(cli);

    // Run step depends on the install step to run from the installation directory
    cli_run.step.dependOn(b.getInstallStep());

    // Support arguments like: `zig build run -- argA argB`
    if (b.args) |args| {
        cli_run.addArgs(args);
    }

    const run_step = b.step("run", "Run the CLI app");
    run_step.dependOn(&cli_run.step);

    // -------------------------------------------------------------------------
    // Bench: always ReleaseFast, on its own optimized copy of the library.

    const bench_lib = b.createModule(.{
        .root_source_file = cfg.module_source_file,
        .target = cfg.target,
        .optimize = .ReleaseFast,
    });
    addLibImports(bench_lib, options, deps);
    const bench_module = b.createModule(.{
        .root_source_file = b.path("bench/main.zig"),
        .target = cfg.target,
        .optimize = .ReleaseFast,
    });
    addAppImports(bench_module, options, cfg.mod_name, bench_lib);
    const bench = b.addExecutable(.{
        .name = b.fmt("{s}-bench", .{cfg.name}),
        .root_module = bench_module,
    });
    const bench_run = b.addRunArtifact(bench);
    bench_run.has_side_effects = true; // timings are never cached
    const bench_step = b.step("bench", "Time backend matmul throughput (ReleaseFast)");
    bench_step.dependOn(&bench_run.step);

    // -------------------------------------------------------------------------
    // Tests

    const tests_step = b.step("test", "Run unit tests");

    const tests = b.addTest(.{
        .root_module = module,
        .filters = test_filters,
    });

    const tests_run = b.addRunArtifact(tests);
    // force tests to run on every test run
    tests_run.has_side_effects = true;
    tests_step.dependOn(&tests_run.step);

    // The CLI is its own module, so its tests need their own compile step.
    const cli_tests_module = b.createModule(.{
        .root_source_file = b.path("cli/module.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    addAppImports(cli_tests_module, options, cfg.mod_name, module);
    cli_tests_module.addImport("zigtui", deps.tui);

    const cli_tests = b.addTest(.{
        .root_module = cli_tests_module,
        .filters = test_filters,
    });

    const cli_tests_run = b.addRunArtifact(cli_tests);
    cli_tests_run.has_side_effects = true;
    tests_step.dependOn(&cli_tests_run.step);

    // The web console is likewise its own module.
    const web_tests_module = b.createModule(.{
        .root_source_file = b.path("web/module.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    addAppImports(web_tests_module, options, cfg.mod_name, module);

    const web_tests = b.addTest(.{
        .root_module = web_tests_module,
        .filters = test_filters,
    });

    const web_tests_run = b.addRunArtifact(web_tests);
    web_tests_run.has_side_effects = true;
    tests_step.dependOn(&web_tests_run.step);

    // Build tools (host): the CUDA kernels' IR fixup.
    const tools_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/nvptx_fixup.zig"),
            .target = b.graph.host,
            .optimize = cfg.optimize,
        }),
        .filters = test_filters,
    });
    const tools_tests_run = b.addRunArtifact(tools_tests);
    tools_tests_run.has_side_effects = true;
    tests_step.dependOn(&tools_tests_run.step);

    // -------------------------------------------------------------------------
    // Docs

    // Autodoc roots a module at `root.zig`, or failing that at the file whose
    // basename matches the artifact name; with `src/module.zig` it can do
    // neither, and the site roots itself at an arbitrary import instead.
    // ZIGSTYLE requires the barrel to be `module.zig`, so docs build from a
    // sibling `root.zig`. It is the same module — the other sources come along.
    const docs_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    addLibImports(docs_mod, options, deps);

    const docs_lib = b.addLibrary(.{
        .name = cfg.name,
        .root_module = docs_mod,
        .linkage = .static,
    });

    if (builtin.os.tag == .macos) {
        var docs_xcode = try Xcode.init(b.allocator, b.graph.io);
        try docs_xcode.resolve();
        docs_lib.root_module.addFrameworkPath(.{ .cwd_relative = docs_xcode.sdk });
    }

    const docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);
}
