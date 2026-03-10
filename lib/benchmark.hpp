/*
  Meadows CPP
  Robbie Alexander
  benchmark.hpp
  Comprehensive benchmarking infrastructure for academic-quality
  performance characterisation of the geometric multigrid solver.

  Provides:
    - HighResTimer: nanosecond-precision wall-clock timing
    - PhaseTimer: hierarchical per-phase accumulator (smooth, restrict, etc.)
    - MemorySnapshot: captures allocation state from track_allocations
    - ConvergenceRecord: per-iteration residual + timing data
    - BenchmarkResult: full suite output with statistics
    - CSVWriter: publication-ready tabular output
    - BenchmarkSuite: orchestrates grid-scaling, thread-scaling, and
      per-phase profiling runs
*/
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "memory.hpp"
#include "grid.hpp"
#include "boundary_conditions.hpp"
#include "operators.hpp"
#include "interlevel.hpp"

// ============================================================================
// HighResTimer: thin wrapper around steady_clock for ns-precision timing
// ============================================================================

struct HighResTimer {
  using clock = std::chrono::steady_clock;
  using time_point = clock::time_point;

  time_point _start;

  void start() { _start = clock::now(); }

  // Returns elapsed time in seconds since start()
  double elapsed_s() const {
    auto now = clock::now();
    return std::chrono::duration<double>(now - _start).count();
  }

  // Returns elapsed time in milliseconds since start()
  double elapsed_ms() const { return elapsed_s() * 1e3; }

  // Returns elapsed time in microseconds since start()
  double elapsed_us() const { return elapsed_s() * 1e6; }

  // Returns elapsed time in nanoseconds since start()
  double elapsed_ns() const { return elapsed_s() * 1e9; }
};

// ============================================================================
// ScopedTimer: RAII timer that accumulates into a target double (seconds)
// ============================================================================

struct ScopedTimer {
  HighResTimer _timer;
  double& _target;

  explicit ScopedTimer(double& target) : _target(target) { _timer.start(); }
  ~ScopedTimer() { _target += _timer.elapsed_s(); }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
};

// ============================================================================
// PhaseTimer: accumulates time across multiple calls per named phase
// ============================================================================

class PhaseTimer {
public:
  void reset() {
    _phases.clear();
    _call_counts.clear();
  }

  // Start timing a phase (returns index for stop)
  void start(const std::string& phase) {
    _active_phase = phase;
    _timer.start();
  }

  void stop() {
    double dt = _timer.elapsed_s();
    _phases[_active_phase] += dt;
    _call_counts[_active_phase]++;
  }

  double total_time(const std::string& phase) const {
    auto it = _phases.find(phase);
    return (it != _phases.end()) ? it->second : 0.0;
  }

  int call_count(const std::string& phase) const {
    auto it = _call_counts.find(phase);
    return (it != _call_counts.end()) ? it->second : 0;
  }

  const std::map<std::string, double>& all_phases() const { return _phases; }

  double total_all_phases() const {
    double sum = 0.0;
    for (const auto& [name, t] : _phases) sum += t;
    return sum;
  }

  void print_summary(const char* header = "Phase Breakdown") const {
    double total = total_all_phases();
    std::printf("\n=== %s ===\n", header);
    std::printf("%-25s %12s %8s %10s\n", "Phase", "Time (ms)", "Calls", "% Total");
    std::printf("%-25s %12s %8s %10s\n", "-------------------------",
                "------------", "--------", "----------");
    for (const auto& [name, t] : _phases) {
      auto it = _call_counts.find(name);
      int calls = (it != _call_counts.end()) ? it->second : 0;
      double pct = (total > 0.0) ? 100.0 * t / total : 0.0;
      std::printf("%-25s %12.3f %8d %9.1f%%\n",
                  name.c_str(), t * 1e3, calls, pct);
    }
    std::printf("%-25s %12.3f %8s %9.1f%%\n", "TOTAL", total * 1e3, "", 100.0);
  }

private:
  std::map<std::string, double> _phases;
  std::map<std::string, int> _call_counts;
  HighResTimer _timer;
  std::string _active_phase;
};

// ============================================================================
// MemorySnapshot: captures allocation stats at a point in time
// ============================================================================

struct MemorySnapshot {
  size_t active_allocations = 0;
  size_t active_bytes = 0;
  size_t cumulative_events = 0;
  size_t cumulative_bytes = 0;

  static MemorySnapshot capture() {
    MemorySnapshot snap;
    snap.active_allocations = track_allocations::get_active_count();
    snap.active_bytes = track_allocations::get_total_bytes();
    snap.cumulative_events = track_allocations::event_count().load(std::memory_order_relaxed);
    snap.cumulative_bytes = track_allocations::total_bytes_ever().load(std::memory_order_relaxed);
    return snap;
  }

  static void reset_counters() {
    track_allocations::reset_event_counters();
  }

  void print(const char* label = "Memory") const {
    std::printf("\n=== %s ===\n", label);
    std::printf("  Active allocations:  %zu\n", active_allocations);
    std::printf("  Active bytes:        %zu (%.2f MB)\n",
                active_bytes, active_bytes / (1024.0 * 1024.0));
    std::printf("  Cumulative allocs:   %zu\n", cumulative_events);
    std::printf("  Cumulative bytes:    %zu (%.2f MB)\n",
                cumulative_bytes, cumulative_bytes / (1024.0 * 1024.0));
  }
};

// ============================================================================
// ConvergenceRecord: one entry per V-cycle iteration
// ============================================================================

struct ConvergenceRecord {
  int cycle;
  double residual_norm;
  double wall_time_s;         // cumulative wall time since solve start
  double cycle_time_s;        // wall time for this single V-cycle
};

// ============================================================================
// Statistics helper
// ============================================================================

struct Stats {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
  double median = 0.0;
  int n = 0;

  static Stats compute(const std::vector<double>& vals) {
    Stats s;
    s.n = static_cast<int>(vals.size());
    if (s.n == 0) return s;

    s.mean = std::accumulate(vals.begin(), vals.end(), 0.0) / s.n;
    s.min = *std::min_element(vals.begin(), vals.end());
    s.max = *std::max_element(vals.begin(), vals.end());

    double var = 0.0;
    for (double v : vals) var += (v - s.mean) * (v - s.mean);
    s.stddev = (s.n > 1) ? std::sqrt(var / (s.n - 1)) : 0.0;

    auto sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    if (s.n % 2 == 0)
      s.median = (sorted[s.n / 2 - 1] + sorted[s.n / 2]) / 2.0;
    else
      s.median = sorted[s.n / 2];

    return s;
  }

  void print(const char* label, const char* unit = "ms") const {
    double scale = 1.0;
    if (std::string(unit) == "ms") scale = 1e3;
    else if (std::string(unit) == "us") scale = 1e6;

    std::printf("  %-20s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f  median=%.4f %s  (n=%d)\n",
                label, mean * scale, stddev * scale, min * scale,
                max * scale, median * scale, unit, n);
  }
};

// ============================================================================
// CSVWriter: writes benchmark data in publication-ready CSV format
// ============================================================================

class CSVWriter {
public:
  explicit CSVWriter(const std::string& filepath) : _path(filepath) {}

  void write_convergence(const std::vector<ConvergenceRecord>& records,
                         int N, int n_levels, int n_threads) {
    std::ofstream f(_path);
    f << "# Convergence history: N=" << N << " levels=" << n_levels
      << " threads=" << n_threads << "\n";
    f << "cycle,residual_norm,wall_time_s,cycle_time_s,convergence_factor\n";
    for (size_t i = 0; i < records.size(); ++i) {
      double factor = (i > 0 && records[i - 1].residual_norm > 0.0)
                          ? records[i].residual_norm / records[i - 1].residual_norm
                          : 0.0;
      f << records[i].cycle << ","
        << records[i].residual_norm << ","
        << records[i].wall_time_s << ","
        << records[i].cycle_time_s << ","
        << factor << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  void write_phase_breakdown(const PhaseTimer& timer, int N, int n_levels, int n_threads) {
    std::ofstream f(_path);
    f << "# Phase breakdown: N=" << N << " levels=" << n_levels
      << " threads=" << n_threads << "\n";
    f << "phase,total_time_s,calls,time_per_call_s,percent\n";
    double total = timer.total_all_phases();
    for (const auto& [name, t] : timer.all_phases()) {
      int calls = timer.call_count(name);
      double per_call = (calls > 0) ? t / calls : 0.0;
      double pct = (total > 0.0) ? 100.0 * t / total : 0.0;
      f << name << "," << t << "," << calls << "," << per_call << "," << pct << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  void write_scaling(const std::vector<std::tuple<int, double, double, double>>& data,
                     const std::string& x_label) {
    std::ofstream f(_path);
    f << "# Scaling study\n";
    f << x_label << ",solve_time_s,dof_per_s,efficiency\n";
    for (const auto& [x, time, dof_s, eff] : data) {
      f << x << "," << time << "," << dof_s << "," << eff << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  void write_memory(const std::vector<std::tuple<int, size_t, size_t, size_t>>& data) {
    std::ofstream f(_path);
    f << "# Memory profile\n";
    f << "N,peak_bytes,active_allocations,cumulative_allocations\n";
    for (const auto& [N, bytes, active, cumulative] : data) {
      f << N << "," << bytes << "," << active << "," << cumulative << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  void write_throughput(
      const std::vector<std::tuple<int, double, double, double, int>>& data) {
    std::ofstream f(_path);
    f << "# Throughput analysis\n";
    f << "N,total_dof,solve_time_s,dof_per_s,vcycles_to_converge\n";
    for (const auto& [N, dof, time, dof_s, vcycles] : data) {
      f << N << "," << dof << "," << time << "," << dof_s << "," << vcycles << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  // Thread scaling with explicit speedup and ideal columns
  void write_thread_scaling(
      const std::vector<std::tuple<int, double, double, double, double>>& data,
      int N) {
    std::ofstream f(_path);
    f << "# Thread scaling (strong): N=" << N << "\n";
    f << "threads,solve_time_s,speedup,ideal_speedup,efficiency,dof_per_s\n";
    for (const auto& [nt, time, speedup, eff, dof_s] : data) {
      f << nt << "," << time << "," << speedup << ","
        << static_cast<double>(nt) << "," << eff << "," << dof_s << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  // Weak scaling: fixed DOF/thread, grow problem with threads
  void write_weak_scaling(
      const std::vector<std::tuple<int, int, double, double, double>>& data) {
    std::ofstream f(_path);
    f << "# Weak scaling: fixed DOF/thread\n";
    f << "threads,N,total_dof,solve_time_s,efficiency\n";
    for (const auto& [nt, N, dof, time, eff] : data) {
      f << nt << "," << N << "," << dof << "," << time << "," << eff << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

  // Solution space / complexity verification: time per DOF should be O(1)
  void write_solution_space(
      const std::vector<std::tuple<int, double, double, double, double, int, double>>& data,
      int n_threads) {
    std::ofstream f(_path);
    f << "# Solution space complexity: threads=" << n_threads << "\n";
    f << "N,total_dof,solve_time_s,time_per_dof_us,dof_per_s,vcycles,asymptotic_convergence_rate\n";
    for (const auto& [N, dof, time, time_per_dof, dof_s, vcycles, conv_rate] : data) {
      f << N << "," << dof << "," << time << "," << time_per_dof << ","
        << dof_s << "," << vcycles << "," << conv_rate << "\n";
    }
    f.close();
    std::printf("  Written: %s\n", _path.c_str());
  }

private:
  std::string _path;
};

// ============================================================================
// InstrumentedSolver: wraps EllipticSolverGMG with per-phase timing
// ============================================================================

template<typename T, int Dims, typename Alloc = HostAllocator<T, track_allocations>>
class InstrumentedSolver {
private:
  GridHierarchy<T, Dims, Alloc>& _hierarchy;
  const BCRegistry<T, Dims>& _bc_registry;
  BCRegistry<T, Dims> _homogeneous_bc;
  RBGS<T, Dims> _smoother;

  int _phi, _rhs, _res;
  int _n_pre_smooth, _n_post_smooth, _n_coarse_smooth;
  int _max_vcycles;
  T _tol;

public:
  PhaseTimer phase_timer;
  std::vector<ConvergenceRecord> convergence_history;

  InstrumentedSolver(GridHierarchy<T, Dims, Alloc>& hierarchy,
                     const BCRegistry<T, Dims>& bc_registry,
                     int phi_comp, int rhs_comp, int res_comp,
                     int n_pre = 2, int n_post = 2, int n_coarse = 20,
                     int max_vcycles = 50, T tol = T{1e-10})
      : _hierarchy(hierarchy), _bc_registry(bc_registry),
        _homogeneous_bc(bc_registry.make_homogeneous()),
        _phi(phi_comp), _rhs(rhs_comp), _res(res_comp),
        _n_pre_smooth(n_pre), _n_post_smooth(n_post),
        _n_coarse_smooth(n_coarse), _max_vcycles(max_vcycles), _tol(tol) {}

  void V_cycle(int level, bool is_correction = false) {
    auto& lvl = _hierarchy.level(level);
    const auto& geom = lvl.geometry();
    auto phi_acc = lvl.accessor(_phi);
    auto rhs_acc = lvl.accessor(_rhs);

    const auto& bc = is_correction ? _homogeneous_bc : _bc_registry;
    auto fill_bc = [&](FieldAccessor<T, Dims>& acc) {
      bc.fill_all(acc, geom);
    };

    if (level == 0) {
      {
        phase_timer.start("boundary_fill");
        fill_bc(phi_acc);
        phase_timer.stop();
      }
      {
        phase_timer.start("coarse_smooth");
        _smoother.smooth(phi_acc, rhs_acc, geom, _n_coarse_smooth, fill_bc);
        phase_timer.stop();
      }
      return;
    }

    // pre-smooth
    {
      phase_timer.start("boundary_fill");
      fill_bc(phi_acc);
      phase_timer.stop();
    }
    {
      phase_timer.start("pre_smooth");
      _smoother.smooth(phi_acc, rhs_acc, geom, _n_pre_smooth, fill_bc);
      phase_timer.stop();
    }

    // residual
    {
      phase_timer.start("boundary_fill");
      fill_bc(phi_acc);
      phase_timer.stop();
    }
    auto res_acc = lvl.accessor(_res);
    {
      phase_timer.start("residual");
      residual<StandardFlux, T, Dims>::compute(phi_acc, rhs_acc, res_acc, geom);
      phase_timer.stop();
    }

    // restriction
    auto& coarse_lvl = _hierarchy.level(level - 1);
    const auto& coarse_geom = coarse_lvl.geometry();
    auto coarse_rhs = coarse_lvl.accessor(_rhs);
    {
      phase_timer.start("restriction");
      Restriction<T, Dims>::apply(res_acc, coarse_rhs, coarse_geom,
                                  _hierarchy.ref_ratio());
      phase_timer.stop();
    }

    // zero correction
    auto coarse_phi = coarse_lvl.accessor(_phi);
    {
      phase_timer.start("zero_field");
      zero_field(coarse_phi, coarse_geom);
      phase_timer.stop();
    }

    // recurse
    V_cycle(level - 1, true);

    // prolongation
    {
      phase_timer.start("prolongation");
      Prolongation<T, Dims>::apply(coarse_phi, phi_acc, geom,
                                   _hierarchy.ref_ratio());
      phase_timer.stop();
    }

    // post-smooth
    {
      phase_timer.start("boundary_fill");
      fill_bc(phi_acc);
      phase_timer.stop();
    }
    {
      phase_timer.start("post_smooth");
      _smoother.smooth(phi_acc, rhs_acc, geom, _n_post_smooth, fill_bc);
      phase_timer.stop();
    }
  }

  void full_multigrid_cycle() {
    int finest = _hierarchy.finest_level();

    phase_timer.start("fmg_restrict_rhs");
    for (int lvl = finest; lvl >= 1; --lvl) {
      auto fine_rhs = _hierarchy.level(lvl).accessor(_rhs);
      auto coarse_rhs = _hierarchy.level(lvl - 1).accessor(_rhs);
      const auto& coarse_geom = _hierarchy.level(lvl - 1).geometry();
      Restriction<T, Dims>::apply(fine_rhs, coarse_rhs, coarse_geom,
                                  _hierarchy.ref_ratio());
    }
    phase_timer.stop();

    // solve coarsest
    auto& coarse_lvl = _hierarchy.level(0);
    const auto& coarse_geom = coarse_lvl.geometry();
    auto coarse_phi = coarse_lvl.accessor(_phi);
    auto coarse_rhs = coarse_lvl.accessor(_rhs);
    zero_field(coarse_phi, coarse_geom);

    auto fill_bc_coarse = [&](FieldAccessor<T, Dims>& acc) {
      _bc_registry.fill_all(acc, coarse_geom);
    };
    fill_bc_coarse(coarse_phi);
    _smoother.smooth(coarse_phi, coarse_rhs, coarse_geom,
                     _n_coarse_smooth, fill_bc_coarse);

    // work upward
    for (int lvl = 1; lvl <= finest; ++lvl) {
      auto& level = _hierarchy.level(lvl);
      const auto& geom = level.geometry();
      auto phi_acc = level.accessor(_phi);

      zero_field(phi_acc, geom);
      auto coarser_phi = _hierarchy.level(lvl - 1).accessor(_phi);

      phase_timer.start("fmg_prolongation");
      Prolongation<T, Dims>::apply(coarser_phi, phi_acc, geom,
                                   _hierarchy.ref_ratio());
      phase_timer.stop();

      V_cycle(lvl, false);
    }
  }

  // Returns total number of V-cycles to convergence
  int solve() {
    phase_timer.reset();
    convergence_history.clear();

    int finest = _hierarchy.finest_level();
    HighResTimer solve_timer;
    solve_timer.start();

    // FMG
    {
      phase_timer.start("fmg_total");
      full_multigrid_cycle();
      phase_timer.stop();
    }

    // record FMG residual
    {
      auto& lvl = _hierarchy.level(finest);
      const auto& geom = lvl.geometry();
      auto phi_acc = lvl.accessor(_phi);
      auto rhs_acc = lvl.accessor(_rhs);
      auto res_acc = lvl.accessor(_res);
      _bc_registry.fill_all(phi_acc, geom);
      T res_norm = residual<StandardFlux, T, Dims>::compute(
          phi_acc, rhs_acc, res_acc, geom);
      convergence_history.push_back({0, static_cast<double>(res_norm),
                                     solve_timer.elapsed_s(), 0.0});
    }

    // V-cycle iterations
    int cycles_done = 0;
    for (int cycle = 0; cycle < _max_vcycles; ++cycle) {
      HighResTimer cycle_timer;
      cycle_timer.start();

      V_cycle(finest, false);

      double cycle_time = cycle_timer.elapsed_s();

      auto& lvl = _hierarchy.level(finest);
      const auto& geom = lvl.geometry();
      auto phi_acc = lvl.accessor(_phi);
      auto rhs_acc = lvl.accessor(_rhs);
      auto res_acc = lvl.accessor(_res);
      _bc_registry.fill_all(phi_acc, geom);
      T res_norm = residual<StandardFlux, T, Dims>::compute(
          phi_acc, rhs_acc, res_acc, geom);

      convergence_history.push_back({cycle + 1, static_cast<double>(res_norm),
                                     solve_timer.elapsed_s(), cycle_time});
      cycles_done = cycle + 1;

      if (res_norm < _tol) break;
    }

    return cycles_done;
  }

  double total_solve_time() const {
    if (convergence_history.empty()) return 0.0;
    return convergence_history.back().wall_time_s;
  }

private:
  static void zero_field(FieldAccessor<T, Dims>& acc,
                         const GridGeometry<T, Dims>& geom) {
    std::array<int, Dims> idx{};
    InteriorLoop<Dims, 0>::run(geom._n_interior, idx,
        [&](const std::array<int, Dims>& cell) { acc[cell] = T{0}; });
  }
};

// ============================================================================
// BenchmarkSuite: runs all benchmark studies and exports CSV
// ============================================================================

template<typename T, int Dims>
class BenchmarkSuite {
public:
  using TrackedAlloc = HostAllocator<T, track_allocations>;

  struct Config {
    std::string output_dir = "build/benchmarks";
    int n_warmup = 1;       // warmup runs (discarded)
    int n_repeat = 5;       // timed runs for statistics
    int n_pre_smooth = 2;
    int n_post_smooth = 2;
    int n_coarse_smooth = 20;
    int max_vcycles = 50;
    T tol = T{1e-10};
  };

  explicit BenchmarkSuite(Config cfg = {}) : _cfg(cfg) {}

  // ---- 1. Solver convergence profile for a single grid size ----
  void run_convergence_profile(int N) {
    std::printf("\n========================================\n");
    std::printf("  CONVERGENCE PROFILE: N=%d\n", N);
    std::printf("========================================\n");

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    auto [solver, hierarchy] = build_problem(N);
    int n_levels = hierarchy->num_levels();

    solver->solve();

    // print convergence history
    std::printf("\n  Cycle  Residual          CycleTime(ms)  Factor\n");
    for (size_t i = 0; i < solver->convergence_history.size(); ++i) {
      const auto& rec = solver->convergence_history[i];
      double factor = (i > 0 && solver->convergence_history[i - 1].residual_norm > 0.0)
                          ? rec.residual_norm / solver->convergence_history[i - 1].residual_norm
                          : 0.0;
      std::printf("  %4d   %.6e   %10.3f       %.4f\n",
                  rec.cycle, rec.residual_norm, rec.cycle_time_s * 1e3, factor);
    }

    // phase breakdown
    solver->phase_timer.print_summary("V-Cycle Phase Breakdown");

    // CSV export
    ensure_output_dir();
    {
      char path[256];
      std::snprintf(path, sizeof(path), "%s/convergence_N%d.csv",
                    _cfg.output_dir.c_str(), N);
      CSVWriter w(path);
      w.write_convergence(solver->convergence_history, N, n_levels, n_threads);
    }
    {
      char path[256];
      std::snprintf(path, sizeof(path), "%s/phases_N%d.csv",
                    _cfg.output_dir.c_str(), N);
      CSVWriter w(path);
      w.write_phase_breakdown(solver->phase_timer, N, n_levels, n_threads);
    }
  }

  // ---- 2. Grid scaling study: time vs N ----
  void run_grid_scaling(const std::vector<int>& grid_sizes) {
    std::printf("\n========================================\n");
    std::printf("  GRID SCALING STUDY\n");
    std::printf("========================================\n");

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // throughput data: (N, total_dof, solve_time, dof/s, vcycles)
    std::vector<std::tuple<int, double, double, double, int>> throughput_data;
    // scaling data: (N, solve_time, dof/s, efficiency)
    std::vector<std::tuple<int, double, double, double>> scaling_data;

    double base_dof_per_s = 0.0;

    for (int N : grid_sizes) {
      std::printf("\n  --- N = %d ---\n", N);

      std::vector<double> times;
      std::vector<double> dof_rates;
      int vcycles_last = 0;
      double total_dof = 0.0;

      for (int run = 0; run < _cfg.n_warmup + _cfg.n_repeat; ++run) {
        auto [solver, hierarchy] = build_problem(N);
        total_dof = static_cast<double>(hierarchy->finest().geometry().total_interior_cells());

        int vc = solver->solve();
        double t = solver->total_solve_time();

        if (run >= _cfg.n_warmup) {
          times.push_back(t);
          dof_rates.push_back(total_dof / t);
          vcycles_last = vc;
        }
      }

      auto time_stats = Stats::compute(times);
      auto rate_stats = Stats::compute(dof_rates);

      time_stats.print("Solve time", "ms");
      rate_stats.print("DOF/s throughput", "us");
      std::printf("  V-cycles to converge: %d\n", vcycles_last);

      if (base_dof_per_s == 0.0) base_dof_per_s = rate_stats.mean;
      double efficiency = rate_stats.mean / base_dof_per_s;

      throughput_data.emplace_back(N, total_dof, time_stats.mean,
                                   rate_stats.mean, vcycles_last);
      scaling_data.emplace_back(N, time_stats.mean, rate_stats.mean, efficiency);
    }

    // CSV export
    ensure_output_dir();
    {
      char path[256];
      std::snprintf(path, sizeof(path), "%s/grid_scaling_threads%d.csv",
                    _cfg.output_dir.c_str(), n_threads);
      CSVWriter w(path);
      w.write_scaling(scaling_data, "N");
    }
    {
      char path[256];
      std::snprintf(path, sizeof(path), "%s/throughput_threads%d.csv",
                    _cfg.output_dir.c_str(), n_threads);
      CSVWriter w(path);
      w.write_throughput(throughput_data);
    }
  }

  // ---- 3. Thread scaling study (strong scaling) ----
  // Fixed problem size N, vary thread count. Reports speedup & efficiency.
  void run_thread_scaling([[maybe_unused]] int N,
                          [[maybe_unused]] const std::vector<int>& thread_counts) {
#ifndef _OPENMP
    std::printf("\n  Thread scaling requires OpenMP (compile with OMP=1). Skipping.\n");
    return;
#else
    std::printf("\n========================================\n");
    std::printf("  STRONG SCALING: N=%d (fixed problem, vary threads)\n", N);
    std::printf("========================================\n");

    double total_dof = 0.0;
    // (threads, time, speedup, efficiency, dof/s)
    std::vector<std::tuple<int, double, double, double, double>> scaling_data;
    double base_time = 0.0;

    std::printf("\n  %-8s %-14s %-10s %-12s %-12s %-14s\n",
                "Threads", "Time (ms)", "Speedup", "Ideal", "Efficiency", "DOF/s");
    std::printf("  %-8s %-14s %-10s %-12s %-12s %-14s\n",
                "--------", "--------------", "----------", "------------",
                "------------", "--------------");

    for (int nt : thread_counts) {
      omp_set_num_threads(nt);

      std::vector<double> times;

      for (int run = 0; run < _cfg.n_warmup + _cfg.n_repeat; ++run) {
        auto [solver, hierarchy] = build_problem(N);
        total_dof = static_cast<double>(hierarchy->finest().geometry().total_interior_cells());

        solver->solve();
        double t = solver->total_solve_time();

        if (run >= _cfg.n_warmup) {
          times.push_back(t);
        }
      }

      auto time_stats = Stats::compute(times);

      if (base_time == 0.0) base_time = time_stats.mean;
      double speedup = base_time / time_stats.mean;
      double efficiency = speedup / nt;
      double dof_s = total_dof / time_stats.mean;

      std::printf("  %-8d %-14.3f %-10.2f %-12.1f %-11.1f%% %-14.0f\n",
                  nt, time_stats.mean * 1e3, speedup,
                  static_cast<double>(nt), efficiency * 100.0, dof_s);

      scaling_data.emplace_back(nt, time_stats.mean, speedup, efficiency, dof_s);
    }

    ensure_output_dir();
    char path[256];
    std::snprintf(path, sizeof(path), "%s/strong_scaling_N%d.csv",
                  _cfg.output_dir.c_str(), N);
    CSVWriter w(path);
    w.write_thread_scaling(scaling_data, N);
#endif
  }

  // ---- 3b. Weak scaling: fixed DOF per thread, grow problem with threads ----
  void run_weak_scaling([[maybe_unused]] int base_N,
                        [[maybe_unused]] const std::vector<int>& thread_counts) {
#ifndef _OPENMP
    std::printf("\n  Weak scaling requires OpenMP (compile with OMP=1). Skipping.\n");
    return;
#else
    std::printf("\n========================================\n");
    std::printf("  WEAK SCALING: base N=%d (fixed DOF/thread)\n", base_N);
    std::printf("========================================\n");

    // (threads, N, dof, time, efficiency)
    std::vector<std::tuple<int, int, double, double, double>> scaling_data;
    double base_time = 0.0;

    std::printf("\n  %-8s %-8s %-12s %-14s %-12s\n",
                "Threads", "N", "DOF", "Time (ms)", "Efficiency");
    std::printf("  %-8s %-8s %-12s %-14s %-12s\n",
                "--------", "--------", "------------", "--------------", "------------");

    for (int nt : thread_counts) {
      omp_set_num_threads(nt);

      // Scale N so total DOF ~ nt * base_N^2
      // For 2D: N = base_N * sqrt(nt), rounded to nearest power of 2
      double scale = std::sqrt(static_cast<double>(nt));
      int N = static_cast<int>(base_N * scale);
      // Round to nearest power of 2 for clean multigrid levels
      int N_pow2 = 1;
      while (N_pow2 * 2 <= N) N_pow2 *= 2;
      if (N - N_pow2 > N_pow2 * 2 - N) N_pow2 *= 2;
      N = N_pow2;

      std::vector<double> times;
      double total_dof = 0.0;

      for (int run = 0; run < _cfg.n_warmup + _cfg.n_repeat; ++run) {
        auto [solver, hierarchy] = build_problem(N);
        total_dof = static_cast<double>(hierarchy->finest().geometry().total_interior_cells());

        solver->solve();
        double t = solver->total_solve_time();

        if (run >= _cfg.n_warmup) {
          times.push_back(t);
        }
      }

      auto time_stats = Stats::compute(times);

      if (base_time == 0.0) base_time = time_stats.mean;
      double efficiency = base_time / time_stats.mean;

      std::printf("  %-8d %-8d %-12.0f %-14.3f %-11.1f%%\n",
                  nt, N, total_dof, time_stats.mean * 1e3, efficiency * 100.0);

      scaling_data.emplace_back(nt, N, total_dof, time_stats.mean, efficiency);
    }

    ensure_output_dir();
    char path[256];
    std::snprintf(path, sizeof(path), "%s/weak_scaling_base%d.csv",
                  _cfg.output_dir.c_str(), base_N);
    CSVWriter w(path);
    w.write_weak_scaling(scaling_data);
#endif
  }

  // ---- 4. Memory profile across grid sizes ----
  void run_memory_profile(const std::vector<int>& grid_sizes) {
    std::printf("\n========================================\n");
    std::printf("  MEMORY PROFILE\n");
    std::printf("========================================\n");

    std::vector<std::tuple<int, size_t, size_t, size_t>> memory_data;

    for (int N : grid_sizes) {
      std::printf("\n  --- N = %d ---\n", N);

      // Reset tracking counters
      MemorySnapshot::reset_counters();

      auto [solver, hierarchy] = build_problem(N);

      // Snapshot after allocation, before solve
      auto pre_solve = MemorySnapshot::capture();
      pre_solve.print("Pre-solve");

      // Reset event counter to measure solve-only allocations
      size_t pre_events = pre_solve.cumulative_events;

      solver->solve();

      auto post_solve = MemorySnapshot::capture();

      size_t solve_allocs = post_solve.cumulative_events - pre_events;
      std::printf("  Allocations during solve: %zu\n", solve_allocs);

      // Compute bytes per DOF
      double dof = static_cast<double>(hierarchy->finest().geometry().total_interior_cells());
      double bytes_per_dof = post_solve.active_bytes / dof;
      std::printf("  Bytes per DOF (finest): %.1f\n", bytes_per_dof);

      // Total bytes across all levels
      size_t total_cells_all_levels = 0;
      for (int lvl = 0; lvl < hierarchy->num_levels(); ++lvl) {
        total_cells_all_levels += hierarchy->level(lvl).geometry().total_cells();
      }
      std::printf("  Total cells (all levels): %zu\n", total_cells_all_levels);

      memory_data.emplace_back(N, post_solve.active_bytes,
                               post_solve.active_allocations,
                               post_solve.cumulative_events);
    }

    ensure_output_dir();
    char path[256];
    std::snprintf(path, sizeof(path), "%s/memory_profile.csv",
                  _cfg.output_dir.c_str());
    CSVWriter w(path);
    w.write_memory(memory_data);
  }

  // ---- 5. Solution space complexity verification ----
  // Verifies O(N) total work: time/DOF should be constant across grid sizes.
  // Also computes asymptotic convergence rate for each grid size.
  void run_solution_space(const std::vector<int>& grid_sizes) {
    std::printf("\n========================================\n");
    std::printf("  SOLUTION SPACE COMPLEXITY\n");
    std::printf("========================================\n");

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // (N, dof, time, time_per_dof_us, dof_s, vcycles, conv_rate)
    std::vector<std::tuple<int, double, double, double, double, int, double>> data;

    std::printf("\n  %-6s %-10s %-12s %-14s %-14s %-8s %-10s\n",
                "N", "DOF", "Time (ms)", "Time/DOF (us)", "DOF/s", "Vcyc", "Conv Rate");
    std::printf("  %-6s %-10s %-12s %-14s %-14s %-8s %-10s\n",
                "------", "----------", "------------", "--------------",
                "--------------", "--------", "----------");

    for (int N : grid_sizes) {
      std::vector<double> times;
      std::vector<double> conv_rates;
      int vcycles_last = 0;
      double total_dof = 0.0;

      for (int run = 0; run < _cfg.n_warmup + _cfg.n_repeat; ++run) {
        auto [solver, hierarchy] = build_problem(N);
        total_dof = static_cast<double>(hierarchy->finest().geometry().total_interior_cells());

        int vc = solver->solve();
        double t = solver->total_solve_time();

        if (run >= _cfg.n_warmup) {
          times.push_back(t);
          vcycles_last = vc;

          // Compute asymptotic convergence rate (geometric mean of last 3 factors)
          const auto& hist = solver->convergence_history;
          if (hist.size() >= 4) {
            double rate_product = 1.0;
            int rate_count = 0;
            for (size_t i = hist.size() - 3; i < hist.size(); ++i) {
              if (hist[i - 1].residual_norm > 0.0) {
                rate_product *= hist[i].residual_norm / hist[i - 1].residual_norm;
                rate_count++;
              }
            }
            if (rate_count > 0) {
              conv_rates.push_back(std::pow(rate_product, 1.0 / rate_count));
            }
          }
        }
      }

      auto time_stats = Stats::compute(times);
      auto rate_stats = Stats::compute(conv_rates);

      double time_per_dof_us = (time_stats.mean / total_dof) * 1e6;
      double dof_s = total_dof / time_stats.mean;

      std::printf("  %-6d %-10.0f %-12.3f %-14.4f %-14.0f %-8d %-10.4f\n",
                  N, total_dof, time_stats.mean * 1e3, time_per_dof_us,
                  dof_s, vcycles_last, rate_stats.mean);

      data.emplace_back(N, total_dof, time_stats.mean, time_per_dof_us,
                        dof_s, vcycles_last, rate_stats.mean);
    }

    // Check O(N) scaling: ratio of time/DOF at largest vs smallest
    if (data.size() >= 2) {
      double first_tpd = std::get<3>(data.front());
      double last_tpd = std::get<3>(data.back());
      double ratio = last_tpd / first_tpd;
      std::printf("\n  O(N) verification: time/DOF ratio (largest/smallest) = %.2f\n", ratio);
      std::printf("  (Should be ~1.0 for optimal multigrid; >2.0 indicates sub-optimal scaling)\n");
    }

    ensure_output_dir();
    char path[256];
    std::snprintf(path, sizeof(path), "%s/solution_space_threads%d.csv",
                  _cfg.output_dir.c_str(), n_threads);
    CSVWriter w(path);
    w.write_solution_space(data, n_threads);
  }

  // ---- 6. Per-level timing breakdown ----
  void run_level_profile(int N) {
    std::printf("\n========================================\n");
    std::printf("  PER-LEVEL TIMING: N=%d\n", N);
    std::printf("========================================\n");

    auto [solver, hierarchy] = build_problem(N);
    solver->solve();

    // The phase timer already captured everything
    solver->phase_timer.print_summary("Solver Phase Breakdown");

    // Print per-level grid info
    std::printf("\n  Level details:\n");
    std::printf("  %-6s %-12s %-12s %-12s\n", "Level", "Interior", "Total", "Spacing");
    for (int lvl = 0; lvl < hierarchy->num_levels(); ++lvl) {
      const auto& geom = hierarchy->level(lvl).geometry();
      std::printf("  %-6d %-12d %-12d %-12.6f\n",
                  lvl, geom.total_interior_cells(), geom.total_cells(),
                  static_cast<double>(geom._spacing[0]));
    }

    // Estimate effective memory bandwidth
    // Each RBGS sweep reads ~5 doubles and writes 1 (2D stencil)
    // per cell per colour pass
    double dof = static_cast<double>(hierarchy->finest().geometry().total_interior_cells());
    double smooth_time = solver->phase_timer.total_time("pre_smooth")
                       + solver->phase_timer.total_time("post_smooth");
    int smooth_calls = solver->phase_timer.call_count("pre_smooth")
                     + solver->phase_timer.call_count("post_smooth");
    if (smooth_time > 0.0 && smooth_calls > 0) {
      // Each smooth call does n_smooth sweeps × 2 colour passes × (5R + 1W) × 8 bytes
      double bytes_per_sweep = dof * 2.0 * (5.0 + 1.0) * sizeof(T);
      double total_smooth_bytes = bytes_per_sweep * _cfg.n_pre_smooth * smooth_calls;
      double bandwidth_gb_s = total_smooth_bytes / smooth_time / 1e9;
      std::printf("\n  Estimated smoother bandwidth: %.2f GB/s\n", bandwidth_gb_s);
      std::printf("  (Based on %d smooth calls, %.0f DOF)\n", smooth_calls, dof);
    }
  }

  // ---- Run all benchmarks ----
  void run_all() {
    std::printf("╔════════════════════════════════════════════════╗\n");
    std::printf("║  MEADOWS-CPP BENCHMARK SUITE                  ║\n");
    std::printf("║  Geometric Multigrid Solver Performance        ║\n");
    std::printf("╚════════════════════════════════════════════════╝\n");

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::printf("  Threads: %d\n", n_threads);
    std::printf("  Precision: %s\n", sizeof(T) == 8 ? "double" : "float");
    std::printf("  Dimensions: %d\n", Dims);
    std::printf("  Warmup runs: %d\n", _cfg.n_warmup);
    std::printf("  Timed runs: %d\n", _cfg.n_repeat);

    // Convergence profile at medium resolution
    run_convergence_profile(128);

    // Grid scaling
    run_grid_scaling({16, 32, 64, 128, 256, 512});

    // Solution space complexity (O(N) verification + convergence rates)
    run_solution_space({16, 32, 64, 128, 256, 512});

    // Memory profile
    run_memory_profile({16, 32, 64, 128, 256, 512});

    // Per-level breakdown at high resolution
    run_level_profile(256);

    // Thread scaling (if OpenMP)
#ifdef _OPENMP
    run_thread_scaling(256, {1, 2, 4, 8});
    run_weak_scaling(128, {1, 2, 4, 8});
#endif

    std::printf("\n========================================\n");
    std::printf("  ALL BENCHMARKS COMPLETE\n");
    std::printf("  Results in: %s/\n", _cfg.output_dir.c_str());
    std::printf("========================================\n");
  }

private:
  Config _cfg;

  void ensure_output_dir() {
    // Use a simple system call to mkdir
    std::string cmd = "mkdir -p " + _cfg.output_dir;
    std::system(cmd.c_str());
  }

  // Build a standard Gaussian charge Poisson problem
  // Returns unique_ptrs to solver and hierarchy so they live long enough
  struct ProblemPack {
    std::unique_ptr<InstrumentedSolver<T, Dims, TrackedAlloc>> solver;
    std::unique_ptr<GridHierarchy<T, Dims, TrackedAlloc>> hierarchy;
    std::unique_ptr<BCRegistry<T, Dims>> bc_reg;
  };

  auto build_problem(int N)
      -> std::pair<std::unique_ptr<InstrumentedSolver<T, Dims, TrackedAlloc>>,
                   std::unique_ptr<GridHierarchy<T, Dims, TrackedAlloc>>> {

    T dx = T{1} / static_cast<T>(N);
    int n_levels = static_cast<int>(std::log2(N)) - 1;
    if (n_levels < 2) n_levels = 2;

    // We store the BC registry alongside the hierarchy via a shared pointer trick
    // Instead, we use a static BC registry since Dirichlet zero is always the same
    static DirichletBC<T, Dims> bc_zero;
    bc_zero.value = T{0};
    static BCRegistry<T, Dims> bc_reg;
    static bool bc_init = false;
    if (!bc_init) {
      for (int dim = 0; dim < Dims; ++dim) {
        for (int side = 0; side < 2; ++side) {
          bc_reg.set(dim, side, bc_zero);
        }
      }
      bc_init = true;
    }

    std::array<T, Dims> origin, spacing;
    std::array<int, Dims> n_interior;
    for (int d = 0; d < Dims; ++d) {
      origin[d] = dx / T{2};
      spacing[d] = dx;
      n_interior[d] = N;
    }

    GridGeometry<T, Dims> finest_geom(origin, spacing, n_interior, 0, 1);

    auto hierarchy = std::make_unique<GridHierarchy<T, Dims, TrackedAlloc>>();
    int phi_idx = hierarchy->register_component("phi");
    int rhs_idx = hierarchy->register_component("rhs");
    int res_idx = hierarchy->register_component("res");
    hierarchy->build(finest_geom, n_levels);

    // Fill RHS: Gaussian charge
    auto rhs_acc = hierarchy->finest().accessor(rhs_idx);
    T sigma = T{0.05};
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        T x = (i + T{0.5}) * dx;
        T y = (j + T{0.5}) * dx;
        T cx = T{0.5}, cy_val = T{0.5};
        T r2 = (x - cx) * (x - cx) + (y - cy_val) * (y - cy_val);
        rhs_acc(i, j) = -std::exp(-r2 / (T{2} * sigma * sigma));
      }
    }

    auto solver = std::make_unique<InstrumentedSolver<T, Dims, TrackedAlloc>>(
        *hierarchy, bc_reg, phi_idx, rhs_idx, res_idx,
        _cfg.n_pre_smooth, _cfg.n_post_smooth, _cfg.n_coarse_smooth,
        _cfg.max_vcycles, _cfg.tol);

    return {std::move(solver), std::move(hierarchy)};
  }
};
