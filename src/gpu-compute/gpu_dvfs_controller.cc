#include "gpu-compute/gpu_dvfs_controller.hh" 
#include "debug/GPU_DVFS.hh" // DPRINTF
#include "gpu-compute/compute_unit.hh" // cu_id, CRISP counters
#include "sim/dvfs_handler.hh" // Perf levels
#include "sim/core.hh" // for sim_clock::Frequency
#include <cmath>
#include <limits> // for std::numeric_limits

namespace gem5
{

GPUDVFSController::GPUDVFSController(const Params &p)
    : SimObject(p),
      dvfsHandler(p.dvfs_handler),
      computeUnit(p.compute_unit),
      evaluationPeriod(p.evaluation_period),
      enableFrequencyTransitions(p.enable_frequency_transitions),
      evaluateEvent([this]{ evaluateAndAdjust(); },
                    name() + ".evaluateEvent")
{
    DPRINTF(GPU_DVFS, "GPU DVFS Controller created for CU %d, "
            "eval period %lu ticks\n",
            computeUnit->cu_id, evaluationPeriod);
}

void
GPUDVFSController::startup()
{
    // Debug: Print CU info
    DPRINTF(GPU_DVFS, "GPU DVFS Controller: CU[%d] at %p, activeWaves=%d\n",
         computeUnit->cu_id, computeUnit, computeUnit->activeWaves);

    // Take cu_id as the domain id. 
    int domain_id = computeUnit->cu_id;

    // Get the number of VF levels
    DVFSHandler::PerfLevel numLevels = dvfsHandler->numPerfLevels(domain_id);
    DPRINTF(GPU_DVFS, "GPU DVFS Controller: CU[%d] has %d performance levels:\n",
         computeUnit->cu_id, numLevels);

    // Print the VF levels available
    for (DVFSHandler::PerfLevel i = 0; i < numLevels; i++) {
        Tick period = dvfsHandler->clkPeriodAtPerfLevel(domain_id, i);
        double freq = tickToFrequencyMHz(period);
        double voltage = dvfsHandler->voltageAtPerfLevel(domain_id, i);
        DPRINTF(GPU_DVFS, "  Level %d: %.2f MHz, %.3f V\n", i, freq, voltage);
    }

    // Schedule the first evaluation at current tick + Evaluation period. 
    schedule(&evaluateEvent, curTick() + evaluationPeriod);
    DPRINTF(GPU_DVFS, "GPU DVFS Controller for CU %d started, first eval at tick %lu\n",
            computeUnit->cu_id, curTick() + evaluationPeriod);
}

double
GPUDVFSController::tickToFrequencyMHz(Tick clkPeriod) const
{
    if (clkPeriod == 0) {
        DPRINTF(GPU_DVFS, "GPUDVFSController: Clock period is 0, returning 0 MHz\n");
        return 0.0;
    }

    double freqHz = static_cast<double>(sim_clock::Frequency) / clkPeriod;
    double freqMHz = freqHz / 1e6; // Convert to MHz
    return freqMHz;
}


// TODO : currently the energy numbers are a part of crispcounters. need 
// to move them to a better named struct. 
bool
GPUDVFSController::extractAveragePower(
    const ComputeUnit::CRISPStatCount &crispCounters,
    double &staticPower, double &dynamicPower) const
{
    // Minimum time threshold to avoid division by very small numbers
    constexpr double MIN_TIME_DELTA = 1e-6;  // 1 microsecond

    // Check if we have valid time delta
    if (crispCounters.totalSimSecondsDelta < MIN_TIME_DELTA) {
        DPRINTF(GPU_DVFS, "CU %d: Time delta too small (%.12f s), "
                "cannot extract power\n",
                computeUnit->cu_id, crispCounters.totalSimSecondsDelta);
        return false;
    }

    // Check if we have energy data (both zero means no power model)
    if (crispCounters.totalDynamicEnergy == 0.0 &&
        crispCounters.totalStaticEnergy == 0.0) {
        DPRINTF(GPU_DVFS, "CU %d: No energy data available, "
                "power model may not be attached\n",
                computeUnit->cu_id);
        return false;
    }

    // Extract average power from energy measurements
    staticPower = static_cast<double>(
        crispCounters.totalStaticEnergy / crispCounters.totalSimSecondsDelta
    );
    dynamicPower = static_cast<double>(
        crispCounters.totalDynamicEnergy / crispCounters.totalSimSecondsDelta
    );

    DPRINTF(GPU_DVFS, "CU %d: Extracted power - Static: %.3f W, Dynamic: %.3f W "
            "(Energy: %.6f J / %.6f s, %.6f J / %.6f s)\n",
            computeUnit->cu_id, staticPower, dynamicPower,
            crispCounters.totalStaticEnergy, crispCounters.totalSimSecondsDelta,
            crispCounters.totalDynamicEnergy, crispCounters.totalSimSecondsDelta);

    return true;
}

uint64_t
GPUDVFSController::calculateCRISPDelay(const ComputeUnit::CRISPStatCount &counters, double currentFreqMHz, double targetFreqMHz) const
{
    uint64_t Tcomp_LCP  = (uint64_t)std::ceil((currentFreqMHz / targetFreqMHz) * counters.OverlappedCompute);
    uint64_t TLCP       = (counters.TStall_LCP >= Tcomp_LCP) ? counters.TStall_LCP : Tcomp_LCP;

    uint64_t Tcomp_CSP_scaled = (uint64_t)std::ceil((currentFreqMHz / targetFreqMHz) * counters.PureCompute);
    uint64_t TCSP             = (Tcomp_CSP_scaled >= (counters.PureCompute + counters.TStall_CSP)) ?
                                    Tcomp_CSP_scaled : (counters.PureCompute + counters.TStall_CSP);

    uint64_t Tdelay = TLCP + TCSP;

    return Tdelay;
}

double
GPUDVFSController::calculateCRISPEDP(const ComputeUnit::CRISPStatCount &counters,
                                     double staticPower,    double dynamicPower,
                                     double currentFreqMHz, double targetFreqMHz,
                                     double voltageCurrent, double voltageTarget) const
{
    // Guard against division by zero
    if (currentFreqMHz == 0.0 || targetFreqMHz == 0.0) {
        DPRINTF(GPU_DVFS, "GPUDVFSController: Invalid frequency (current=%.2f, target=%.2f)\n",
             currentFreqMHz, targetFreqMHz);
        return std::numeric_limits<double>::max();
    }

    uint64_t Tdelay = calculateCRISPDelay(counters, currentFreqMHz, targetFreqMHz);

    double staticPowerScaled = staticPower * (voltageTarget / voltageCurrent);
    
    double dynamicPowerScaled = dynamicPower * ((voltageTarget * voltageTarget) / (voltageCurrent * voltageCurrent)) * (targetFreqMHz / currentFreqMHz);
    
    double Estatic = staticPowerScaled * Tdelay;
    
    double Edynamic = dynamicPowerScaled * Tcurrent;
    
    double Etotal = Estatic + Edynamic;
    
    double EDP = Etotal * Tdelay;
    
    return EDP;
}

int
GPUDVFSController::selectOptimalFrequencyEDP(
    const ComputeUnit::CRISPStatCount &crispCounters) const
{
    int domain_id = computeUnit->cu_id;
    DVFSHandler::PerfLevel currentLevel = dvfsHandler->perfLevel(domain_id);
    DVFSHandler::PerfLevel numLevels = dvfsHandler->numPerfLevels(domain_id);

    // Edge case: No CRISP data collected
    bool hasData = (crispCounters.TMemoryStallCycles > 0 ||
                    crispCounters.TStall_LCP > 0 ||
                    crispCounters.TStall_CSP > 0 ||
                    crispCounters.OverlappedCompute > 0 ||
                    crispCounters.PureCompute > 0);

    if (!hasData) {
        DPRINTF(GPU_DVFS, "CU %d: No CRISP data, staying at level %d\n",
                computeUnit->cu_id, currentLevel);
        return currentLevel;
    }

    // Edge case: Only one performance level
    if (numLevels <= 1) {
        return 0;
    }

    // Get current frequency and voltage
    Tick currentPeriod = dvfsHandler->clkPeriodAtPerfLevel(domain_id, currentLevel);
    double currentFreqMHz = tickToFrequencyMHz(currentPeriod);
    double currentVoltage = dvfsHandler->voltageAtPerfLevel(domain_id, currentLevel);

    // Extract average power from accumulated energy measurements
    double staticPower, dynamicPower;
    if (!extractAveragePower(crispCounters, staticPower, dynamicPower)) {
        // No valid power data - stay at current level
        DPRINTF(GPU_DVFS, "CU %d: No valid power data, staying at level %d\n",
                computeUnit->cu_id, currentLevel);
        return currentLevel;
    }

    // Calculate EDP for each performance level
    double minEDP = std::numeric_limits<double>::max();
    int optimalLevel = currentLevel;

    for (DVFSHandler::PerfLevel level = 0; level < numLevels; level++) {
        Tick targetPeriod = dvfsHandler->clkPeriodAtPerfLevel(domain_id, level);
        double targetFreqMHz = tickToFrequencyMHz(targetPeriod);
        double targetVoltage = dvfsHandler->voltageAtPerfLevel(domain_id, level);

        // Calculate EDP for this target level using measured power
        double edp = calculateCRISPEDP(crispCounters,
                                      staticPower, dynamicPower,
                                      currentFreqMHz, targetFreqMHz,
                                      currentVoltage, targetVoltage);

        DPRINTF(GPU_DVFS, "  Level %d: freq=%.2f MHz, V=%.3f, EDP=%.6e\n",
                level, targetFreqMHz, targetVoltage, edp);

        if (edp < minEDP) {
            minEDP = edp;
            optimalLevel = level;
        }
    }

    DPRINTF(GPU_DVFS, "CU %d: Optimal level=%d (minEDP=%.6e)\n",
            computeUnit->cu_id, optimalLevel, minEDP);

    return optimalLevel;
}

void
GPUDVFSController::evaluateAndAdjust()
{
    // Get and clear cycle log and CRISP counters
    auto [cycleCatCounters, crispCounters] = computeUnit->dumpAndClearCycleLog();
    (void)cycleCatCounters;

    // Use CRISP counters for EDP-based frequency selection
    int currentLevel = dvfsHandler->perfLevel(computeUnit->cu_id);
    int newLevel = selectOptimalFrequencyEDP(crispCounters);

    // Debug: Log CRISP counter summary
    DPRINTF(GPU_DVFS, "CU %d CRISP: MemStall=%lu, LCP=%lu, CSP=%lu, "
            "Overlap=%lu, Pure=%lu\n",
            computeUnit->cu_id,
            crispCounters.TMemoryStallCycles,
            crispCounters.TStall_LCP,
            crispCounters.TStall_CSP,
            crispCounters.OverlappedCompute,
            crispCounters.PureCompute);

    // Apply frequency change if needed
    if (newLevel != currentLevel && enableFrequencyTransitions) {
        DPRINTF(GPU_DVFS, "CU %d: EDP transition %d -> %d\n",
                computeUnit->cu_id, currentLevel, newLevel);
        adjustFrequency(newLevel);
    }

    // Schedule next evaluation
    schedule(&evaluateEvent, curTick() + evaluationPeriod);
}
double GPUDVFSController::computeIPC() {
  static int eval_count = 0;
  eval_count++;

  // Get current cumulative stats for this CU
  uint64_t currentInsts = computeUnit->stats.numInstrExecuted.value();
  uint64_t currentCycles = computeUnit->stats.totalCycles.value();

  // Debug: Print raw stats periodically or when non-zero
  if (eval_count % 100 == 0 || currentInsts > 0) {
    DPRINTF(GPU_DVFS, 
        "  [Eval %d] CU[%d]: activeWaves=%d, totalInsts=%lu, totalCycles=%lu\n",
        eval_count, computeUnit->cu_id, computeUnit->activeWaves, currentInsts,
        currentCycles);
  }

  // Compute deltas since last check
  uint64_t deltaInsts = currentInsts - lastInstCount;
  uint64_t deltaCycles = currentCycles - lastCycleCount;

  // Update history
  lastInstCount = currentInsts;
  lastCycleCount = currentCycles;

  // Compute IPC for this period
  double ipc = 0.0;
  if (deltaCycles > 0) {
    ipc = static_cast<double>(deltaInsts) / deltaCycles;
    DPRINTF(GPU_DVFS, "CU %d: deltaInsts=%lu, deltaCycles=%lu, IPC=%.3f\n",
            computeUnit->cu_id, deltaInsts, deltaCycles, ipc);
  }

  return ipc;
}
void
GPUDVFSController::adjustFrequency(int newLevel)
{
    // Use cu_id as domain_id (CU 0 has domain_id 0, CU 1 has domain_id 1, etc.)
    int domain_id = computeUnit->cu_id;
    int currentLevel = dvfsHandler->perfLevel(domain_id);

    DPRINTF(GPU_DVFS, "Adjusting CU %d frequency: Level %d -> %d at tick %lu\n",
            computeUnit->cu_id, currentLevel, newLevel, curTick());

    // Request DVFS change for this CU's domain
    bool success = dvfsHandler->perfLevel(domain_id, newLevel);

    if (!success) {
        DPRINTF(GPU_DVFS, "CU %d DVFS: Failed to transition from level %d to %d\n",
             computeUnit->cu_id, currentLevel, newLevel);
    }
}

} // namespace gem5
