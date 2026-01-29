//---------------------------------------------------------------------------------//


//---------------------------------------------------------------------------------//


#ifndef __GPU_COMPUTE_GPU_DVFS_CONTROLLER_HH__
#define __GPU_COMPUTE_GPU_DVFS_CONTROLLER_HH__

#include <vector>
#include "params/GPUDVFSController.hh"
#include "gpu-compute/compute_unit.hh"
#include "sim/eventq.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class DVFSHandler;

class GPUDVFSController : public SimObject
{
  public:
    typedef GPUDVFSControllerParams Params;
    GPUDVFSController(const Params &p);
    void startup() override;

  private:
    // Configuration
    DVFSHandler *dvfsHandler;
    ComputeUnit *computeUnit;
    Tick evaluationPeriod;
    bool enableFrequencyTransitions;

    // Event for periodic evaluation
    EventFunctionWrapper evaluateEvent;

    // CRISP methods
    uint64_t calculateCRISPDelay(const ComputeUnit::CRISPCycleCounter &counters,
                                 double currentFreqMHz,
                                 double targetFreqMHz) const;

    double calculateCRISPEDP(const ComputeUnit::CRISPCycleCounter &counters,
                            double staticPower, double dynamicPower,
                            double currentFreqMHz, double targetFreqMHz,
                            double voltageCurrent, double voltageTarget) const;

    // Helper methods
    double tickToFrequencyMHz(Tick clkPeriod) const;
    int selectOptimalFrequencyEDP(const ComputeUnit::CRISPCycleCounter &crispCounters) const;

    // Extract average power from accumulated energy measurements
    // Returns true if valid power data available, false otherwise
    bool extractAveragePower(const ComputeUnit::CRISPCycleCounter &crispCounters,
                            double &staticPower, double &dynamicPower) const;

    // Core policy methods
    void evaluateAndAdjust();
    double computeIPC();
    void adjustFrequency(int newLevel);
};

} // namespace gem5

#endif // __GPU_COMPUTE_GPU_DVFS_CONTROLLER_HH__
