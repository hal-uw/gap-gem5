#-------------------------------------------------------------------------------------#
# SimObject wrapper for the GPU DVFS controller.
 
# Defines a configurable gem5 object that binds Python configs to the C++
# implementation (`gem5::GPUDVFSController`) and exposes the parameters needed to
# wire it to a `DVFSHandler` and a target `ComputeUnit` (plus basic DVFS timing and
# policy knobs).
#-------------------------------------------------------------------------------------#


from m5.params import *
from m5.proxy import *
from m5.SimObject import SimObject

class GPUDVFSController(SimObject):
    type        = "GPUDVFSController"
    cxx_header  = "gpu-compute/gpu_dvfs_controller.hh"
    cxx_class   = "gem5::GPUDVFSController"

    dvfs_handler = Param.DVFSHandler("DVFS handler to control")
    compute_unit = Param.ComputeUnit("Compute unit to monitor")

    #Knob to enable/disable the handler
    enable_frequency_transitions = Param.Bool(True, "Enable the GPU DVFS Handler.")

    #Knob to specify the evaluation window size
    evaluation_period = Param.Latency(
        "100us", "CRISP DVFS Evaluation window period")
    
    

    
