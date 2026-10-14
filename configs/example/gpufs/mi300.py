# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Create an X86 KVM system with a scaled MI300X (gfx942) GPU model.

The default configuration represents the compute and cache organization of
one 38-CU XCD while retaining the existing GPU-FS platform and memory-image
constraints. It is a simulation-scaled configuration, not a complete physical
MI300X package model. Most of this file constructs a runscript that loads an
application from the host and executes it inside gem5.
"""

import argparse
import base64
import os
import sys
import tempfile
from typing import Optional

import runfs
from amd import AmdGPUOptions
from common import (
    GPUTLBOptions,
    Options,
)
from ruby import Ruby

import m5

from gem5.resources.resource import AbstractResource

demo_runscript_without_checkpoint = """\
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HSA_ENABLE_INTERRUPT=0
export HCC_AMDGPU_TARGET=gfx942
echo 0 > /proc/sys/kernel/randomize_va_space
dmesg -n8
cat /proc/cpuinfo
dd if=/root/roms/mi300.rom of=/dev/mem bs=1k seek=768 count=128

# Check if exists (backwards compat with ROCm <7.0)
if [ -e /usr/lib/firmware/amdgpu/mi300_discovery ]; then
    rm -f /usr/lib/firmware/amdgpu/ip_discovery.bin
    ln -s /usr/lib/firmware/amdgpu/mi300_discovery /usr/lib/firmware/amdgpu/ip_discovery.bin
    echo "Here"
fi

if [ -f /home/gem5/load_amdgpu.sh ]; then
    bash /home/gem5/load_amdgpu.sh
elif [ ! -f /lib/modules/`uname -r`/updates/dkms/amdgpu.ko ]; then
    echo "ERROR: Missing DKMS package for kernel `uname -r`. Exiting gem5."
    # m5 exit
else
    # develop support
    echo "options amdgpu ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0 discovery=2" > /etc/modprobe.d/amdgpu.conf
    modprobe -v amdgpu
fi
modprobe -v amdgpu ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0 discovery=2

echo "Running {} {}"
echo "{}" | base64 -d > myapp
chmod +x myapp
./myapp {}
m5 exit
"""

demo_runscript_with_checkpoint = """\
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HSA_ENABLE_INTERRUPT=0
export HCC_AMDGPU_TARGET=gfx942
export HSA_OVERRIDE_GFX_VERSION="9.4.2"
echo 0 > /proc/sys/kernel/randomize_va_space
dmesg -n8
dd if=/root/roms/mi300.rom of=/dev/mem bs=1k seek=768 count=128
if [ ! -f /lib/modules/`uname -r`/updates/dkms/amdgpu.ko ]; then
    echo "ERROR: Missing DKMS package for kernel `uname -r`. Exiting gem5."
    /sbin/m5 exit
fi
modprobe -v amdgpu ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0 discovery=2
echo "Running {} {}"
echo "{}" | base64 -d > myapp
chmod +x myapp
/sbin/m5 checkpoint
./myapp {}
/sbin/m5 exit
"""


# Scaled MI300X GPU-FS configuration representing one 38-CU XCD.
SCALED_MI300_DEFAULTS = {
    "dgpu_mem_size": "16GiB",
    "dgpu_mem_type": "HBM3_MI300X_1x64",
    "dgpu_num_dirs": 64,
    "dgpu_mem_locality": 1,
    "cu_per_sa": 15,
    "num_compute_units": 38,
    "num_gpu_complex": 4,
    "gpu_clock": "2.1GHz",
    "fabric_clock": "1.3GHz",
    "gpu_topology": "Crossbar",
    "cpu_topology": "Crossbar",
    "gpu_mesh_routers": 0,
    "link_width_bits": 512,
    "cu_per_sqc": 2,
    "cu_per_scalar_cache": 2,
    "simds_per_cu": 4,
    "wfs_per_simd": 8,
    "wf_size": 64,
    "mfma_scale": 1.0,
    "hbm_ctrl": True,
    "issue_period": 2,
    "scalar_issue_period": 1,
    "lds_req_latency": 55,
    "reg_alloc_policy": "dynamic",
    "vreg_file_size": 2048,
    "sreg_file_size": 3200,
    "tcp_size": "32KiB",
    "tcp_assoc": 16,
    "tcp_num_banks": 16,
    "TCP_latency": 1,
    "TCP_latency_data": 1,
    "tcp_issue_latency": 25,
    "WB_L1": False,
    "noL1": False,
    "mandatory_queue_latency": 1,
    "mem_req_latency": 62,
    "mem_resp_latency": 62,
    "scalar_mem_req_latency": 40,
    "TCC_latency": 15,
    "tcc_size": "4MiB",
    "num_tccs": 16,
    "tcc_assoc": 16,
    "tcc_num_atomic_alus": 96,
    "tcc_num_banks": 32,
    "tcc_tag_access_latency": 1,
    "tcc_data_access_latency": 2,
    "l2_latency": 50,
    "WB_L2": True,
    "tcc_rp": "BRRIPRP",
    "l3_data_latency": 20,
    "l3_tag_latency": 15,
    "use_L3_on_WT": False,
    "use_gpu_l3": True,
    "l3_exclusive": False,
    "num_l3caches": 1,
    "l3_size": "256MiB",
    "l3_assoc": 16,
    "num_dirs": 4,
    "num_subcaches": 4,
    "cpu_to_dir_latency": 120,
    "gpu_to_dir_latency": 1,
    "vrf_lm_bus_latency": 3,
    "no_resource_stalls": False,
    "no_tcc_resource_stalls": True,
    "num_tbes": 512,
    "sqc_size": "64KiB",
    "sqc_assoc": 8,
    "scalar_size": "16KiB",
    "scalar_assoc": 8,
    "max_coalesces_per_cycle": 10,
    "max_cu_tokens": 160,
    "glc_atomic_latency": 150,
    "atomic_alu_latency": 25,
    "cacheline_size": 128,
    "pwc_fetch_bytes": 128,
}


def addDemoOptions(parser):
    parser.add_argument(
        "-a", "--app", default=None, help="GPU application to run"
    )
    parser.add_argument(
        "-o", "--opts", default="", help="GPU application arguments"
    )


def runMI300GPUFS(
    cpu_type,
    disk: Optional[AbstractResource] = None,
    kernel: Optional[AbstractResource] = None,
    app: Optional[AbstractResource] = None,
    config_defaults=None,
):
    parser = argparse.ArgumentParser()
    runfs.addRunFSOptions(parser)
    Options.addCommonOptions(parser)
    AmdGPUOptions.addAmdGPUOptions(parser)
    Ruby.define_options(parser)
    GPUTLBOptions.tlb_options(parser)
    addDemoOptions(parser)

    if config_defaults is None:
        config_defaults = SCALED_MI300_DEFAULTS
    parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    demo_runscript = ""

    if disk != None:
        args.disk_image = disk.get_local_path()
    if kernel != None:
        args.kernel = kernel.get_local_path()
    if app != None:
        args.app = app.get_local_path()

    # Create temp script to run application
    if not os.path.isfile(args.app):
        print("Could not find applcation", args.app)
        sys.exit(1)

    # Choose runscript Based on whether any checkpointing args are set
    if args.checkpoint_dir is not None:
        demo_runscript = demo_runscript_with_checkpoint
    else:
        demo_runscript = demo_runscript_without_checkpoint

    with open(os.path.abspath(args.app), "rb") as binfile:
        encodedBin = base64.b64encode(binfile.read()).decode()

    _, tempRunscript = tempfile.mkstemp()
    with open(tempRunscript, "w") as b64file:
        runscriptStr = demo_runscript.format(
            args.app, args.opts, encodedBin, args.opts
        )
        b64file.write(runscriptStr)

    args.script = tempRunscript

    # The MI300 GPU-FS boot flow requires these fixed platform identities.
    args.cpu_type = "X86KvmCPU"
    args.mem_size = "16GiB"
    args.gpu_device = "MI300X"

    # Run gem5
    runfs.runGpuFSSystem(args)


if __name__ == "__m5_main__":
    runMI300GPUFS("X86KvmCPU")
