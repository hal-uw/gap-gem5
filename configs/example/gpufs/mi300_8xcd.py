# Copyright (c) 2026 Basem Mohammed
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Simplified eight-XCD MI300X-oriented (gfx942) configuration.

This preset models the complete 304-CU and 32-MiB aggregate L2 organization,
but retains the base configuration's simulation-sized memory capacity. Each
XCD owns 16 distinct 256-KiB TCC partitions. A global crossbar connects all
128 non-overlapping, address-selected TCC partitions to one logical L3. The
model intentionally omits XCD-local links and local-versus-remote latency
differences.
"""

import mi300


FULL_MI300_8XCD_DEFAULTS = {
    **mi300.SCALED_MI300_DEFAULTS,
    # Eight XCDs with 38 active CUs each.
    "num_compute_units": 304,
    "num_xcds": 8,
    # Sixteen 256 KiB TCC partitions per XCD (32 MiB aggregate).
    "num_tccs": 128,
    "tcc_size": "32MiB",
    # One logical shared Infinity Cache/L3.
    "use_gpu_l3": True,
    "num_l3caches": 1,
    "l3_size": "256MiB",
    "l3_num_banks": 16,
}


if __name__ == "__m5_main__":
    mi300.runMI300GPUFS(
        "X86KvmCPU", config_defaults=FULL_MI300_8XCD_DEFAULTS
    )
