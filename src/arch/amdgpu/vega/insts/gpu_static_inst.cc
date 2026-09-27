/*
 * Copyright (c) 2015-2021 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "arch/amdgpu/vega/insts/gpu_static_inst.hh"

#include "arch/amdgpu/vega/gpu_decoder.hh"
#include "arch/amdgpu/vega/insts/instructions.hh"
#include "debug/GPUExec.hh"
#include "gpu-compute/shader.hh"

namespace gem5
{

namespace VegaISA
{
VEGAGPUStaticInst::VEGAGPUStaticInst(const std::string &opcode)
    : GPUStaticInst(opcode), _srcLiteral(0)
{}

VEGAGPUStaticInst::~VEGAGPUStaticInst()
{}

void
VEGAGPUStaticInst::panicUnimplemented() const
{
    fatal("Encountered unimplemented VEGA instruction: %s\n", _opcode);
}

void
VEGAGPUStaticInst::setCachePolicyBits(bool sc0, bool sc1, bool nt)
{
    // gfx942 (CDNA3) cache-policy bits, per the CDNA3 ISA guide's memory
    // scope controls. For loads and stores SC1:SC0 is the scope: wavefront
    // (0), workgroup (1), device (2) or system (3). A workgroup runs on a
    // single CU, so only device and system scope bypass the CU (L1) cache,
    // and only system scope also bypasses the L2. For atomics SC0 requests
    // the pre-op value (the subclass sets AtomicReturn from it) and SC1
    // alone selects system scope. NT is a non-temporal (streaming) hint
    // that is passed to the memory system for the caches to act on.
    if (nt) {
        setFlag(NonTemporal);
    }

    if (isAtomic()) {
        if (sc1) {
            setFlag(GloballyCoherent);
            setFlag(SystemCoherent);
        }
    } else {
        if (sc1) {
            setFlag(GloballyCoherent);
        }

        if (sc0 && sc1) {
            setFlag(SystemCoherent);
        }
    }
}
} // namespace VegaISA
} // namespace gem5
