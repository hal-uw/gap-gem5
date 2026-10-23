/*
 * Copyright (c) 1999-2008 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __MEM_RUBY_SLICC_INTERFACE_RUBYSLICC_COMPONENTMAPPINGS_HH__
#define __MEM_RUBY_SLICC_INTERFACE_RUBYSLICC_COMPONENTMAPPINGS_HH__

#include "mem/ruby/common/Address.hh"
#include "mem/ruby/common/MachineID.hh"
#include "mem/ruby/common/NetDest.hh"
#include "mem/ruby/protocol/MachineType.hh"

namespace gem5
{

namespace ruby
{

// XOR-fold every address bit at and above low_bit, num_bits at a time.
// Unlike a plain bit select, strides that are multiples of the interleave
// size still spread across all of the 1 << num_bits targets.
inline int
hashSelect(Addr addr, int low_bit, int num_bits)
{
    const Addr mask = (Addr(1) << num_bits) - 1;
    Addr sel = 0;
    for (Addr x = addr >> low_bit; x != 0; x >>= num_bits)
        sel ^= x & mask;
    return sel;
}

inline MachineID
mapAddressToRange(Addr addr, MachineType type, int low_bit,
                  int num_bits, int cluster_id = 0)
{
    MachineID mach = {type, 0};
    if (num_bits == 0)
        mach.num = cluster_id;
    else
        mach.num = bitSelect(addr, low_bit, low_bit + num_bits - 1)
            + (1 << num_bits) * cluster_id;
    return mach;
}

inline NodeID
machineIDToNodeID(MachineID machID)
{
    return machID.num;
}

inline MachineType
machineIDToMachineType(MachineID machID)
{
    return machID.type;
}

inline MachineID
createMachineID(MachineType type, NodeID id)
{
    MachineID mach = {type, id};
    return mach;
}

inline MachineID
MachineTypeAndNodeIDToMachineID(MachineType type, NodeID node)
{
    MachineID mach = {type, node};
    return mach;
}

} // namespace ruby
} // namespace gem5

#endif  // __MEM_RUBY_SLICC_INTERFACE_COMPONENTMAPPINGS_HH__
