/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "arch/amdgpu/vega/pagetable_walker.hh"

#include <memory>

#include "arch/amdgpu/vega/faults.hh"
#include "mem/abstract_mem.hh"
#include "mem/packet_access.hh"

namespace gem5
{
namespace VegaISA
{

/*
 * Functional/atomic mode methods
 */
Fault
Walker::startFunctional(Addr base, Addr &addr, unsigned &logBytes,
                        BaseMMU::Mode mode, bool &isSystem)
{
    PageTableEntry pte;
    Addr vaddr = addr;
    Fault fault = startFunctional(base, vaddr, pte, logBytes, mode);
    isSystem = pte.s;
    addr = ((pte.ppn << PageShift) + (vaddr & mask(logBytes)));

    return fault;
}

Fault
Walker::startFunctional(Addr base, Addr vaddr, PageTableEntry &pte,
                        unsigned &logBytes, BaseMMU::Mode mode)
{
    DPRINTF(GPUPTWalker,
            "Vega walker walker: %p funcState: %p "
            "funcState->walker %p\n",
            this, &funcState, funcState.getWalker());
    funcState.initState(mode, base, vaddr, true);
    return funcState.startFunctional(base, vaddr, pte, logBytes);
}

Fault
Walker::WalkerState::startFunctional(Addr base, Addr vaddr,
                                     PageTableEntry &pte, unsigned &logBytes)
{
    Fault fault = NoFault;
    DPRINTF(GPUPTWalker,
            "Vega walker starting with addr: %#lx "
            "logical: %#lx\n",
            vaddr, vaddr >> PageShift);

    assert(!started);
    started = true;

    do {
        DPRINTF(GPUPTWalker, "Sending functional read to %#lx\n",
                read->getAddr());

        auto devmem = walker->system->getDeviceMemory(read);
        assert(devmem);
        devmem->access(read);

        // Extract the requested 8B entry from the fetched block.
        // Copy the full line before rewriting word 0 so
        // pendingLineEntries preserves the original data.
        assert(lineIndex < walker->pwcFetchEntries);
        assert((read->getAddr() & (walker->pwcFetchBytes - 1)) == 0);
        const uint64_t *lineData = read->getConstPtr<uint64_t>();
        pendingLineAddr = read->getAddr();
        pendingLineIndex = lineIndex;
        for (unsigned i = 0; i < walker->pwcFetchEntries; i++)
            pendingLineEntries[i] = letoh(lineData[i]);
        pendingLineValid = true;

        // Rewrite word 0 so stepWalk()'s getLE<uint64_t>() sees the
        // correct entry.
        uint64_t selectedEntry = letoh(lineData[lineIndex]);
        read->setLE<uint64_t>(selectedEntry);

        // Populate pending PWC metadata for deferred insertion in stepWalk()
        Addr originalEntryAddr =
            read->getAddr() + lineIndex * sizeof(uint64_t);
        assert((originalEntryAddr & 0x7) == 0);
        pendingEntryAddr = originalEntryAddr;
        pendingEntry = selectedEntry;
        pendingFromPwc = false;

        fault = stepWalk();
        assert(fault == NoFault || read == NULL);

        state = nextState;
    } while (read);

    logBytes = entry.logBytes;
    pte = entry.pte;

    return fault;
}

/*
 * Timing mode methods
 */
void
Walker::startTiming(PacketPtr pkt, Addr base, Addr vaddr, BaseMMU::Mode mode)
{
    DPRINTF(GPUPTWalker,
            "Vega walker starting with addr: %#lx "
            "logical: %#lx\n",
            vaddr, vaddr >> PageShift);

    WalkerState *newState = new WalkerState(this, pkt);

    newState->initState(mode, base, vaddr);
    currStates.push_back(newState);
    DPRINTF(GPUPTWalker, "There are %ld walker states\n", currStates.size());

    newState->startWalk();
}

void
Walker::WalkerState::initState(BaseMMU::Mode _mode, Addr baseAddr, Addr vaddr,
                               bool is_functional)
{
    DPRINTF(GPUPTWalker, "Walker::WalkerState::initState\n");
    DPRINTF(GPUPTWalker, "Walker::WalkerState::initState %p\n", this);
    DPRINTF(GPUPTWalker, "Walker::WalkerState::initState %d\n", state);
    assert(state == Ready);

    started = false;
    mode = _mode;
    timing = !is_functional;
    enableNX = true;
    dataSize = walker->pwcFetchBytes;
    nextState = PDE2;

    DPRINTF(GPUPTWalker, "Setup walk with base %#lx\n", baseAddr);

    // First level in Vega is PDE2. Calculate the address for that PDE using
    // baseAddr and vaddr.
    state = PDE2;
    Addr logical_addr = vaddr >> PageShift;
    Addr pde2Addr = (((baseAddr >> 6) << 3) + (logical_addr >> 3 * 9)) << 3;
    DPRINTF(GPUPTWalker, "Walk PDE2 address is %#lx\n", pde2Addr);

    // Align to the configured page-walk fetch block boundary.
    const Addr blkSize = walker->pwcFetchBytes;
    assert((pde2Addr & 0x7) == 0);           // 8-byte aligned
    Addr alignedAddr = pde2Addr & ~(blkSize - 1);
    assert((alignedAddr & (blkSize - 1)) == 0);
    unsigned idx = (pde2Addr - alignedAddr) / sizeof(uint64_t);
    assert(idx < walker->pwcFetchEntries);
    lineIndex = idx;
    DPRINTF(GPUPTWalker, "Aligned PDE2 %#lx -> %#lx lineIndex %u\n",
            pde2Addr, alignedAddr, lineIndex);

    // Start populating the VegaTlbEntry response
    entry.vaddr = logical_addr;

    // Prepare the read packet that will be used at each level
    Request::Flags flags = Request::PHYSICAL;

    RequestPtr request = std::make_shared<Request>(
        alignedAddr, dataSize, flags, walker->deviceRequestorId);

    read = new Packet(request, MemCmd::ReadReq);
    read->allocate();
}

void
Walker::WalkerState::startWalk()
{
    if (!started) {
        // Read the first PDE to begin
        DPRINTF(GPUPTWalker, "Sending timing read to %#lx\n", read->getAddr());

        started = true;
        sendPackets();
    } else {
        // This is mostly the same as stepWalk except we update the state and
        // send the new timing read request.
        timingFault = stepWalk();
        assert(timingFault == NoFault || read == NULL);

        state = nextState;

        if (read) {
            DPRINTF(GPUPTWalker, "Sending timing read to %#lx\n",
                    read->getAddr());
            sendPackets();
        } else {
            // Set physical page address in entry
            entry.paddr = entry.pte.ppn << PageShift;
            entry.paddr += entry.vaddr & mask(entry.logBytes);

            // Send translation return event. The TLB allocates the returned entry in the normal miss-return path.
            walker->walkerResponse(this, entry, tlbPkt);
        }
    }
}

Fault
Walker::WalkerState::stepWalk()
{
    assert(state != Ready && state != Waiting && read);
    Fault fault = NoFault;
    PageTableEntry pte = read->getLE<uint64_t>();

    bool uncacheable = !pte.c;
    Addr nextRead = 0;
    bool doEndWalk = false;

    walkStateMachine(pte, nextRead, doEndWalk, fault);

    // Deferred PWC insertion: insert the previous response into the original
    // PWC only if it was not final (doEndWalk means it IS the final entry)
    // and not already from the PWC.
    if (!doEndWalk && !pendingFromPwc && walker->enable_pwc
        && walker->pwc.findEntry(pendingEntryAddr) == nullptr) {
        walker->pwc.insert(pendingEntryAddr, pendingEntry);
        walker->stats.pwcInsertions++;
    }

    // Deferred PWC2 neighbour insertion: on final entries from real memory
    // responses, insert the other entries from the fetched line into PWC2.
    if (doEndWalk && !pendingFromPwc && pendingLineValid
        && walker->enable_pwc) {
        DPRINTF(GPUPTWalker, "Skipping PWC insert for final entry %#lx, "
                "inserting neighbours into PWC2\n", pendingEntryAddr);
        for (unsigned i = 0; i < walker->pwcFetchEntries; i++) {
            if (i == pendingLineIndex)
                continue;

            Addr neighbourAddr =
                pendingLineAddr + i * sizeof(uint64_t);
            PageTableEntry neighbourPte = pendingLineEntries[i];

            // Skip invalid/non-present page-table entries
            if (!neighbourPte.v) {
                walker->stats.pwc2InvalidNeighboursSkipped++;
                continue;
            }

            if (walker->neighbourPwc.findEntry(neighbourAddr) == nullptr) {
                walker->neighbourPwc.insert(neighbourAddr, neighbourPte);
                walker->stats.pwc2Insertions++;
            }
        }
    }

    if (doEndWalk) {
        DPRINTF(GPUPTWalker, "ending walk\n");
        endWalk();
    } else {
        PacketPtr oldRead = read;

        // Align the next read to the configured page-walk fetch block.
        const Addr blkSize = walker->pwcFetchBytes;
        assert((nextRead & 0x7) == 0);           // 8-byte aligned
        Addr alignedAddr = nextRead & ~(blkSize - 1);
        assert((alignedAddr & (blkSize - 1)) == 0);
        unsigned idx = (nextRead - alignedAddr) / sizeof(uint64_t);
        assert(idx < walker->pwcFetchEntries);
        lineIndex = idx;
        DPRINTF(GPUPTWalker, "Aligned next read %#lx -> %#lx lineIndex %u\n",
                nextRead, alignedAddr, lineIndex);
        // If we didn't return, we're setting up another read.
        Request::Flags flags = oldRead->req->getFlags();
        flags.set(Request::UNCACHEABLE, uncacheable);
        RequestPtr request = std::make_shared<Request>(
            alignedAddr, oldRead->getSize(), flags,
            walker->deviceRequestorId);

        read = new Packet(request, MemCmd::ReadReq);
        read->allocate();

        delete oldRead;
    }

    return fault;
}

void
Walker::WalkerState::walkStateMachine(PageTableEntry &pte, Addr &nextRead,
                                      bool &doEndWalk, Fault &fault)
{
    Addr vaddr = entry.vaddr;
    bool badNX = pte.x && mode == BaseMMU::Execute && enableNX;
    Addr part1 = 0;
    Addr part2 = 0;
    PageDirectoryEntry pde = static_cast<PageDirectoryEntry>(pte);

    // Block fragment size can change the size of the pages pointed to while
    // moving to the next PDE. A value of 0 implies native page size. A
    // non-zero value implies the next leaf in the page table is a PTE unless
    // the F bit is set. If we see a non-zero value, set it here and print
    // for debugging.
    if (pde.blockFragmentSize) {
        DPRINTF(GPUPTWalker,
                "blockFragmentSize: %d, pde: %#016lx, state: %d\n",
                pde.blockFragmentSize, pde, state);
        blockFragmentSize = pde.blockFragmentSize;

        // At this time, only a value of 9 is used in the driver:
        // https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/
        //     amd/amdgpu/gmc_v9_0.c#L1165
        assert(pde.blockFragmentSize == 9);
    }

    switch (state) {
        case PDE2:
            if (pde.p) {
                DPRINTF(GPUPTWalker, "Treating PDE2 as PTE: %#016x frag: %d\n",
                        (uint64_t)pte, pte.fragment);
                entry.pte = pte;
                int fragment = pte.fragment;
                entry.logBytes = PageShift + std::min(3 * 9, fragment);
                entry.vaddr <<= PageShift;
                entry.vaddr = entry.vaddr & ~mask(entry.logBytes);
                doEndWalk = true;
            }

            // Read the pde1Addr
            part1 = ((((uint64_t)pte) >> 6) << 3);
            part2 = offsetFunc(vaddr, 3 * 9, 2 * 9);
            nextRead = ((part1 + part2) << 3) & mask(48);
            DPRINTF(GPUPTWalker,
                    "Got PDE2 entry %#016x. write:%s->%#016x va:%#016x\n",
                    (uint64_t)pte, pte.w == 0 ? "yes" : "no", nextRead, vaddr);
            nextState = PDE1;
            break;
        case PDE1:
            if (pde.p) {
                DPRINTF(GPUPTWalker, "Treating PDE1 as PTE: %#016x frag: %d\n",
                        (uint64_t)pte, pte.fragment);
                entry.pte = pte;
                int fragment = pte.fragment;
                entry.logBytes = PageShift + std::min(2 * 9, fragment);
                entry.vaddr <<= PageShift;
                entry.vaddr = entry.vaddr & ~mask(entry.logBytes);
                doEndWalk = true;
            }

            // Read the pde0Addr
            part1 = ((((uint64_t)pte) >> 6) << 3);
            part2 = offsetFunc(vaddr, 2 * 9, 9);
            nextRead = ((part1 + part2) << 3) & mask(48);
            DPRINTF(GPUPTWalker,
                    "Got PDE1 entry %#016x. write:%s->%#016x va: %#016x\n",
                    (uint64_t)pte, pte.w == 0 ? "yes" : "no", nextRead, vaddr);
            nextState = PDE0;
            break;
        case PDE0:
            if (pde.p || (blockFragmentSize && !pte.f)) {
                DPRINTF(GPUPTWalker, "Treating PDE0 as PTE: %#016x frag: %d\n",
                        (uint64_t)pte, pte.fragment);
                entry.pte = pte;
                int fragment = pte.fragment;
                entry.logBytes = PageShift + std::min(9, fragment);
                entry.vaddr <<= PageShift;
                entry.vaddr = entry.vaddr & ~mask(entry.logBytes);
                doEndWalk = true;
            }
            // Read the PteAddr
            part1 = ((((uint64_t)pte) >> 6) << 3);
            if (pte.f) {
                // For F bit we want to use the blockFragmentSize in the
                // previous PDE and the blockFragmentSize in this PTE for
                // offset function.
                part2 = offsetFunc(vaddr, blockFragmentSize,
                                   pde.blockFragmentSize);
            } else {
                part2 = offsetFunc(vaddr, 9, 0);
            }
            nextRead = ((part1 + part2) << 3) & mask(48);
            DPRINTF(GPUPTWalker,
                    "Got PDE0 entry %#016x. write:%s->%#016x va:%#016x\n",
                    (uint64_t)pte, pte.w == 0 ? "yes" : "no", nextRead, vaddr);
            nextState = PTE;
            break;
        case PTE:
            DPRINTF(GPUPTWalker, " PTE entry %#016x. write: %s va: %#016x\n",
                    (uint64_t)pte, pte.w == 0 ? "yes" : "no", vaddr);
            entry.pte = pte;
            entry.logBytes = PageShift;
            entry.vaddr <<= PageShift;
            entry.vaddr = entry.vaddr & ~mask(entry.logBytes);
            doEndWalk = true;
            break;
        default:
            panic("Unknown page table walker state %d!\n");
    }

    if (badNX || !pte.v) {
        doEndWalk = true;
        fault = pageFault(pte.v);
        nextState = state;
    }
}

void
Walker::WalkerState::endWalk()
{
    nextState = Ready;
    delete read;
    read = NULL;
    walker->currStates.remove(this);
}

/**
 * Port related methods
 */
void
Walker::WalkerState::sendPackets()
{
    // If we're already waiting for the port to become available, just return.
    if (retrying) {
        return;
    }
    [[maybe_unused]] auto addr = read->getAddr();
    if (!walker->sendTiming(this, read)) {
        DPRINTF(GPUPTWalker, "Timing request for %#lx failed\n", addr);

        retrying = true;
    }
    // No lines after sendTiming because the state might be freed
    // else {
    //     DPRINTF(GPUPTWalker, "Timing request for %#lx successful\n",
    //             addr);
    // }
}

bool
Walker::sendTiming(WalkerState *sending_walker, PacketPtr pkt)
{
    auto walker_state = new WalkerSenderState(sending_walker,
                                              sending_walker->lineIndex);
    pkt->pushSenderState(walker_state);

    // Reconstruct original 8B entry address from aligned fetch block address
    Addr originalEntryAddr =
        pkt->getAddr() + sending_walker->lineIndex * sizeof(uint64_t);
    assert(sending_walker->lineIndex < pwcFetchEntries);
    assert((originalEntryAddr & 0x7) == 0);

    if (enable_pwc) {
        // Check original PWC first (non-final walk entries)
        stats.pwcAccesses++;
        PWCEntry *entry = pwc.findEntry(originalEntryAddr);
        if (entry != nullptr) {
            stats.pwcHits++;
            DPRINTF(GPUPTWalker,
                    "PTE found in PWC, skipping timing request.");
            pkt->setLE<uint64_t>(entry->pteEntry);
            walker_state->fromPwc = true;

            recvTimingResp(pkt);

            return true;
        }
        stats.pwcMisses++;

        // Check PWC2 second (neighbouring final-level entries)
        stats.pwc2Accesses++;
        PWCEntry *entry2 = neighbourPwc.findEntry(originalEntryAddr);
        if (entry2 != nullptr) {
            stats.pwc2Hits++;
            DPRINTF(GPUPTWalker,
                    "PTE found in PWC2, skipping timing request.");
            pkt->setLE<uint64_t>(entry2->pteEntry);
            walker_state->fromPwc = true;

            recvTimingResp(pkt);

            return true;
        }
        stats.pwc2Misses++;
    }

    if (port.sendTimingReq(pkt)) {
        DPRINTF(GPUPTWalker, "Sending timing read to %#lx from walker %p\n",
                pkt->getAddr(), sending_walker);

        return true;
    } else {
        (void)pkt->popSenderState();
        delete walker_state;
    }

    return false;
}

bool
Walker::WalkerPort::recvTimingResp(PacketPtr pkt)
{
    walker->recvTimingResp(pkt);

    return true;
}

void
Walker::recvTimingResp(PacketPtr pkt)
{
    WalkerSenderState *senderState =
        safe_cast<WalkerSenderState *>(pkt->popSenderState());

    assert(senderState->lineIndex < pwcFetchEntries);
    // Reconstruct original 8B entry address from aligned fetch block address
    Addr originalEntryAddr =
        pkt->getAddr() + senderState->lineIndex * sizeof(uint64_t);
    assert((originalEntryAddr & 0x7) == 0);

    WalkerState *ws = senderState->senderWalk;
    uint64_t selectedEntry;

    if (senderState->fromPwc) {
        // PWC hit: word 0 already contains the correct entry
        selectedEntry = pkt->getLE<uint64_t>();
        ws->pendingLineValid = false;
    } else {
        // Real memory response. The timing read port can return stale data:
        // page tables are written functionally to device memory (by the
        // driver/CP) and are NOT coherent with the GPU timing cache hierarchy,
        // so the timing read may observe an old/invalid PDE (e.g. ppn=0, v=0)
        // even though device memory holds the correct entry. Re-read the
        // configured page-table fetch block directly from device memory (as
        // the functional walk does) so the walk observes the correct PTEs.
        // The timing request is still used to model walk latency.
        std::vector<uint64_t> devline(pwcFetchEntries, 0);
        RequestPtr freq = std::make_shared<Request>(
            pkt->getAddr(), pwcFetchBytes, Request::PHYSICAL,
            deviceRequestorId);
        PacketPtr fpkt = new Packet(freq, MemCmd::ReadReq);
        fpkt->dataStatic(reinterpret_cast<uint8_t *>(devline.data()));
        auto *devmem = system->getDeviceMemory(fpkt);
        const uint64_t *lineData;
        if (devmem) {
            devmem->access(fpkt);
            lineData = devline.data();
        } else {
            lineData = pkt->getConstPtr<uint64_t>();
        }

        ws->pendingLineAddr = pkt->getAddr();
        ws->pendingLineIndex = senderState->lineIndex;
        assert((ws->pendingLineAddr & (pwcFetchBytes - 1)) == 0);
        for (unsigned i = 0; i < pwcFetchEntries; i++)
            ws->pendingLineEntries[i] = letoh(lineData[i]);
        ws->pendingLineValid = true;

        // Extract from the fetched block and rewrite word 0
        // so stepWalk()'s getLE<uint64_t>() sees the correct entry.
        selectedEntry = letoh(lineData[senderState->lineIndex]);
        pkt->setLE<uint64_t>(selectedEntry);

        delete fpkt;
    }

    DPRINTF(GPUPTWalker, "Got response for %#lx (entry %#lx) from walker %p "
            "lineIndex %u fromPwc %d -- %#lx\n",
            pkt->getAddr(), originalEntryAddr, ws,
            senderState->lineIndex, senderState->fromPwc, selectedEntry);

    // Store response metadata in WalkerState for deferred PWC insertion.
    // stepWalk() will insert into the original PWC only for non-final entries.
    ws->pendingEntryAddr = originalEntryAddr;
    ws->pendingEntry = selectedEntry;
    ws->pendingFromPwc = senderState->fromPwc;

    ws->startWalk();

    delete senderState;
}

void
Walker::invalidatePWC()
{
    for (auto &i : pwc) {
        if (i.valid) {
            pwc.invalidate(&i);
            stats.pwcInvalidations++;
        }
    }
    for (auto &i : neighbourPwc) {
        if (i.valid) {
            neighbourPwc.invalidate(&i);
            stats.pwc2Invalidations++;
        }
    }
}

void
Walker::WalkerPort::recvReqRetry()
{
    walker->recvReqRetry();
}

void
Walker::recvReqRetry()
{
    std::list<WalkerState *>::iterator iter;
    for (iter = currStates.begin(); iter != currStates.end();) {
        WalkerState *walkerState = *(iter);
        // retry() might finish the walk, thus the current iterator
        // might be invalid after retry(). Need to update iter now
        iter++;
        if (walkerState->isRetrying()) {
            walkerState->retry();
        }
    }
}

void
Walker::walkerResponse(WalkerState *state, VegaTlbEntry &entry, PacketPtr pkt)
{
    // Propagate whether the final PTE came from the PWC/PWC2 (a PWC hit) or
    // from memory (a real miss) so the TLB can update its line predictor.
    tlb->walkerResponse(entry, pkt, state->pendingFromPwc);

    delete state;
}

/*
 *  Helper methods
 */
bool
Walker::WalkerState::isRetrying()
{
    return retrying;
}

void
Walker::WalkerState::retry()
{
    retrying = false;
    sendPackets();
}

Fault
Walker::WalkerState::pageFault(bool present)
{
    DPRINTF(GPUPTWalker, "Raising page fault.\n");
    ExceptionCode code;
    if (mode == BaseMMU::Read) {
        code = ExceptionCode::LOAD_PAGE;
    } else if (mode == BaseMMU::Write) {
        code = ExceptionCode::STORE_PAGE;
    } else {
        code = ExceptionCode::INST_PAGE;
    }
    if (mode == BaseMMU::Execute && !enableNX) {
        mode = BaseMMU::Read;
    }
    return std::make_shared<PageFault>(entry.vaddr, code, present, mode, true);
}

uint64_t
Walker::WalkerState::offsetFunc(Addr logicalAddr, int top, int lsb)
{
    assert(top < 32);
    assert(lsb < 32);
    return ((logicalAddr & ((1 << top) - 1)) >> lsb);
}

/**
 * Stats
 */
Walker::WalkerStats::WalkerStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(pwcAccesses, "Number of accesses to the original PWC"),
      ADD_STAT(pwcHits, "Number of hits in the original PWC"),
      ADD_STAT(pwcMisses, "Number of misses in the original PWC"),
      ADD_STAT(pwcInsertions, "Number of insertions into the original PWC"),
      ADD_STAT(pwcInvalidations,
               "Number of invalidations in the original PWC"),
      ADD_STAT(pwc2Accesses, "Number of accesses to the neighbour PWC"),
      ADD_STAT(pwc2Hits, "Number of hits in the neighbour PWC"),
      ADD_STAT(pwc2Misses, "Number of misses in the neighbour PWC"),
      ADD_STAT(pwc2Insertions,
               "Number of insertions into the neighbour PWC"),
      ADD_STAT(pwc2Invalidations,
               "Number of invalidations in the neighbour PWC"),
      ADD_STAT(pwc2InvalidNeighboursSkipped,
               "Number of invalid neighbour entries skipped during "
               "PWC2 insertion")
{
}


/**
 * gem5 methods
 */
Port &
Walker::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "port") {
        return port;
    } else {
        return ClockedObject::getPort(if_name, idx);
    }
}

} // namespace VegaISA
} // namespace gem5
