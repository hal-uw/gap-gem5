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

#ifndef __DEV_AMDGPU_PAGETABLE_WALKER_HH__
#define __DEV_AMDGPU_PAGETABLE_WALKER_HH__

#include <vector>

#include "arch/amdgpu/vega/page_walk_cache.hh"
#include "arch/amdgpu/vega/pagetable.hh"
#include "arch/amdgpu/vega/tlb.hh"
#include "base/logging.hh"
#include "base/statistics.hh"
#include "base/types.hh"
#include "debug/GPUPTWalker.hh"
#include "mem/packet.hh"
#include "params/VegaPagetableWalker.hh"
#include "sim/clocked_object.hh"
#include "sim/system.hh"

namespace gem5
{

class ThreadContext;

namespace VegaISA
{

class Walker : public ClockedObject
{
  protected:
    // PWC for non-final page-table walk levels (Global, Upper, Middle).
    PageWalkCache pwc;

    // PWC for neighbouring final-level entries from the same page-table
    // memory fetch. The fetch width is controlled by pwcFetchBytes.
    PageWalkCache neighbourPwc;

    // Port for accessing memory
    class WalkerPort : public RequestPort
    {
      public:
        WalkerPort(const std::string &_name, Walker *_walker)
            : RequestPort(_name), walker(_walker)
        {}

      protected:
        Walker *walker;

        bool recvTimingResp(PacketPtr pkt);
        void recvReqRetry();
    };

    friend class WalkerPort;
    WalkerPort port;

    static constexpr Addr MaxPwcFetchBytes = 1024;
    static constexpr unsigned MaxPwcFetchEntries =
        MaxPwcFetchBytes / sizeof(uint64_t);

    // Width of page-table memory fetches used by the walker/PWC2 model.
    Addr pwcFetchBytes;
    unsigned pwcFetchEntries;

    // State to track each walk of the page table
    class WalkerState
    {
        friend class Walker;

      private:
        enum State
        {
            Ready,
            Waiting,
            PDE2,
            PDE1,
            PDE0,
            PTE
        };

      protected:
        Walker *walker;
        State state;
        State nextState;
        int dataSize;
        bool enableNX;
        VegaTlbEntry entry;
        PacketPtr read;
        Fault timingFault;
        BaseMMU::Mode mode;
        bool retrying;
        bool started;
        bool timing;
        PacketPtr tlbPkt;
        int blockFragmentSize;
        // Index of the requested 8B entry within the current fetch block.
        unsigned lineIndex;

        // Response metadata for deferred PWC insertion (set by
        // recvTimingResp, consumed by stepWalk after walkStateMachine).
        Addr pendingEntryAddr;
        uint64_t pendingEntry;
        bool pendingFromPwc;

        // Full fetched line data for deferred PWC2 neighbour insertion.
        // Populated from real memory responses; invalid for PWC hits.
        Addr pendingLineAddr;
        uint64_t pendingLineEntries[MaxPwcFetchEntries];
        unsigned pendingLineIndex;
        bool pendingLineValid;

      public:
        WalkerState(Walker *_walker, PacketPtr pkt, bool is_functional = false)
            : walker(_walker), state(Ready), nextState(Ready), dataSize(0),
              enableNX(true), retrying(false), started(false), tlbPkt(pkt),
              blockFragmentSize(0), lineIndex(0),
              pendingEntryAddr(0), pendingEntry(0), pendingFromPwc(false),
              pendingLineAddr(0), pendingLineIndex(0),
              pendingLineValid(false)
        {
            DPRINTF(GPUPTWalker, "Walker::WalkerState %p %p %d\n", this,
                    walker, state);
        }

        void initState(BaseMMU::Mode _mode, Addr baseAddr, Addr vaddr,
                       bool is_functional = false);
        void startWalk();
        Fault startFunctional(Addr base, Addr vaddr, PageTableEntry &pte,
                              unsigned &logBytes);

        bool isRetrying();
        void retry();
        std::string
        name() const
        {
            return walker->name();
        }
        Walker *
        getWalker() const
        {
            return walker;
        }

      private:
        Fault stepWalk();
        void stepTimingWalk();
        void walkStateMachine(PageTableEntry &pte, Addr &nextRead,
                              bool &doEndWalk, Fault &fault);
        void sendPackets();
        void endWalk();
        Fault pageFault(bool present);
        uint64_t offsetFunc(Addr logicalAddr, int top, int lsb);
    };

    friend class WalkerState;


    // State for timing and atomic accesses (need multiple per walker in
    // the case of multiple outstanding requests in timing mode)
    std::list<WalkerState *> currStates;
    // State for functional accesses (only need one of these per walker)
    WalkerState funcState;

    struct WalkerSenderState : public Packet::SenderState
    {
        WalkerState * senderWalk;
        // Index of the requested 8B entry within the fetch block.
        unsigned lineIndex;
        // True when the response comes from a PWC hit, not real memory
        bool fromPwc;
        WalkerSenderState(WalkerState * _senderWalk, unsigned _lineIndex = 0,
                          bool _fromPwc = false)
            : senderWalk(_senderWalk), lineIndex(_lineIndex),
              fromPwc(_fromPwc) {}
    };

  public:
    // Kick off the state machine.
    void startTiming(PacketPtr pkt, Addr base, Addr vaddr, BaseMMU::Mode mode);
    Fault startFunctional(Addr base, Addr vaddr, PageTableEntry &pte,
                          unsigned &logBytes, BaseMMU::Mode mode);
    Fault startFunctional(Addr base, Addr &addr, unsigned &logBytes,
                          BaseMMU::Mode mode, bool &isSystem);

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;

    Addr
    getBaseAddr() const
    {
        return baseAddr;
    }
    void
    setBaseAddr(Addr ta)
    {
        baseAddr = ta;
    }

    void
    setDevRequestor(RequestorID mid)
    {
        deviceRequestorId = mid;
    }
    RequestorID
    getDevRequestor() const
    {
        return deviceRequestorId;
    }

    void invalidatePWC();

  protected:
    bool enable_pwc;
    // The TLB we're supposed to load.
    GpuTLB *tlb;
    RequestorID requestorId;

    // Base address set by MAP_PROCESS packet
    Addr baseAddr;
    RequestorID deviceRequestorId;

    // Functions for dealing with packets.
    void recvTimingResp(PacketPtr pkt);
    void recvReqRetry();
    bool sendTiming(WalkerState *sendingState, PacketPtr pkt);

    void walkerResponse(WalkerState *state, VegaTlbEntry &entry,
                        PacketPtr pkt);

    // System pointer for functional accesses
    System *system;

    struct WalkerStats : public statistics::Group
    {
        WalkerStats(statistics::Group *parent);

        // Original PWC stats (non-final walk levels)
        statistics::Scalar pwcAccesses;
        statistics::Scalar pwcHits;
        statistics::Scalar pwcMisses;
        statistics::Scalar pwcInsertions;
        statistics::Scalar pwcInvalidations;

        // Neighbour PWC stats (final-level neighbour entries)
        statistics::Scalar pwc2Accesses;
        statistics::Scalar pwc2Hits;
        statistics::Scalar pwc2Misses;
        statistics::Scalar pwc2Insertions;
        statistics::Scalar pwc2Invalidations;
        statistics::Scalar pwc2InvalidNeighboursSkipped;
    } stats;

  public:
    void
    setTLB(GpuTLB *_tlb)
    {
        assert(tlb == nullptr); // only set it once
        tlb = _tlb;
    }

    Walker(const VegaPagetableWalkerParams &p)
      : ClockedObject(p),
        pwc(name()+".pwc", p.page_walk_cache_entries,
            p.page_walk_cache_entries, p.pwc_replacement_policy,
            p.pwc_indexing_policy),
        neighbourPwc(name()+".neighbourPwc", p.neighbour_pwc_entries,
            p.neighbour_pwc_entries, p.neighbour_pwc_replacement_policy,
            p.neighbour_pwc_indexing_policy),
        port(name() + ".port", this),
        pwcFetchBytes(p.pwc_fetch_bytes),
        pwcFetchEntries(p.pwc_fetch_bytes / sizeof(uint64_t)),
        funcState(this, nullptr, true),
        enable_pwc(p.enable_pwc),
	tlb(nullptr),
        requestorId(p.system->getRequestorId(this)),
        deviceRequestorId(999),
        system(p.system),
        stats(this)
    {
        fatal_if(pwcFetchBytes < sizeof(uint64_t),
                 "VegaPagetableWalker pwc_fetch_bytes must be at least 8 "
                 "bytes");
        fatal_if(pwcFetchBytes % sizeof(uint64_t) != 0,
                 "VegaPagetableWalker pwc_fetch_bytes must be divisible "
                 "by 8");
        fatal_if((pwcFetchBytes & (pwcFetchBytes - 1)) != 0,
                 "VegaPagetableWalker pwc_fetch_bytes must be a power "
                 "of two");
        fatal_if(pwcFetchBytes > MaxPwcFetchBytes,
                 "VegaPagetableWalker pwc_fetch_bytes exceeds supported "
                 "maximum of 1024 bytes");
        fatal_if(pwcFetchBytes > system->cacheLineSize(),
                 "VegaPagetableWalker pwc_fetch_bytes (%lu) exceeds the "
                 "system cache-line size (%lu)", pwcFetchBytes,
                 system->cacheLineSize());
        fatal_if(pwcFetchEntries > MaxPwcFetchEntries,
                 "VegaPagetableWalker pwc_fetch_entries exceeds supported "
                 "maximum");

        DPRINTF(GPUPTWalker, "Walker::Walker %p fetchBytes %lu "
                "fetchEntries %u\n", this, pwcFetchBytes, pwcFetchEntries);
    }
};

} // namespace VegaISA
} // namespace gem5

#endif // __DEV_AMDGPU_PAGETABLE_WALKER_HH__
