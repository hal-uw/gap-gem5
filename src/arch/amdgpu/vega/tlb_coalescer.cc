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

#include "arch/amdgpu/vega/tlb_coalescer.hh"

#include <algorithm>
#include <cstring>

#include "arch/amdgpu/common/gpu_translation_state.hh"
#include "arch/amdgpu/vega/pagetable.hh"
#include "arch/generic/mmu.hh"
#include "base/logging.hh"
#include "debug/GPUTLB.hh"
#include "sim/process.hh"

namespace gem5
{

VegaTLBCoalescer::VegaTLBCoalescer(const VegaTLBCoalescerParams &p)
    : ClockedObject(p),
      TLBProbesPerCycle(p.probesPerCycle),
      coalescingWindow(p.coalescingWindow),
      disableCoalescing(p.disableCoalescing),
      probeTLBEvent([this] { processProbeTLBEvent(); }, "Probe the TLB below",
                    false, Event::CPU_Tick_Pri),
      cleanupEvent([this] { processCleanupEvent(); },
                   "Cleanup issuedTranslationsTable hashmap", false,
                   Event::Maximum_Pri),
      // Start the size predictor saturated toward 2 MiB (MSB clear), matching
      // the previous hardcoded default_pgSize speculation.
      sizePredictor(SizePredBits, 0),
      downstreamTLB(p.downstream_tlb),
      tlb_level(p.tlb_level),
      maxDownstream(p.maxDownstream),
      numDownstream(0)
{
    // create the response ports based on the number of connected ports
    for (size_t i = 0; i < p.port_cpu_side_ports_connection_count; ++i) {
        cpuSidePort.push_back(
            new CpuSidePort(csprintf("%s-port%d", name(), i), this, i));
    }

    // create the request ports based on the number of connected ports
    for (size_t i = 0; i < p.port_mem_side_ports_connection_count; ++i) {
        memSidePort.push_back(
            new MemSidePort(csprintf("%s-port%d", name(), i), this, i));
    }

    default_pgSize = p.default_pgSize;
    potentialPagesize.insert(default_pgSize);
    // Always consider the 4 KiB boundary as well. The issue-side reissue path
    // keys mispredicted requests in issuedTranslationsTable at 4 KiB, so both
    // the outstanding-page block check and updatePhysAddresses must probe the
    // 4 KiB boundary even before any 4 KiB translation has returned.
    potentialPagesize.insert(VegaISA::PageBytes);
}

void
VegaTLBCoalescer::updateSizePredictor(Addr returned_pgsize)
{
    // 2 MiB (or larger) return -> bias toward 2 MiB (decrement);
    // anything smaller -> bias toward 4 KiB (increment).
    if (returned_pgsize >= default_pgSize) {
        sizePredictor--;
    } else {
        sizePredictor++;
    }
}

Addr
VegaTLBCoalescer::predictedPageSize() const
{
    // MSB set -> predict 4 KiB, otherwise the large (default) page size.
    const uint8_t msb = 1u << (SizePredBits - 1);
    return (static_cast<uint8_t>(sizePredictor) & msb) ? VegaISA::PageBytes
                                                       : default_pgSize;
}

Addr
VegaTLBCoalescer::newGroupPageSize()
{
    Addr pg_size = predictedPageSize();
    if (lineCoalesceEnabled()) {
        pg_size *= LineGroupPages;
        // The line granule becomes a key in issuedTranslationsTable, so the
        // return-side finder and the outstanding-page block check must probe
        // it too.
        potentialPagesize.insert(pg_size);
    }
    return pg_size;
}

size_t
VegaTLBCoalescer::coalescerFIFOEntries() const
{
    size_t entries = 0;
    for (const auto &tick_entry : coalescerFIFO) {
        entries += tick_entry.second.size();
    }
    return entries;
}

size_t
VegaTLBCoalescer::issuedTranslationPackets() const
{
    size_t packets = 0;
    for (const auto &entry : issuedTranslationsTable) {
        packets += entry.second.size();
    }
    return packets;
}

void
VegaTLBCoalescer::updateOccupancyStats()
{
    const size_t fifo_entries = coalescerFIFOEntries();
    if (fifo_entries > fifoMaxEntries.value()) {
        fifoMaxEntries = fifo_entries;
    }

    const size_t issued_entries = issuedTranslationsTable.size();
    if (issued_entries > issuedTranslationsMax.value()) {
        issuedTranslationsMax = issued_entries;
    }

    const size_t issued_packets = issuedTranslationPackets();
    if (issued_packets > issuedTranslationPacketsMax.value()) {
        issuedTranslationPacketsMax = issued_packets;
    }
}

Port &
VegaTLBCoalescer::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cpu_side_ports") {
        if (idx >= static_cast<PortID>(cpuSidePort.size())) {
            panic("VegaTLBCoalescer::getPort: unknown index %d\n", idx);
        }

        return *cpuSidePort[idx];
    } else if (if_name == "mem_side_ports") {
        if (idx >= static_cast<PortID>(memSidePort.size())) {
            panic("VegaTLBCoalescer::getPort: unknown index %d\n", idx);
        }

        return *memSidePort[idx];
    } else {
        panic("VegaTLBCoalescer::getPort: unknown port %s\n", if_name);
    }
}

/*
 * This method returns true if the <incoming_pkt>
 * can be coalesced with <coalesced_pkt> and false otherwise.
 * A given set of rules is checked.
 * The rules can potentially be modified based on the TLB level.
 */
bool
VegaTLBCoalescer::canCoalesce(PacketPtr incoming_pkt, PacketPtr coalesced_pkt,
                              Addr pagebytes = VegaISA::PageBytes)
{
    if (disableCoalescing) {
        return false;
    }

    GpuTranslationState *incoming_state =
        safe_cast<GpuTranslationState *>(incoming_pkt->senderState);

    GpuTranslationState *coalesced_state =
        safe_cast<GpuTranslationState *>(coalesced_pkt->senderState);

    // Rule 1: Coalesce requests only if they
    // fall within the same virtual page
    Addr incoming_virt_page_addr =
        roundDown(incoming_pkt->req->getVaddr(), pagebytes);

    Addr coalesced_virt_page_addr =
        roundDown(coalesced_pkt->req->getVaddr(), pagebytes);

    if (incoming_virt_page_addr != coalesced_virt_page_addr) {
        return false;
    }

    //* Rule 2: Coalesce requests only if they
    // share a TLB Mode, i.e. they are both read
    // or write requests.
    BaseMMU::Mode incoming_mode = incoming_state->tlbMode;
    BaseMMU::Mode coalesced_mode = coalesced_state->tlbMode;

    if (incoming_mode != coalesced_mode) {
        return false;
    }

    // when we can coalesce a packet update the reqCnt
    // that is the number of packets represented by
    // this coalesced packet
    if (!incoming_state->isPrefetch) {
        coalesced_state->reqCnt.back() += incoming_state->reqCnt.back();
    }

    return true;
}

/*
 * We need to update the physical addresses of all the translation requests
 * that were coalesced into the one that just returned.
 */
void
VegaTLBCoalescer::updatePhysAddresses(PacketPtr pkt)
{
    GpuTranslationState *sender_state =
        safe_cast<GpuTranslationState *>(pkt->senderState);

    // Make a copy. This gets deleted after the first is sent back on the port
    assert(sender_state->tlbEntry);
    VegaISA::VegaTlbEntry tlb_entry =
        *safe_cast<VegaISA::VegaTlbEntry *>(sender_state->tlbEntry);
    Addr first_entry_vaddr = tlb_entry.vaddr;
    Addr first_entry_paddr = tlb_entry.paddr;
    Addr page_size = tlb_entry.size();

    // Clamp giant walker sizes before they become key granules; true
    // page_size is still used for the range check and paddr computation.
    potentialPagesize.insert(clampCoalescePageSize(page_size));

    // Train future page-size speculation from the actual return size.
    updateSizePredictor(page_size);

    bool uncacheable = tlb_entry.uncacheable();
    int first_hit_level = sender_state->hitLevel;
    bool is_system = pkt->req->systemReq();

    // Save before responding; pkt may be recycled after sendTimingResp().
    const Addr ret_vaddr = pkt->req->getVaddr();

    // Find the outstanding bucket that actually contains this packet.
    // Overlapping 2 MiB and 4 KiB buckets can both be live, so the key alone
    // is ambiguous; packet identity disambiguates it.
    Addr virt_page_addr = 0;
    auto table_it = issuedTranslationsTable.end();
    for (auto pgsize_seen : potentialPagesize) {
        Addr loc_virt_page_addr = roundDown(ret_vaddr, pgsize_seen);
        auto it = issuedTranslationsTable.find(loc_virt_page_addr);
        if (it == issuedTranslationsTable.end()) {
            continue;
        }
        if (std::find(it->second.begin(), it->second.end(), pkt) !=
            it->second.end()) {
            virt_page_addr = loc_virt_page_addr;
            table_it = it;
            break;
        }
    }

    // A returned packet must belong to one outstanding bucket.
    assert(table_it != issuedTranslationsTable.end());

    // Copy the list since sends/reissues may mutate the table.
    std::vector<PacketPtr> coalesced_pkts = table_it->second;

    DPRINTF(GPUTLB, "Update phys. addr. for %d coalesced reqs for "
            "page %#x\n", coalesced_pkts.size(), virt_page_addr);

    for (int i = 0; i < coalesced_pkts.size(); ++i) {
        PacketPtr local_pkt = coalesced_pkts[i];

        Addr local_pkt_vaddr = local_pkt->req->getVaddr();

        // Reissue packets outside the returned page at a clamped granule;
        // range check uses the true page_size.
        if (!(first_entry_vaddr <= local_pkt_vaddr &&
              local_pkt_vaddr < first_entry_vaddr + page_size)) {
            reissue_pkt_helper(local_pkt, clampCoalescePageSize(page_size));
            continue;
        }

        GpuTranslationState *local_sender_state =
            safe_cast<GpuTranslationState *>(local_pkt->senderState);

        // we are sending the packet back, so pop the reqCnt associated
        // with this level in the TLB hiearchy
        if (!local_sender_state->isPrefetch) {
            local_sender_state->reqCnt.pop_back();
            localCycles += curCycle();
        }

        // Only the returned packet already has its physical address.
        // Every other coalesced packet needs to be filled in from the returned page + offset.
        // Use pointer identity instead of the loop index so this stays correct regardless of the packet's position in the bucket.
        if (local_pkt != pkt) {
            Addr paddr = first_entry_paddr +
                         (local_pkt_vaddr & (page_size - 1));
            local_pkt->req->setPaddr(paddr);

            if (uncacheable) {
                local_pkt->req->setFlags(Request::UNCACHEABLE);
            }

            // update senderState->tlbEntry, so we can insert
            // the correct TLBEentry in the TLBs above.
            if (local_sender_state->tlbEntry == NULL) {
                // not set by lower(l2) coalescer
                local_sender_state->tlbEntry =
                    new VegaISA::VegaTlbEntry(
                        1 /* VMID TODO */, first_entry_vaddr,
                        first_entry_paddr, tlb_entry.logBytes,
                        tlb_entry.pte);
            }

            // update the hitLevel for all uncoalesced reqs
            // so that each packet knows where it hit
            // (used for statistics in the CUs)
            local_sender_state->hitLevel = first_hit_level;
        }

        // Copy PTE system bit information to coalesced requests
        local_pkt->req->setSystemReq(is_system);

        ResponsePort *return_port = local_sender_state->ports.back();
        local_sender_state->ports.pop_back();

        // Translation is done - Convert to a response pkt if necessary and
        // send the translation back
        if (local_pkt->isRequest()) {
            local_pkt->makeTimingResponse();
        }

        return_port->sendTimingResp(local_pkt);
    }

    // schedule clean up for end of this cycle
    // This is a maximum priority event and must be on
    // the same cycle as GPUTLB cleanup event to prevent
    // race conditions with an IssueProbeEvent caused by
    // MemSidePort::recvReqRetry
    cleanupQueue.push(virt_page_addr);

    if (!cleanupEvent.scheduled()) {
        schedule(cleanupEvent, curTick());
    }
}

// Re-coalesce a packet after page-size speculation failed.
// 2 MiB returns reissue remaining siblings at 2 MiB; otherwise use 4 KiB.
void
VegaTLBCoalescer::reissue_pkt_helper(PacketPtr pkt, Addr reissue_pgsize)
{
    // first packet of a coalesced request
    PacketPtr first_packet = nullptr;
    // true if we are able to do coalescing
    bool didCoalesce = false;
    // number of coalesced reqs for a given window
    int coalescedReq_cnt = 0;

    GpuTranslationState *sender_state =
        safe_cast<GpuTranslationState *>(pkt->senderState);

    // Never key a group larger than the coalescer's max granule.
    reissue_pgsize = clampCoalescePageSize(reissue_pgsize);

    // Ensure return-side lookup probes this reissue size.
    potentialPagesize.insert(reissue_pgsize);

    DPRINTF(GPUTLB, "Trying to re-issue req at tick: %llu, addr: %#x\n",
            sender_state->issueTime, pkt->req->getVaddr());

    // The tick index is used as a key to the coalescerFIFO hashmap.
    // It is shared by all candidates that fall within the
    // given coalescingWindow.
    Tick tick_index = sender_state->issueTime / coalescingWindow;

    if (coalescerFIFO.count(tick_index)) {
        coalescedReq_cnt = coalescerFIFO[tick_index].size();
    }

    // see if we can coalesce the incoming pkt with another
    // coalesced request with the same tick_index
    for (int i = 0; i < coalescedReq_cnt; ++i) {
        first_packet = coalescerFIFO[tick_index][i].first[0];
        if (coalescerFIFO[tick_index][i].second != reissue_pgsize) {
            continue;
        }

        if (canCoalesce(pkt, first_packet, reissue_pgsize)) {
            coalescerFIFO[tick_index][i].first.push_back(pkt);

            DPRINTF(GPUTLB, "Coalesced re-issued req %i \
                    w/ tick_index %d has %d reqs\n",
                    i, tick_index, coalescerFIFO[tick_index][i].first.size());

            didCoalesce = true;
            break;
        }
    }

    // if this is the first request for this tick_index
    // or we did not manage to coalesce, update stats
    // and make necessary allocations.
    if (!coalescedReq_cnt || !didCoalesce) {
        std::vector<PacketPtr> new_array;
        new_array.push_back(pkt);
        coalescerFIFO[tick_index].push_back(
            std::make_pair(new_array, reissue_pgsize));

        DPRINTF(GPUTLB,
                "coalescerFIFO[%d] now has %d coalesced reqs after "
                "push re-issued req\n",
                tick_index, coalescerFIFO[tick_index].size());
    }

    // schedule probeTLBEvent next cycle to send the
    // coalesced requests to the TLB
    if (!probeTLBEvent.scheduled()) {
        schedule(probeTLBEvent, curTick() + clockPeriod());
    }
}

// Receive translation requests, create a coalesced request,
// and send them to the TLB (TLBProbesPerCycle)
bool
VegaTLBCoalescer::CpuSidePort::recvTimingReq(PacketPtr pkt)
{
    // first packet of a coalesced request
    PacketPtr first_packet = nullptr;
    // true if we are able to do coalescing
    bool didCoalesce = false;
    // number of coalesced reqs for a given window
    int coalescedReq_cnt = 0;

    GpuTranslationState *sender_state =
        safe_cast<GpuTranslationState *>(pkt->senderState);

    bool update_stats = !sender_state->isPrefetch;

    if (coalescer->tlb_level == 1 && coalescer->mustStallCUPort(this)) {
        return false;
    }

    // push back the port to remember the path back
    sender_state->ports.push_back(this);

    if (update_stats) {
        // if reqCnt is empty then this packet does not represent
        // multiple uncoalesced reqs(pkts) but just a single pkt.
        // If it does though then the reqCnt for each level in the
        // hierarchy accumulates the total number of reqs this packet
        // represents
        int req_cnt = 1;

        if (!sender_state->reqCnt.empty()) {
            req_cnt = sender_state->reqCnt.back();
        }

        sender_state->reqCnt.push_back(req_cnt);

        // update statistics
        coalescer->uncoalescedAccesses++;
        req_cnt = sender_state->reqCnt.back();
        DPRINTF(GPUTLB, "receiving pkt w/ req_cnt %d\n", req_cnt);
        coalescer->queuingCycles -= (coalescer->curCycle() * req_cnt);
        coalescer->localqueuingCycles -= coalescer->curCycle();
        coalescer->localCycles -= coalescer->curCycle();
    }

    // Coalesce based on the time the packet arrives at the coalescer (here).
    if (!sender_state->issueTime) {
        sender_state->issueTime = curTick();
    }

    // The tick index is used as a key to the coalescerFIFO hashmap.
    // It is shared by all candidates that fall within the
    // given coalescingWindow.
    Tick tick_index = sender_state->issueTime / coalescer->coalescingWindow;

    if (coalescer->coalescerFIFO.count(tick_index)) {
        coalescedReq_cnt = coalescer->coalescerFIFO[tick_index].size();
    }

    // see if we can coalesce the incoming pkt with another
    // coalesced request with the same tick_index
    for (int i = 0; i < coalescedReq_cnt; ++i) {
        first_packet = coalescer->coalescerFIFO[tick_index][i].first[0];
        Addr pg_size = coalescer->coalescerFIFO[tick_index][i].second;

        if (coalescer->canCoalesce(pkt, first_packet, pg_size)) {
            coalescer->coalescerFIFO[tick_index][i].first.push_back(pkt);

            DPRINTF(GPUTLB, "Coalesced req %i w/ tick_index %d has %d reqs\n",
                    i, tick_index,
                    coalescer->coalescerFIFO[tick_index][i].first.size());

            didCoalesce = true;
            break;
        }
    }

    // if this is the first request for this tick_index
    // or we did not manage to coalesce, update stats
    // and make necessary allocations.
    if (!coalescedReq_cnt || !didCoalesce) {
        if (update_stats) {
            coalescer->coalescedAccesses++;
        }

        std::vector<PacketPtr> new_array;
        new_array.push_back(pkt);
        coalescer->coalescerFIFO[tick_index].push_back(
            std::make_pair(new_array, coalescer->newGroupPageSize()));

        DPRINTF(GPUTLB,
                "coalescerFIFO[%d] now has %d coalesced reqs after "
                "push\n",
                tick_index, coalescer->coalescerFIFO[tick_index].size());
    }

    // schedule probeTLBEvent next cycle to send the
    // coalesced requests to the TLB
    if (!coalescer->probeTLBEvent.scheduled()) {
        coalescer->schedule(coalescer->probeTLBEvent,
                            curTick() + coalescer->clockPeriod());
    }

    return true;
}

void
VegaTLBCoalescer::CpuSidePort::recvReqRetry()
{
    panic("recvReqRetry called");
}

void
VegaTLBCoalescer::CpuSidePort::recvFunctional(PacketPtr pkt)
{

    GpuTranslationState *sender_state =
        safe_cast<GpuTranslationState *>(pkt->senderState);

    bool update_stats = !sender_state->isPrefetch;

    if (update_stats) {
        coalescer->uncoalescedAccesses++;
    }

    Addr virt_page_addr = roundDown(pkt->req->getVaddr(), VegaISA::PageBytes);
    int map_count = coalescer->issuedTranslationsTable.count(virt_page_addr);

    if (map_count) {
        DPRINTF(GPUTLB,
                "Warning! Functional access to addr %#x sees timing "
                "req. pending\n",
                virt_page_addr);
    }

    coalescer->memSidePort[0]->sendFunctional(pkt);
}

AddrRangeList
VegaTLBCoalescer::CpuSidePort::getAddrRanges() const
{
    // currently not checked by the requestor
    AddrRangeList ranges;

    return ranges;
}

/*
 *  a translation completed and returned
 */
bool
VegaTLBCoalescer::MemSidePort::recvTimingResp(PacketPtr pkt)
{
    coalescer->updatePhysAddresses(pkt);

    if (coalescer->tlb_level != 1) {
        return true;
    }

    coalescer->decrementNumDownstream();

    DPRINTF(GPUTLB,
            "recvTimingReq: clscr = %p, numDownstream = %d, max = %d\n",
            coalescer, coalescer->numDownstream, coalescer->maxDownstream);

    coalescer->unstallPorts();
    return true;
}

void
VegaTLBCoalescer::MemSidePort::recvReqRetry()
{
    coalescer->retryEvents++;

    // we've receeived a retry. Schedule a probeTLBEvent
    if (!coalescer->probeTLBEvent.scheduled()) {
        coalescer->schedule(coalescer->probeTLBEvent,
                curTick() + coalescer->clockPeriod());
    }
}

void
VegaTLBCoalescer::MemSidePort::recvFunctional(PacketPtr pkt)
{
    fatal("Memory side recvFunctional() not implemented in TLB coalescer.\n");
}

/*
 * Here we scan the coalescer FIFO and issue the max
 * number of permitted probes to the TLB below. We
 * permit bypassing of coalesced requests for the same
 * tick_index.
 *
 * We do not access the next tick_index unless we've
 * drained the previous one. The coalesced requests
 * that are successfully sent are moved to the
 * issuedTranslationsTable table (the table which keeps
 * track of the outstanding reqs)
 */
void
VegaTLBCoalescer::processProbeTLBEvent()
{
    // number of TLB probes sent so far
    int sent_probes = 0;

    // It is set to true either when the recvTiming of the TLB below
    // returns false or when there is another outstanding request for the
    // same virt. page.

    DPRINTF(GPUTLB, "triggered VegaTLBCoalescer %s\n", __func__);

    if ((tlb_level == 1) && (availDownstreamSlots() == 0)) {
        DPRINTF(GPUTLB, "IssueProbeEvent - no downstream slots, bail out\n");
        downstreamSlotBlocked++;
        updateOccupancyStats();
        return;
    }

    for (auto iter = coalescerFIFO.begin(); iter != coalescerFIFO.end();) {
        int coalescedReq_cnt = iter->second.size();
        int i = 0;
        int vector_index = 0;

        DPRINTF(GPUTLB, "coalescedReq_cnt is %d for tick_index %d\n",
                coalescedReq_cnt, iter->first);

        while (i < coalescedReq_cnt) {
            ++i;
            PacketPtr first_packet = iter->second[vector_index].first[0];
            // The request to coalescer is organized as follows.
            // The coalescerFIFO is a map which is indexed by coalescingWindow
            //  cycle. Only requests that falls in the same coalescingWindow
            //  considered for coalescing. Each entry of a coalescerFIFO is a
            //  vector of vectors. There is one entry for each different
            //  virtual page number and it contains vector of all request that
            //  are coalesced for the same virtual page address

            // compute virtual page address for this request use the assumed
            // page size, stored in pair.second of the coalesced req
            Addr virt_page_addr = roundDown(first_packet->req->getVaddr(),
                                            iter->second[vector_index].second);

            // is there another outstanding request for the same page addr?
            // consider all possible page sizes already observed by the walker.
            int pending_reqs = 0;
            for (auto i_pgsize : potentialPagesize) {
                pending_reqs += issuedTranslationsTable.count(
                    roundDown(first_packet->req->getVaddr(), i_pgsize));
            }

            if (pending_reqs) {
                DPRINTF(GPUTLB,
                        "Cannot issue - There are pending reqs for "
                        "page %#x\n",
                        virt_page_addr);

                pendingBlockedProbes++;
                pendingBlockedPackets +=
                    iter->second[vector_index].first.size();
                updateOccupancyStats();

                ++vector_index;
                continue;
            }

            // send the coalesced request for virt_page_addr
            if (!memSidePort[0]->sendTimingReq(first_packet)) {
                DPRINTF(GPUTLB, "Failed to send TLB request for page %#x",
                        virt_page_addr);

                sendTimingReqFailed++;
                updateOccupancyStats();

                // No need for a retries queue since we are already
                // buffering the coalesced request in coalescerFIFO.
                // Arka:: No point trying to send other requests to TLB at
                // this point since it is busy. Retries will be called later
                // by the TLB below
                return;
            } else {

                if (tlb_level == 1) {
                    incrementNumDownstream();
                }

                GpuTranslationState *tmp_sender_state =
                    safe_cast<GpuTranslationState *>(
                        first_packet->senderState);

                bool update_stats = !tmp_sender_state->isPrefetch;

                if (update_stats) {
                    // req_cnt is total number of packets represented
                    // by the one we just sent counting all the way from
                    // the top of TLB hierarchy (i.e., from the CU)
                    int req_cnt = tmp_sender_state->reqCnt.back();
                    queuingCycles += (curCycle() * req_cnt);

                    DPRINTF(GPUTLB, "%s sending pkt w/ req_cnt %d\n", name(),
                            req_cnt);

                    // pkt_cnt is number of packets we coalesced into the one
                    // we just sent but only at this coalescer level
                    int pkt_cnt = iter->second[vector_index].first.size();
                    localqueuingCycles += (curCycle() * pkt_cnt);
                }

                DPRINTF(GPUTLB, "Successfully sent TLB request for page %#x\n",
                        virt_page_addr);

                // copy coalescedReq to issuedTranslationsTable
                issuedTranslationsTable[virt_page_addr] =
                    iter->second[vector_index].first;
                probesIssued++;
                updateOccupancyStats();

                // erase the entry of this coalesced req
                iter->second.erase(iter->second.begin() + vector_index);

                if (iter->second.empty()) {
                    assert(i == coalescedReq_cnt);
                }

                sent_probes++;

                if (sent_probes == TLBProbesPerCycle ||
                    ((tlb_level == 1) && (!availDownstreamSlots()))) {
                    // Before returning make sure that empty vectors are taken
                    //  out. Not a big issue though since a later invocation
                    //  will take it out anyway.
                    if (iter->second.empty()) {
                        coalescerFIFO.erase(iter);
                    }

                    // schedule probeTLBEvent next cycle to send the
                    // coalesced requests to the TLB
                    if (!probeTLBEvent.scheduled()) {
                        schedule(probeTLBEvent,
                                 cyclesToTicks(curCycle() + Cycles(1)));
                    }
                    return;
                }
            }
        }

        // if there are no more coalesced reqs for this tick_index
        // erase the hash_map with the first iterator
        if (iter->second.empty()) {
            coalescerFIFO.erase(iter++);
        } else {
            ++iter;
        }
    }
}

void
VegaTLBCoalescer::processCleanupEvent()
{
    bool cleaned = false;

    while (!cleanupQueue.empty()) {
        Addr cleanup_addr = cleanupQueue.front();
        cleanupQueue.pop();
        issuedTranslationsTable.erase(cleanup_addr);
        cleanupEntries++;
        cleaned = true;
        updateOccupancyStats();

        DPRINTF(GPUTLB, "Cleanup - Delete coalescer entry with key %#x\n",
                cleanup_addr);
    }

    if (cleaned && !coalescerFIFO.empty() && !probeTLBEvent.scheduled()) {
        schedule(probeTLBEvent, cyclesToTicks(curCycle() + Cycles(1)));
    }
}

void
VegaTLBCoalescer::regStats()
{
    ClockedObject::regStats();

    uncoalescedAccesses.name(name() + ".uncoalesced_accesses")
        .desc("Number of uncoalesced TLB accesses");

    coalescedAccesses.name(name() + ".coalesced_accesses")
        .desc("Number of coalesced TLB accesses");

    queuingCycles.name(name() + ".queuing_cycles")
        .desc("Number of cycles spent in queue");

    localqueuingCycles.name(name() + ".local_queuing_cycles")
        .desc("Number of cycles spent in queue for all incoming reqs");

    localCycles.name(name() + ".local_cycles")
        .desc("Number of cycles spent in queue for all incoming reqs");

    pendingBlockedProbes.name(name() + ".pending_blocked_probes")
        .desc("Probe attempts blocked by an overlapping outstanding translation");

    pendingBlockedPackets.name(name() + ".pending_blocked_packets")
        .desc("Buffered packets in probe attempts blocked by an overlapping outstanding translation");

    downstreamSlotBlocked.name(name() + ".downstream_slot_blocked")
        .desc("Probe event invocations blocked by downstream slot exhaustion");

    sendTimingReqFailed.name(name() + ".send_timing_req_failed")
        .desc("TLB probe sends rejected by the downstream port");

    probesIssued.name(name() + ".probes_issued")
        .desc("Coalesced translation probes issued downstream");

    cleanupEntries.name(name() + ".cleanup_entries")
        .desc("Outstanding translation entries cleaned up after response");

    retryEvents.name(name() + ".retry_events")
        .desc("Request retry callbacks from downstream TLB port");

    fifoMaxEntries.name(name() + ".fifo_max_entries")
        .desc("Maximum number of coalesced requests buffered in the FIFO");

    issuedTranslationsMax.name(name() + ".issued_translations_max")
        .desc("Maximum number of outstanding translation table entries");

    issuedTranslationPacketsMax.name(name() + ".issued_translation_packets_max")
        .desc("Maximum number of packets represented by outstanding translations");

    localLatency.name(name() + ".local_latency")
        .desc("Avg. latency over all incoming pkts");

    latency.name(name() + ".latency")
        .desc("Avg. latency over all incoming pkts");

    localLatency = localqueuingCycles / uncoalescedAccesses;
    latency = localCycles / uncoalescedAccesses;
}

void
VegaTLBCoalescer::insertStalledPortIfNotMapped(CpuSidePort *port)
{
    assert(tlb_level == 1);
    if (stalledPortsMap.count(port) != 0) {
        return; // we already know this port is stalled
    }

    stalledPortsMap[port] = port;
    stalledPortsQueue.push(port);
    DPRINTF(GPUTLB,
            "insertStalledPortIfNotMapped: port %p, mapSz = %d, qsz = %d\n",
            port, stalledPortsMap.size(), stalledPortsQueue.size());
}

bool
VegaTLBCoalescer::mustStallCUPort(CpuSidePort *port)
{
    assert(tlb_level == 1);

    DPRINTF(GPUTLB, "mustStallCUPort: downstream = %d, max = %d\n",
            numDownstream, maxDownstream);

    if (availDownstreamSlots() == 0 || numDownstream == maxDownstream) {
        warn("RED ALERT - VegaTLBCoalescer::mustStallCUPort\n");
        insertStalledPortIfNotMapped(port);
        return true;
    } else {
        return false;
    }
}

void
VegaTLBCoalescer::unstallPorts()
{
    assert(tlb_level == 1);
    if (!stalledPorts() || availDownstreamSlots() == 0) {
        return;
    }

    DPRINTF(GPUTLB, "unstallPorts()\n");
    /*
     * this check is needed because we can be called from recvTiiningResponse()
     * or, synchronously due to having called sendRetry, from recvTimingReq()
     */
    if (availDownstreamSlots() == 0) { // can happen if retry sent 1 downstream
        return;
    }
    /*
     *  Consider this scenario
     *        1) max downstream is reached
     *        2) port1 tries to send a req, cant => stalledPortsQueue = [port1]
     *        3) port2 tries to send a req, cant => stalledPortsQueue = [port1,
     *              port2]
     *        4) a request completes and we remove port1 from both data
     *              structures & call
     *             sendRetry => stalledPortsQueue = [port2]
     *        5) port1 sends one req downstream and a second is rejected
     *             => stalledPortsQueue = [port2, port1]
     *
     *        so we round robin and each stalled port can send 1 req on retry
     */
    assert(availDownstreamSlots() == 1);
    auto port = stalledPortsQueue.front();
    DPRINTF(GPUTLB, "sending retry for port = %p(%s)\n", port, port->name());
    stalledPortsQueue.pop();
    auto iter = stalledPortsMap.find(port);
    assert(iter != stalledPortsMap.end());
    stalledPortsMap.erase(iter);
    port->sendRetryReq(); // cu will synchronously call recvTimingReq
}

} // namespace gem5
