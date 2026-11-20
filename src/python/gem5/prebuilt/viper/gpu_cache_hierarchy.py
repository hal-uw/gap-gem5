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

import math
from typing import (
    Callable,
    Optional,
)

from m5.objects import (
    AddrRange,
    DMASequencer,
    HBM_2000_4H_1x64,
    MemCtrl,
    RubyCache,
    RubySequencer,
    RubySystem,
    SrcClockDomain,
    TreePLRURP,
    VIPERCoalescer,
    VIPERSequencer,
    VoltageDomain,
)

from ...coherence_protocol import CoherenceProtocol
from ...components.cachehierarchies.ruby.abstract_ruby_cache_hierarchy import (
    AbstractRubyCacheHierarchy,
)
from ...components.cachehierarchies.ruby.caches.viper.directory import (
    ViperGPUDirectory,
)
from ...components.cachehierarchies.ruby.caches.viper.dma_controller import (
    ViperGPUDMAController,
)
from ...components.cachehierarchies.ruby.caches.viper.sqc import SQCCache
from ...components.cachehierarchies.ruby.caches.viper.tcc import TCCCache
from ...components.cachehierarchies.ruby.caches.viper.tcp import TCPCache
from ...components.devices.gpus.viper_shader import ViperShader
from ...components.memory.abstract_memory_system import AbstractMemorySystem
from ...utils.requires import requires
from .viper_network import SimpleDoubleCrossbar


class ViperGPUCacheHierarchy(AbstractRubyCacheHierarchy):
    _seqs = 0

    @classmethod
    def seqCount(cls):
        # Use SeqCount not class since we need global count
        cls._seqs += 1
        return cls._seqs - 1

    def __init__(
        self,
        gpu_memory: AbstractMemorySystem,
        tcp_size: str,
        tcp_assoc: int,
        sqc_size: str,
        sqc_assoc: int,
        scalar_size: str,
        scalar_assoc: int,
        tcc_size: str,
        tcc_assoc: int,
        tcc_count: int,
        cu_per_sqc: int,
        cache_line_size: int,
        shader: ViperShader,
        cu_per_scalar_cache: Optional[int] = None,
        fabric_clock: str = "1801MHz",
        gpu_clock: Optional[str] = None,
        network_factory: Optional[Callable] = None,
        num_virtual_networks: int = 6,
    ):
        """
        :param size: The size of each cache in the heirarchy.
        :param assoc: The associativity of each cache.
        :param cu_per_scalar_cache: CUs sharing one scalar cache. Defaults to
            cu_per_sqc.
        :param fabric_clock: Clock of the network, directories (L3) and DMA
            controllers.
        :param gpu_clock: Clock of the CU caches (TCP, SQC, scalar) and L2
            (TCC), which sit inside the XCD. Defaults to fabric_clock.
        :param network_factory: Called with the RubySystem to create the GPU
            network. Defaults to SimpleDoubleCrossbar.
        :param num_virtual_networks: Virtual networks of the GPU network
            (VIPER needs at least 6).
        """
        super().__init__()

        self._tcp_size = tcp_size
        self._tcp_assoc = tcp_assoc
        self._sqc_size = sqc_size
        self._sqc_assoc = sqc_assoc
        self._scalar_size = scalar_size
        self._scalar_assoc = scalar_assoc
        self._tcc_size = tcc_size
        self._tcc_assoc = tcc_assoc
        self._cache_line_size = cache_line_size

        # We have everything we need to know to create the GPU cache hierarchy
        # immediately. Therefore, an incorporate_cache method is not part of
        # this cache hierarchy. Go ahead and incorporate everything now.
        requires(coherence_protocol_required=CoherenceProtocol.GPU_VIPER)

        self.ruby_gpu = RubySystem()
        self.ruby_gpu.block_size_bytes = cache_line_size

        # Ruby network for this GPU
        if network_factory is None:
            network_factory = SimpleDoubleCrossbar
        self.ruby_gpu.network = network_factory(self.ruby_gpu)

        # VIPER uses at least 6 virtual networks.
        self.ruby_gpu.number_of_virtual_networks = num_virtual_networks
        self.ruby_gpu.network.number_of_virtual_networks = num_virtual_networks

        # There is a single local list of all of the controllers to make it
        # easier to connect everything to the GPU network. This can be
        # customized depending on the topology/network requirements.
        self._tcp_controllers = []
        self._sqc_controllers = []
        self._scalar_controllers = []
        self._tcc_controllers = []
        self._directory_controllers = []
        self._dma_controllers = []
        self._mem_ctrls = []

        self.clk_domain = SrcClockDomain(
            clock=fabric_clock,
            voltage_domain=VoltageDomain(),
        )
        if gpu_clock is None:
            xcd_clk_domain = self.clk_domain
        else:
            self.gpu_clk_domain = SrcClockDomain(
                clock=gpu_clock,
                voltage_domain=VoltageDomain(),
            )
            xcd_clk_domain = self.gpu_clk_domain

        # Variables used by multiple objects are defined once here
        tcc_bits = int(math.log(tcc_count, 2))
        deadlock_threshold = 500000
        # Sets are indexed above the line offset (RubyCache defaults to bit
        # 6, i.e. 64 B lines). The TCC is also selected by the tcc_bits above
        # the line offset, which are constant within one TCC.
        line_bits = int(math.log(cache_line_size, 2))
        tcc_index_bit = line_bits + tcc_bits

        # Create one TCP per CU
        compute_units = shader.get_compute_units()
        for idx, cu in enumerate(compute_units):
            tcp = TCPCache(
                tcp_size=self._tcp_size,
                tcp_assoc=self._tcp_assoc,
                network=self.ruby_gpu.network,
                cache_line_size=self._cache_line_size,
            )

            tcp.version = idx
            tcp.L1cache.start_index_bit = line_bits

            tcp.sequencer = RubySequencer(
                version=self.seqCount(),
                dcache=tcp.L1cache,
                ruby_system=self.ruby_gpu,
                is_cpu_sequencer=True,
            )

            tcp.coalescer = VIPERCoalescer(
                version=self.seqCount(),
                icache=tcp.L1cache,
                dcache=tcp.L1cache,
                ruby_system=self.ruby_gpu,
                support_inst_reqs=False,
                is_cpu_sequencer=False,
                deadlock_threshold=deadlock_threshold,
                max_coalesces_per_cycle=1,
                gmTokenPort=cu.gmTokenPort,
            )

            for port_idx in range(cu.wf_size):
                cu.memory_port[port_idx] = tcp.coalescer.in_ports

            tcp.ruby_system = self.ruby_gpu
            tcp.TCC_select_num_bits = tcc_bits
            tcp.use_seq_not_coal = False
            tcp.issue_latency = 1
            tcp.clk_domain = xcd_clk_domain
            tcp.recycle_latency = 10
            tcp.WB = False
            tcp.disableL1 = False

            self._tcp_controllers.append(tcp)

        # This check ensures there are a same number of CUs with shared SQC
        # and Scalar caches.
        num_cus = len(shader.get_compute_units())
        assert (num_cus % cu_per_sqc) == 0
        num_sqcs = num_cus // cu_per_sqc

        for idx in range(num_sqcs):
            sqc = SQCCache(
                sqc_size=self._sqc_size,
                sqc_assoc=self._sqc_assoc,
                network=self.ruby_gpu.network,
                cache_line_size=self._cache_line_size,
            )

            sqc.version = idx
            sqc.L1cache.start_index_bit = line_bits

            sqc.sequencer = VIPERSequencer(
                version=self.seqCount(),
                dcache=sqc.L1cache,
                ruby_system=self.ruby_gpu,
                support_data_reqs=False,
                is_cpu_sequencer=False,
                deadlock_threshold=deadlock_threshold,
            )

            # SQC is shared across {cu_per_sqc} CUs.
            cu_base = cu_per_sqc * idx
            for cu_num in range(cu_per_sqc):
                cu_id = cu_base + cu_num
                compute_units[cu_id].sqc_port = sqc.sequencer.in_ports

            sqc.ruby_system = self.ruby_gpu
            sqc.TCC_select_num_bits = tcc_bits
            sqc.clk_domain = xcd_clk_domain
            sqc.recycle_latency = 10

            self._sqc_controllers.append(sqc)

        if cu_per_scalar_cache is None:
            cu_per_scalar_cache = cu_per_sqc
        assert (num_cus % cu_per_scalar_cache) == 0
        num_scalars = num_cus // cu_per_scalar_cache
        for idx in range(num_scalars):
            scalar = SQCCache(
                sqc_size=self._scalar_size,
                sqc_assoc=self._scalar_assoc,
                network=self.ruby_gpu.network,
                cache_line_size=self._cache_line_size,
            )

            # Scalar uses same controller as SQC, so add SQC count
            scalar.version = idx + num_sqcs
            scalar.L1cache.start_index_bit = line_bits

            scalar.sequencer = VIPERSequencer(
                version=self.seqCount(),
                dcache=scalar.L1cache,
                ruby_system=self.ruby_gpu,
                support_data_reqs=False,
                is_cpu_sequencer=False,
                deadlock_threshold=deadlock_threshold,
            )

            # Scalar cache is shared across {cu_per_scalar_cache} CUs.
            cu_base = cu_per_scalar_cache * idx
            for cu_num in range(cu_per_scalar_cache):
                cu_id = cu_base + cu_num
                compute_units[cu_id].scalar_port = scalar.sequencer.in_ports

            scalar.ruby_system = self.ruby_gpu
            scalar.TCC_select_num_bits = tcc_bits
            scalar.clk_domain = xcd_clk_domain
            scalar.recycle_latency = 10

            self._scalar_controllers.append(scalar)

        # Create TCCs (GPU L2 cache)
        for idx in range(tcc_count):
            tcc = TCCCache(
                tcc_size=self._tcc_size,
                tcc_assoc=self._tcc_assoc,
                network=self.ruby_gpu.network,
                cache_line_size=self._cache_line_size,
            )

            tcc.version = idx
            tcc.L2cache.start_index_bit = tcc_index_bit

            tcc.ruby_system = self.ruby_gpu
            tcc.WB = False
            tcc.clk_domain = xcd_clk_domain
            tcc.recycle_latency = 10

            self._tcc_controllers.append(tcc)

        # Create DMA controllers
        for i, port in enumerate(shader.get_gpu_dma_ports()):
            ctrl = ViperGPUDMAController(
                self.ruby_gpu.network, self._cache_line_size
            )
            ctrl.dma_sequencer = DMASequencer(version=i, in_ports=port)

            ctrl.version = len(self._dma_controllers)
            ctrl.ruby_system = self.ruby_gpu
            ctrl.dma_sequencer.ruby_system = self.ruby_gpu

            self._dma_controllers.append(ctrl)

        gpu_memory.set_memory_range([AddrRange(0, size=gpu_memory.get_size())])
        self._mem_ctrls = gpu_memory.get_memory_controllers()
        for addr_range, port in gpu_memory.get_mem_ports():
            dir = ViperGPUDirectory(
                self.ruby_gpu.network,
                self._cache_line_size,
                addr_range,
                port,
            )

            dir.ruby_system = self.ruby_gpu
            dir.TCC_select_num_bits = tcc_bits
            dir.version = len(self._directory_controllers)
            self._directory_controllers.append(dir)

            dir.L3CacheMemory = RubyCache(
                size="16MiB",
                assoc=16,
                atomicALUs=64,
                replacement_policy=TreePLRURP(),
                resourceStalls=False,
                dataArrayBanks=16,
                tagArrayBanks=16,
                dataAccessLatency=20,
                tagAccessLatency=15,
            )
            # Index the L3 above the interleave bits that select this
            # directory, which are constant for every line it holds.
            if addr_range.masks:
                dir.L3CacheMemory.start_index_bit = max(
                    (m & -m).bit_length() for m in addr_range.masks
                )

        # Number of sequencers = one per TCP, SQC, and Scalar + one per DMA.
        self.ruby_gpu.num_of_sequencers = (
            len(self._tcp_controllers)
            + len(self._sqc_controllers)
            + len(self._scalar_controllers)
            + len(self._tcc_controllers)
            + len(self._dma_controllers)
        )

        # Assign the controllers to their parent objects. Give nice names.
        self.ruby_gpu.tcp_controllers = self._tcp_controllers
        self.ruby_gpu.sqc_controllers = self._sqc_controllers
        self.ruby_gpu.scalar_controllers = self._scalar_controllers
        self.ruby_gpu.tcc_controllers = self._tcc_controllers
        self.ruby_gpu.directory_controllers = self._directory_controllers

        # Connect the controllers using the network topology
        self.ruby_gpu.network.connect(
            self._tcp_controllers
            + self._sqc_controllers
            + self._scalar_controllers
            + self._tcc_controllers
            + self._directory_controllers
            + self._dma_controllers
        )
        self.ruby_gpu.network.setup_buffers()

    def get_mem_ctrls(self):
        return self._mem_ctrls
