# Copyright (c) 2021 Advanced Micro Devices, Inc.
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

from importlib import *

from network import Network

from m5.objects import *
from m5.util import fatal
from m5.util.convert import toFrequency


class DisjointSimple(SimpleNetwork):
    def __init__(self, ruby_system):
        super().__init__()

        self.netifs = []
        self.routers = []
        self.int_links = []
        self.ext_links = []
        self.ruby_system = ruby_system

    def connectCPU(self, opts, controllers):
        # Setup parameters for makeTopology call for CPU network
        topo_module = import_module(f"topologies.{opts.cpu_topology}")
        topo_class = getattr(topo_module, opts.cpu_topology)
        _topo = topo_class(controllers)
        _topo.makeTopology(opts, self, SimpleIntLink, SimpleExtLink, Switch)

        self.initSimple(opts, self.int_links, self.ext_links)

    def connectGPU(self, opts, controllers):
        # Setup parameters for makeTopology call for GPU network
        topo_module = import_module(f"topologies.{opts.gpu_topology}")
        topo_class = getattr(topo_module, opts.gpu_topology)
        _topo = topo_class(controllers)
        _topo.makeTopology(opts, self, SimpleIntLink, SimpleExtLink, Switch)

        # Without --bw-scalor, keep the widths the topology set from
        # --link-width-bits instead of scaling them with the CU count.
        if opts.bw_scalor > 0:
            for link in self.int_links:
                link.bandwidth_factor = (
                    16 * opts.num_compute_units * opts.bw_scalor
                )

            for link in self.ext_links:
                link.bandwidth_factor = (
                    16 * opts.num_compute_units * opts.bw_scalor
                )

        # The L2 (TCC) side of the crossbar keeps the full link width: each
        # L2 channel reads out a 128B line per clock toward the CUs. The
        # L2<->IOD side is half as wide: each L2 channel connects to the
        # IOD over a 64B channel, and each Infinity Cache channel (one GPU
        # directory here) is 64B wide (CDNA 3 white paper, pp. 10-11). So
        # halve every link on a GPU directory's path through the crossbar:
        # its external link and both internal links of its router.
        # --gpu-dir-link-width-bits sets that width directly instead.
        dir_width = getattr(opts, "gpu_dir_link_width_bits", None)
        if dir_width is not None:
            assert (
                dir_width % 8 == 0
            ), "--gpu-dir-link-width-bits must be a whole number of bytes"

        def dir_link_width(link):
            if dir_width is not None:
                return dir_width // 8
            return link.bandwidth_factor // 2

        dir_routers = set()
        for link in self.ext_links:
            if isinstance(link.ext_node, GPU_VIPER_Directory_Controller):
                link.bandwidth_factor = dir_link_width(link)
                dir_routers.add(link.int_node)
        for link in self.int_links:
            if link.src_node in dir_routers or link.dst_node in dir_routers:
                link.bandwidth_factor = dir_link_width(link)

        # The whole GPU network runs on the fabric clock, but the L1<->L2
        # interconnect is inside the XCD and runs on the GPU clock (the
        # L2 is on the GPU clock too). Scale the widths of the XCD-side
        # links (TCP, SQC, scalar cache, TCC) by gpu_clock / fabric_clock
        # so they carry their width per GPU cycle, e.g. 128B per GPU cycle
        # is 149B per 1.8GHz fabric cycle at 2.1GHz. Directory (IOD) and
        # DMA links stay at their fabric-clock widths. A separate L1-L2
        # network on the GPU clock would model this directly.
        clock_ratio = toFrequency(opts.gpu_clock) / toFrequency(
            opts.fabric_clock
        )
        xcd_types = (
            GPU_VIPER_TCP_Controller,
            GPU_VIPER_SQC_Controller,
            GPU_VIPER_TCC_Controller,
        )
        xcd_routers = set()
        for link in self.ext_links:
            if isinstance(link.ext_node, xcd_types):
                link.bandwidth_factor = round(
                    int(link.bandwidth_factor) * clock_ratio
                )
                xcd_routers.add(link.int_node)
        for link in self.int_links:
            if link.src_node in xcd_routers or link.dst_node in xcd_routers:
                link.bandwidth_factor = round(
                    int(link.bandwidth_factor) * clock_ratio
                )

        self.initSimple(opts, self.int_links, self.ext_links)

    def initSimple(self, opts, int_links, ext_links):
        # Attach links to network
        self.int_links = int_links
        self.ext_links = ext_links

        self.setup_buffers()


class DisjointGarnet(GarnetNetwork):
    def __init__(self, ruby_system):
        super().__init__()

        self.netifs = []
        self.ruby_system = ruby_system

    def connectCPU(self, opts, controllers):
        # Setup parameters for makeTopology call for CPU network
        topo_module = import_module(f"topologies.{opts.cpu_topology}")
        topo_class = getattr(topo_module, opts.cpu_topology)
        _topo = topo_class(controllers)
        _topo.makeTopology(
            opts, self, GarnetIntLink, GarnetExtLink, GarnetRouter
        )

        Network.init_network(opts, self, GarnetNetworkInterface)

    def connectGPU(self, opts, controllers):
        # Setup parameters for makeTopology call
        topo_module = import_module(f"topologies.{opts.gpu_topology}")
        topo_class = getattr(topo_module, opts.gpu_topology)
        _topo = topo_class(controllers)
        _topo.makeTopology(
            opts, self, GarnetIntLink, GarnetExtLink, GarnetRouter
        )

        Network.init_network(opts, self, GarnetNetworkInterface)
