# Copyright (c) 2021 The Regents of the University of California.
# All Rights Reserved
#
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

from m5.objects import (
    SimpleExtLink,
    SimpleIntLink,
    SimpleNetwork,
    Switch,
)


class SimplePt2Pt(SimpleNetwork):
    """A simple point-to-point network. This does not use garnet."""

    def __init__(self, ruby_system):
        super().__init__()
        self.netifs = []

        # TODO: These should be in a base class
        # https://gem5.atlassian.net/browse/GEM5-1039
        self.ruby_system = ruby_system

    def connect(self, controllers):
        """Connect all of the controllers to routers and connect the routers
        together in a point-to-point network.
        """
        # Create one router/switch per controller in the system
        self.routers = [Switch(router_id=i) for i in range(len(controllers))]

        # Make a link from each controller to the router. The link goes
        # externally to the network.
        self.ext_links = [
            SimpleExtLink(link_id=i, ext_node=c, int_node=self.routers[i])
            for i, c in enumerate(controllers)
        ]

        # Make an "internal" link (internal to the network) between every pair
        # of routers.
        link_count = 0
        int_links = []
        for ri in self.routers:
            for rj in self.routers:
                if ri == rj:
                    continue  # Don't connect a router to itself!
                link_count += 1
                int_links.append(
                    SimpleIntLink(link_id=link_count, src_node=ri, dst_node=rj)
                )
        self.int_links = int_links


class SimpleDoubleCrossbar(SimpleNetwork):
    """
    GPU network with crossbars between CU caches and L2 caches and between L2
    caches and directories/memory controllers/DMAs using SimpleNetwork.
    """

    def __init__(self, ruby_system):
        super().__init__()
        self.netifs = []

        self.ruby_system = ruby_system

    def connect(self, controllers):
        l2_xbar_types = (
            "GPU_VIPER_TCP_Controller",
            "GPU_VIPER_SQC_Controller",
            "GPU_VIPER_TCC_Controller",
        )
        soc_xbar_types = (
            "GPU_VIPER_DMA_Controller",
            "GPU_VIPER_Directory_Controller",
        )

        # Create one router per controller plus a crossbar for L2 controllers
        # and a crossbar for SoC controllers.
        routers = [Switch(router_id=i) for i in range(len(controllers))]
        routers.append(Switch(router_id=len(routers)))
        routers.append(Switch(router_id=len(routers)))
        self.routers = routers

        # Routers 0 ... N-2 connect to the individual controllers
        self.ext_links = [
            SimpleExtLink(link_id=i, ext_node=c, int_node=self.routers[i])
            for i, c in enumerate(controllers)
        ]

        # Connect compute unit components and L2s to L2 crossbar in both
        # directions.
        l2_xbar_id = len(controllers)
        soc_xbar_id = l2_xbar_id + 1
        int_links = []

        for ext_link in self.ext_links:
            if ext_link.ext_node.type in l2_xbar_types:
                int_links.append(
                    SimpleIntLink(
                        link_id=len(int_links),
                        src_node=ext_link.int_node,
                        dst_node=self.routers[l2_xbar_id],
                    )
                )
                int_links.append(
                    SimpleIntLink(
                        link_id=len(int_links),
                        src_node=self.routers[l2_xbar_id],
                        dst_node=ext_link.int_node,
                    )
                )
            elif ext_link.ext_node.type in soc_xbar_types:
                int_links.append(
                    SimpleIntLink(
                        link_id=len(int_links),
                        src_node=ext_link.int_node,
                        dst_node=self.routers[soc_xbar_id],
                    )
                )
                int_links.append(
                    SimpleIntLink(
                        link_id=len(int_links),
                        src_node=self.routers[soc_xbar_id],
                        dst_node=ext_link.int_node,
                    )
                )

        # Connect L2 xbar to SoC xbar.
        int_links.append(
            SimpleIntLink(
                link_id=len(int_links),
                src_node=self.routers[l2_xbar_id],
                dst_node=self.routers[soc_xbar_id],
            )
        )
        int_links.append(
            SimpleIntLink(
                link_id=len(int_links),
                src_node=self.routers[soc_xbar_id],
                dst_node=self.routers[l2_xbar_id],
            )
        )

        # Finalize network int_links for unproxy
        self.int_links = int_links


class SimpleCrossbar(SimpleNetwork):
    """
    GPU network with one router per controller and a single central crossbar
    router, using SimpleNetwork. Every controller reaches every other one in
    two internal hops (its router -> crossbar -> destination router).

    Link widths (``bandwidth_factor``, bytes per network cycle):

    * ``link_width_bytes``: default for every link.
    * ``dir_link_width_bytes``: if set, the external link of each GPU
      directory and both internal links of its router (the L2 <-> IOD side).
    * ``xcd_clock_ratio``: if set, the links of the TCP, SQC and TCC
      controllers and their routers are scaled by this factor. The network
      runs on the fabric clock while these caches sit inside the XCD on the
      GPU clock, so the factor gpu_clock / fabric_clock keeps their width per
      GPU cycle.
    """

    def __init__(
        self,
        ruby_system,
        link_width_bytes: int = 16,
        dir_link_width_bytes: int = None,
        xcd_clock_ratio: float = None,
        link_latency: int = 1,
    ):
        super().__init__()
        self.netifs = []

        self.ruby_system = ruby_system
        self._link_width_bytes = link_width_bytes
        self._dir_link_width_bytes = dir_link_width_bytes
        self._xcd_clock_ratio = xcd_clock_ratio
        self._link_latency = link_latency

    def connect(self, controllers):
        dir_types = ("GPU_VIPER_Directory_Controller",)
        xcd_types = (
            "GPU_VIPER_TCP_Controller",
            "GPU_VIPER_SQC_Controller",
            "GPU_VIPER_TCC_Controller",
        )

        def width(ctrl):
            w = self._link_width_bytes
            if ctrl.type in dir_types and self._dir_link_width_bytes:
                w = self._dir_link_width_bytes
            if ctrl.type in xcd_types and self._xcd_clock_ratio:
                w = round(w * self._xcd_clock_ratio)
            return w

        # One router per controller plus the crossbar router (last).
        routers = [Switch(router_id=i) for i in range(len(controllers) + 1)]
        self.routers = routers
        xbar = routers[-1]

        self.ext_links = [
            SimpleExtLink(
                link_id=i,
                ext_node=c,
                int_node=routers[i],
                latency=self._link_latency,
                bandwidth_factor=width(c),
            )
            for i, c in enumerate(controllers)
        ]

        int_links = []
        for i, c in enumerate(controllers):
            int_links.append(
                SimpleIntLink(
                    link_id=len(controllers) + len(int_links),
                    src_node=routers[i],
                    dst_node=xbar,
                    latency=self._link_latency,
                    bandwidth_factor=width(c),
                )
            )
        for i, c in enumerate(controllers):
            int_links.append(
                SimpleIntLink(
                    link_id=len(controllers) + len(int_links),
                    src_node=xbar,
                    dst_node=routers[i],
                    latency=self._link_latency,
                    bandwidth_factor=width(c),
                )
            )
        self.int_links = int_links
