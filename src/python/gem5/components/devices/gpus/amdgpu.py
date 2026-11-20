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

from typing import Optional

from m5.objects import (
    HBM3_MI300X_1x64,
    AMDGPUDevice,
    PciMemBar,
    SrcClockDomain,
    SubSystem,
    VoltageDomain,
)

from ....components.memory.abstract_memory_system import AbstractMemorySystem
from ....components.memory.hbm import HighBandwidthMemory
from ....prebuilt.viper.gpu_cache_hierarchy import ViperGPUCacheHierarchy
from ....prebuilt.viper.viper_network import SimpleCrossbar
from .viper_shader import ViperShader


class BaseViperGPU(SubSystem):
    _base_pci_dev = 8
    _gpu_count = 0
    _my_id = 0
    _debug_commands = ""
    _discovery_value = 2

    @classmethod
    def next_pci_dev(cls):
        cls._gpu_count += 1  # Use count for this particular type
        return cls._base_pci_dev + cls._gpu_count - 1

    @classmethod
    def get_gpu_count(cls):
        return cls._gpu_count

    def __init__(
        self,
        gpu_memory: AbstractMemorySystem,
        ipt_binary: str = "",
        debug: bool = False,
    ):
        super().__init__()
        if gpu_memory.has_parent():
            raise ValueError(
                "`memory` should not have a parent, i.e. you should "
                "instantiate the gpu memory like gpu_memory = HBM2Stack() "
                "and **not** like board.gpu_memory = HBM2Stack()"
            )
        self.memory = gpu_memory
        if debug:
            self._debug_commands = (
                "dmesg -n8\n"
                "uname -r\n"
                "lspci -v\n"
                "echo 0 > /proc/sys/kernel/printk_ratelimit\n"
            )

        # From amdgpu driver: discovery=0 -> VRAM (gem5 user provided ipt_binary)
        #                     discovery=2 -> Use file from gem5-resources disk
        self._discovery_value = 2 if ipt_binary == "" else 0

        # Setup various PCI related parameters
        self._my_id = self.get_gpu_count()
        pci_dev = self.next_pci_dev()

        self.device = AMDGPUDevice(
            pci_func=0, pci_dev=pci_dev, ipt_binary=ipt_binary
        )

    def _cache_hierarchy_options(self) -> dict:
        """Extra keyword arguments for ViperGPUCacheHierarchy."""
        return {}

    def get_cache_line_size(self) -> int:
        return self._cache_line_size

    def setup_caches(self):
        # The Shader sends one SQC invalidate per SQC at kernel launch, so its
        # cu_per_sqc must match the number of SQCs built below.
        self.shader.cu_per_sqc = self._cu_per_sqc

        # Make the cache hierarchy. This will create an independent RubySystem
        # class containing only the GPU caches with no network connection to
        # the CPU cache hierarchy.
        self.gpu_caches = ViperGPUCacheHierarchy(
            gpu_memory=self.memory,
            tcp_size=self._tcp_size,
            tcp_assoc=self._tcp_assoc,
            sqc_size=self._sqc_size,
            sqc_assoc=self._sqc_assoc,
            scalar_size=self._scalar_size,
            scalar_assoc=self._scalar_assoc,
            tcc_size=self._tcc_size,
            tcc_assoc=self._tcc_assoc,
            tcc_count=self._tcc_count,
            cu_per_sqc=self._cu_per_sqc,
            cache_line_size=self._cache_line_size,
            shader=self.shader,
            **self._cache_hierarchy_options(),
        )

    def set_shader(self, shader: ViperShader):
        self.shader = shader

    def get_cpu_dma_ports(self):
        return self.shader.get_cpu_dma_ports()

    def connectGPU(self, board: "ViperBoard") -> None:
        # Connect a CPU pointer. This is only used for SE mode. Any CPU will
        # work, so pick assuming there is at least one
        cpus = board.get_processor()
        self.shader.set_cpu_pointer(cpus.cores[0].core)

        # Connect all PIO buses
        self.shader.connect_iobus(board.get_io_bus(), board.get_pci_host())

        # Collect GPU memory controllers created in the GPU cache hierarchy.
        # First assign them as a child to the device so the SimObject unproxy.
        # The device requires the memories parameter to be set as the system
        # pointer required by the AbstractMemory class is set by AMDGPUDevice.
        self.device.memories = self.memory.get_mem_interfaces()

        board.get_pci_host().devices.append(self.device)


# A scaled down MI210-like device. Defaults to ~1/4th of an MI210.
class MI210(BaseViperGPU):
    def __init__(
        self,
        gpu_memory: AbstractMemorySystem,
        num_cus: int = 32,
        cu_per_sqc: int = 4,
        tcp_size: str = "16KiB",
        tcp_assoc: int = 16,
        sqc_size: str = "32KiB",
        sqc_assoc: int = 8,
        scalar_size: str = "32KiB",
        scalar_assoc: int = 8,
        tcc_size: str = "256KiB",
        tcc_assoc: int = 16,
        tcc_count: int = 8,
        cache_line_size: int = 64,
    ):
        super().__init__(gpu_memory=gpu_memory)

        self._cu_per_sqc = cu_per_sqc
        self._tcp_size = tcp_size
        self._tcp_assoc = tcp_assoc
        self._sqc_size = sqc_size
        self._sqc_assoc = sqc_assoc
        self._scalar_size = scalar_size
        self._scalar_assoc = scalar_assoc
        self._tcc_size = tcc_size
        self._tcc_assoc = tcc_assoc
        self._tcc_count = tcc_count
        self._cache_line_size = cache_line_size

        self.device.device_name = "MI200"

        self.device.DeviceID = 0x740F
        self.device.SubsystemVendorID = 0x1002
        self.device.SubsystemID = 0x0C34

        # Setup device-specific address ranges for various SoC components.
        shader = ViperShader(
            self._my_id,
            num_cus,
            cache_line_size,
            self.device,
            gpu_memory.get_size(),
        )
        self.set_shader(shader)
        self.setup_caches()

        # Setup the SDMA engines depending on device. The MMIO base addresses
        # can be found in the driver code under:
        # include/asic_reg/sdmaX/sdmaX_Y_Z_offset.h
        num_sdmas = 5
        sdma_bases = [0x4980, 0x6180, 0x78000, 0x79000, 0x7A000]
        sdma_sizes = [0x1000] * 5

        self.device.sdmas = shader._create_sdmas(sdma_bases, sdma_sizes)

        # Setup the Command Processor's PM4 engines.
        pm4_starts = [0xC000]
        pm4_ends = [0xD000]

        self.device.pm4_pkt_procs = shader._create_pm4s(pm4_starts, pm4_ends)

    def get_driver_command(self, debug: bool = False):
        debug_commands = "dmesg -n8\nuname -r\nlspci -v\n" if debug else ""

        driver_load_command = (
            "export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            "export HSA_ENABLE_INTERRUPT=0\n"
            "export HCC_AMDGPU_TARGET=gfx90a\n"
            "echo 0 > /proc/sys/kernel/randomize_va_space\n"
            f"{debug_commands}\n"
            "dd if=/root/roms/mi200.rom of=/dev/mem bs=1k seek=768 count=128\n"
            "if [ -f /home/gem5/load_amdgpu.sh ]; then\n"
            "    sh /home/gem5/load_amdgpu.sh\n"
            "elif [ ! -f /lib/modules/`uname -r`/updates/dkms/amdgpu.ko ]; then\n"
            '    echo "ERROR: Missing DKMS package for kernel `uname -r`. Exiting gem5."\n'
            "    m5 exit\n"
            "else\n"
            "    echo 'options amdgpu ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0' > /etc/modprobe.d/amdgpu.conf\n"
            "    modprobe -v amdgpu\n"
            "fi\n"
        )

        return driver_load_command


# Validated single-XCD MI300X (gpu-microbenchmarks
# configs/gem5/mi300x-validated.yaml, the GAP project's calibration against a
# real MI300X; flag names in the comments). MI300X uses these values; MI355X
# keeps the generic VIPER defaults.
MI300X_GPU_CLOCK = "2100MHz"  # --gpu-clock (CUs, TLBs, TCP/SQC/scalar, TCC)
MI300X_FABRIC_CLOCK = "1800MHz"  # --fabric-clock (network, directories/L3)
MI300X_MEMORY_CLOCK = "1300MHz"  # --memory-clock (HBM controllers)
# --sys-clock: the GPU device's own engines (SDMA, PM4, interrupt handler,
# memory manager) and the GPU DMA controllers. Only the GPU device subtree
# gets this domain: the board clock also drives the guest's timers, which
# must stay consistent with the KVM CPU clock.
MI300X_DEVICE_CLOCK = "1GHz"
MI300X_L3_SIZE_MIB = 64  # --dgpu-l3-size, split over the directories


def MI300XMemory() -> HighBandwidthMemory:
    """GPU memory of the validated MI300X: 16 HBM3 channels (one directory
    each), two pseudo channels per channel, 16 GiB.

    --dgpu-mem-type HBM3_MI300X_1x64 --dgpu-num-dirs 16 --dgpu-mem-size 16GiB
    --hbm-ctrl. The legacy config interleaves 128 B granules (one line):
    address bit 7 selects the pseudo channel, bits 8-11 the channel.
    """
    return HighBandwidthMemory(
        HBM3_MI300X_1x64,
        16,
        256,
        size="16GiB",
        pseudo_channel_interleaving_size=128,
    )


# Defaults to a single "XCD" (i.e., 1/8th of a full MI300X).
class MI300X(BaseViperGPU):
    # Apply the validated MI300X structure and parameters. Subclasses for
    # other devices turn this off.
    _validated_mi300x = True

    def __init__(
        self,
        gpu_memory: Optional[AbstractMemorySystem] = None,
        num_cus: int = 38,  # --num-compute-units
        cu_per_sqc: int = 2,  # --cu-per-sqc
        tcp_size: str = "32KiB",  # --tcp-size
        tcp_assoc: int = 16,
        sqc_size: str = "64KiB",  # --sqc-size
        sqc_assoc: int = 8,
        scalar_size: str = "16KiB",  # --scalar-size
        scalar_assoc: int = 8,
        tcc_size: str = "256KiB",  # per TCC: --tcc-size 4MB / --num-tccs 16
        tcc_assoc: int = 16,  # --tcc-assoc
        tcc_count: int = 16,  # --num-tccs
        cache_line_size: int = 128,  # --cacheline_size
        ipt_binary: str = "",
        debug: bool = False,
        cu_per_scalar_cache: int = 2,  # --cu-per-scalar-cache
    ):
        """
        :param gpu_memory: GPU memory. Defaults to MI300XMemory().
        """
        if gpu_memory is None:
            gpu_memory = MI300XMemory()
        super().__init__(
            gpu_memory=gpu_memory, ipt_binary=ipt_binary, debug=debug
        )

        self._cu_per_sqc = cu_per_sqc
        self._cu_per_scalar_cache = cu_per_scalar_cache
        self._tcp_size = tcp_size
        self._tcp_assoc = tcp_assoc
        self._sqc_size = sqc_size
        self._sqc_assoc = sqc_assoc
        self._scalar_size = scalar_size
        self._scalar_assoc = scalar_assoc
        self._tcc_size = tcc_size
        self._tcc_assoc = tcc_assoc
        self._tcc_count = tcc_count
        self._cache_line_size = cache_line_size

        self.device.device_name = "MI300X"

        self.device.DeviceID = 0x74A1
        self.device.SubsystemVendorID = 0x1002
        self.device.SubsystemID = 0x0C34
        self.device.BAR5 = PciMemBar(size="2MiB")

        # Setup device-specific address ranges for various SoC components.
        if self._validated_mi300x:
            # Legacy TLB hierarchy (GPUTLBConfig): per-CU vector L1 TLBs,
            # SQC and scalar TLBs shared like the caches, and one L2 TLB that
            # is the last level and owns the page table walker.
            tlb_options = dict(
                l3_tlb=False,
                cu_per_sqc_tlb=cu_per_sqc,
                cu_per_scalar_tlb=cu_per_scalar_cache,
            )
        else:
            tlb_options = {}
        shader = ViperShader(
            self._my_id,
            num_cus,
            cache_line_size,
            self.device,
            gpu_memory.get_size(),
            **tlb_options,
        )
        self.set_shader(shader)

        num_sdmas = 16
        sdma_bases = [
            0x4980,
            0x6180,
            0x65000,
            0x66000,
            0x84980,
            0x86180,
            0xE5000,
            0xE6000,
            0x104980,
            0x106180,
            0x165000,
            0x166000,
            0x184980,
            0x186180,
            0x1E5000,
            0x1E6000,
        ]
        sdma_sizes = [0x1000] * num_sdmas

        self.device.sdmas = shader._create_sdmas(sdma_bases, sdma_sizes)

        # Setup the Command Processor's PM4 engines.
        pm4_starts = [
            0xC000,
            0x4C000,
            0x8C000,
            0xCC000,
            0x10C000,
            0x14C000,
            0x18C000,
            0x1CC000,
        ]
        pm4_ends = [
            0xD000,
            0x4D000,
            0x8D000,
            0xCD000,
            0x10D000,
            0x14D000,
            0x18D000,
            0x1CD000,
        ]

        self.device.pm4_pkt_procs = shader._create_pm4s(pm4_starts, pm4_ends)

        # The cache hierarchy gives every GPU DMA port a DMA controller, so
        # create it after the SDMA engines (their page table walkers).
        self.setup_caches()
        if self._validated_mi300x:
            self._apply_validated_parameters()

    def _cache_hierarchy_options(self) -> dict:
        if not self._validated_mi300x:
            return {}
        gpu_hz = float(MI300X_GPU_CLOCK.rstrip("MHz"))
        fabric_hz = float(MI300X_FABRIC_CLOCK.rstrip("MHz"))
        return dict(
            cu_per_scalar_cache=self._cu_per_scalar_cache,
            fabric_clock=MI300X_FABRIC_CLOCK,
            gpu_clock=MI300X_GPU_CLOCK,
            # --gpu-topology Crossbar (set by mi300.py), --link-width-bits
            # 1024, --gpu-dir-link-width-bits 512; XCD-side links scaled by
            # gpu/fabric clock as in DisjointNetwork.connectGPU.
            network_factory=lambda ruby_system: SimpleCrossbar(
                ruby_system,
                link_width_bytes=1024 // 8,
                dir_link_width_bytes=512 // 8,
                xcd_clock_ratio=gpu_hz / fabric_hz,
            ),
            num_virtual_networks=11,  # Disjoint_VIPER
        )

    def _apply_validated_parameters(self):
        """Set the validated MI300X parameters on the created objects."""
        num_cus = len(self.shader.CUs)
        gpu_clk = SrcClockDomain(
            clock=MI300X_GPU_CLOCK, voltage_domain=VoltageDomain()
        )

        # CU harvesting: an MI300X XCC has 40 CUs (4 SEs x 10) with 2 fused
        # off; the driver reads the fused mask per SE. Which CUs are fused
        # differs per chip; CU 9 of SE0 and SE2 is assumed.
        if num_cus == 38:
            self.device.inactive_cus = [1 << 9, 0, 1 << 9, 0]

        # Shader, CUs, dispatcher and all TLBs run on the GPU clock.
        self.shader.clk_domain = gpu_clk

        # Device engines and GPU DMA controllers on the system clock.
        device_clk = SrcClockDomain(
            clock=MI300X_DEVICE_CLOCK, voltage_domain=VoltageDomain()
        )
        self.device.clk_domain = device_clk
        for ctrl in self.gpu_caches._dma_controllers:
            ctrl.clk_domain = device_clk
        for cu in self.shader.CUs:
            cu.num_scalar_cores = 4  # --scalar-cores-per-cu
            cu.scalar_issue_period = 4  # --scalar-issue-period
            cu.valu_dual_issue = True  # --valu-dual-issue
            cu.lds_req_latency = 55  # --lds-req-latency
            cu.mem_req_latency = 62  # --mem-req-latency
            cu.mem_resp_latency = 62  # --mem-resp-latency
            cu.scalar_mem_req_latency = 40  # --scalar-mem-req-latency
            cu.scalar_mem_resp_latency = 0  # legacy default
            cu.max_cu_tokens = 160  # --max-cu-tokens
            cu.sqc_bus_latency = 1  # legacy default
            cu.coalescer_to_vrf_bus_width = 32  # --glbmem-rd-bus-width
            cu.vrf_to_coalescer_bus_width = 32  # --glbmem-wr-bus-width
            for srf in cu.scalar_register_file:
                srf.num_regs = 3200  # --sreg-file-size
            for pool in cu.register_manager.srf_pool_managers:
                pool.pool_size = 3200

        # TLBs: --L1TLBentries/assoc 32/32, --L1AccessLatency 1,
        # --L1MissLatency 1; --L2TLBentries/assoc 256/64,
        # --L2AccessLatency 100, --L2MissLatency 15.
        l1_tlbs = [cu.l1_tlb for cu in self.shader.CUs]
        l1_tlbs += list(self.shader.sqc_tlb) + list(self.shader.scalar_tlb)
        for tlb in l1_tlbs:
            tlb.size = 32
            tlb.assoc = 32
            tlb.hitLatency = 1
            tlb.missLatency1 = 1
            tlb.missLatency2 = 1
        l2_tlb = self.shader.l2_tlb
        l2_tlb.size = 256
        l2_tlb.assoc = 64
        l2_tlb.hitLatency = 100
        l2_tlb.missLatency1 = 15
        l2_tlb.missLatency2 = 15

        ruby = self.gpu_caches.ruby_gpu

        # Vector L1 (TCP)
        for tcp in ruby.tcp_controllers:
            tcp.L1cache.tagAccessLatency = 1  # --TCP_latency
            tcp.L1cache.dataAccessLatency = 1
            tcp.issue_latency = 25  # --tcp-issue-latency
            tcp.mandatory_queue_latency = 1  # --mandatory_queue_latency
            tcp.number_of_TBEs = 2560
            tcp.coalescer.max_coalesces_per_cycle = 10
            tcp.coalescer.deadlock_threshold = 2000000

        # Instruction (SQC) and scalar caches
        for ctrl in list(ruby.sqc_controllers) + list(
            ruby.scalar_controllers
        ):
            ctrl.issue_latency = 80
            ctrl.sequencer.deadlock_threshold = 2000000

        # L2 (TCC)
        for tcc in ruby.tcc_controllers:
            tcc.WB = True  # --WB_L2
            tcc.l2_request_latency = 101  # --gpu-to-dir-latency
            tcc.l2_response_latency = 15  # --TCC_latency
            tcc.glc_atomic_latency = 302  # --glc-atomic-latency
            tcc.number_of_TBEs = 2560 * num_cus
            tcc.L2cache.tagAccessLatency = 1  # --tcc-tag-access-latency
            tcc.L2cache.dataAccessLatency = 1  # --tcc-data-access-latency
            tcc.L2cache.tagArrayBanks = 1  # --tcc-num-banks
            tcc.L2cache.dataArrayBanks = 1
            tcc.L2cache.atomicLatency = 42  # --atomic-alu-latency
            # --tcc-num-atomic-alus 48, split over the TCCs
            tcc.L2cache.atomicALUs = 48 // len(ruby.tcc_controllers)

        # Directories with the Infinity Cache as a memory-side L3
        dirs = list(ruby.directory_controllers)
        for d in dirs:
            d.response_latency = 30
            d.L2isWB = True  # --WB_L2
            d.l3AllocOnRead = True  # --infinity-cache
            d.l3KeepOnReadHit = True
            d.to_memory_controller_latency = 96  # --dir-to-mem-ctrl-latency
            d.L3CacheMemory.size = f"{MI300X_L3_SIZE_MIB // len(dirs)}MiB"
            d.L3CacheMemory.assoc = 16
            d.L3CacheMemory.dataAccessLatency = 20
            d.L3CacheMemory.tagAccessLatency = 15
            d.l3_hit_latency = 20

        # HBM controllers (--hbm-ctrl): memory clock, 128-entry queues.
        mem_clk = SrcClockDomain(
            clock=MI300X_MEMORY_CLOCK, voltage_domain=VoltageDomain()
        )
        for ctrl in self.memory.get_memory_controllers():
            ctrl.clk_domain = mem_clk
            ctrl.disable_sanity_check = False
            if hasattr(ctrl, "dram_2"):
                for intf in (ctrl.dram, ctrl.dram_2):
                    intf.read_buffer_size = 128
                    intf.write_buffer_size = 128

    def get_driver_command(self):
        driver_load_command = (
            "export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            "export HSA_ENABLE_INTERRUPT=0\n"
            "export HCC_AMDGPU_TARGET=gfx942\n"
            "echo 0 > /proc/sys/kernel/randomize_va_space\n"
            f"{self._debug_commands}\n"
            "dd if=/root/roms/mi300.rom of=/dev/mem bs=1k seek=768 count=128\n"
            # Check if exists (backwards compat with ROCm <7.0)
            "if [ -e /usr/lib/firmware/amdgpu/mi300_discovery ]; then\n"
            "    rm -f /usr/lib/firmware/amdgpu/ip_discovery.bin\n"
            "    ln -s /usr/lib/firmware/amdgpu/mi300_discovery /usr/lib/firmware/amdgpu/ip_discovery.bin\n"
            "fi\n"
            "if [ -f /home/gem5/load_amdgpu.sh ]; then\n"
            # Last stable support
            "    sh /home/gem5/load_amdgpu.sh\n"
            "else\n"
            # develop support
            f"    echo 'options amdgpu ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0 discovery={self._discovery_value}' > /etc/modprobe.d/amdgpu.conf\n"
            "    modprobe -v amdgpu\n"
            "fi\n"
        )

        return driver_load_command


# Defaults to a single "XCD" (i.e., 1/8th of a full MI355X).
class MI355X(MI300X):
    # Not calibrated: keep the generic VIPER structure and parameters.
    _validated_mi300x = False

    def __init__(
        self,
        gpu_memory: AbstractMemorySystem,
        num_cus: int = 40,
        cu_per_sqc: int = 4,
        tcp_size: str = "16KiB",
        tcp_assoc: int = 16,
        sqc_size: str = "32KiB",
        sqc_assoc: int = 8,
        scalar_size: str = "32KiB",
        scalar_assoc: int = 8,
        tcc_size: str = "256KiB",
        tcc_assoc: int = 16,
        tcc_count: int = 16,
        cache_line_size: int = 64,
        ipt_binary: str = "",
        debug: bool = False,
    ):
        super().__init__(
            gpu_memory=gpu_memory,
            num_cus=num_cus,
            cu_per_sqc=cu_per_sqc,
            tcp_size=tcp_size,
            tcp_assoc=tcp_assoc,
            sqc_size=sqc_size,
            sqc_assoc=sqc_assoc,
            scalar_size=scalar_size,
            scalar_assoc=scalar_assoc,
            tcc_size=tcc_size,
            tcc_assoc=tcc_assoc,
            tcc_count=tcc_count,
            cache_line_size=cache_line_size,
            ipt_binary=ipt_binary,
            debug=debug,
        )

        self.device.DeviceID = 0x75A0
        self.device.device_name = "MI355X"

    def get_driver_command(self):
        driver_load_command = (
            "export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH\n"
            "export HSA_ENABLE_INTERRUPT=0\n"
            "export HCC_AMDGPU_TARGET=gfx950\n"
            "echo 0 > /proc/sys/kernel/randomize_va_space\n"
            f"{self._debug_commands}\n"
            "dd if=/root/roms/mi300.rom of=/dev/mem bs=1k seek=768 count=128\n"
            "if [ -e /usr/lib/firmware/amdgpu/mi350_discovery ]; then\n"
            "    rm -f /usr/lib/firmware/amdgpu/ip_discovery.bin\n"
            "    ln -s /usr/lib/firmware/amdgpu/mi350_discovery /usr/lib/firmware/amdgpu/ip_discovery.bin\n"
            "else\n"
            '    echo "ERROR: ROCm 7.0+ disk image required for MI355X. Exiting gem5."\n'
            "    m5 exit\n"
            "fi\n"
            "if [ -f /home/gem5/load_amdgpu.sh ]; then\n"
            # Last stable support
            "    sh /home/gem5/load_amdgpu.sh\n"
            "else\n"
            # develop support
            f"    echo 'options amdgpu ip_block_mask=0x6f ppfeaturemask=0 dpm=0 audio=0 ras_enable=0 discovery={self._discovery_value}' > /etc/modprobe.d/amdgpu.conf\n"
            "    modprobe -v amdgpu\n"
            "fi\n"
        )

        return driver_load_command
