# MI300X memory/compute partitioning in gem5 FS mode

This directory has what you need to reproduce the MI300X partitioning
experiments on the `memory-partition` branch of
<https://github.com/hal-uw/gap-gem5>: boot the GPU full-system image, then
use `rocm-smi` to query and change the compute partition (SPX/CPX) and the
memory partition (NPS1/NPS4).

| File | Purpose |
|------|---------|
| `x86-mi300x-gpu-partition.py` | gem5 run config (MI300X, KVM CPU, `debug=True`, IP discovery binary, guest console mirrored into gem5 stdout) |
| `bash.sh` | Guest "application": starts an interactive shell on the serial terminal |
| `drive_term.py` | Optional. Types the `rocm-smi` commands for you instead of `gem5term` |

Part A covers building gem5 and running the experiment. Part B covers
changing the amdgpu kernel driver (for example, adding debug prints) and
installing it into the disk image.

---

## Part A: build and run gem5

### A.1 Requirements

- An x86 Linux host with KVM access (`/dev/kvm` must be readable and
  writable by you). Lab machine used: eldin.cs.wisc.edu.
- Enough free host RAM for the guest: 8 GiB of DRAM plus 16 GiB of GPU
  HBM.
- Read access to the resources below. They currently live under
  `/nobackup2/cassiesu/` on eldin. On another machine, copy them and pass
  your own paths.

| Resource | Path |
|----------|------|
| Disk image (ROCm 7.2, has the 6.17 kernel modules) | `/nobackup2/cassiesu/gpu_fs/gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-rocm72` |
| Guest kernel | `/nobackup2/cassiesu/gpu_fs/gem5-resources/src/x86-ubuntu-gpu-ml/vmlinux-rocm72` |
| MI300X IP discovery binary | `/nobackup2/cassiesu/gpu_fs/gem5-resources/src/x86-ubuntu-gpu-ml/files/mi300_discovery` |

The disk image is about 55 GB. **Do not run two simulations from the same
image at the same time,** and do not run one while the image is mounted by
`systemd-nspawn` (see Part B). Each user should keep their own copy if
they plan to modify it.

### A.2 Get the code and build

```sh
git clone https://github.com/hal-uw/gap-gem5.git gem5
cd gem5
git checkout memory-partition
scons build/VEGA_X86/gem5.opt -j $(nproc)
```

A warning about Capstone at the end of the build is harmless.

The branch must include commit
`dev-amdgpu: Reset SDMA gfx rptr when RB_RPTR is written`. Without it, the
driver reload after `--setmemorypartition NPS4` fails early with
`ring sdma0.1 test failed (-110)` instead of reaching the expected state
described in A.5. Check with:

```sh
git log --oneline | grep "Reset SDMA gfx rptr"
```

### A.3 Run gem5

Run from the gem5 root directory. The output goes to a file because it is
large (hundreds of MB with the debug flags below).

```sh
mkdir -p output
DIR=configs/example/gem5_library/mi300x-memory-partition
RES=/nobackup2/cassiesu/gpu_fs/gem5-resources/src/x86-ubuntu-gpu-ml

build/VEGA_X86/gem5.opt -d m5out \
  --debug-flags=GPUCommandProc,AMDGPUDevice,SDMAEngine,PM4PacketProcessor,HSAPacketProcessor,GPUALL,PciDevice \
  $DIR/x86-mi300x-gpu-partition.py \
  --image=$RES/disk-image/x86-ubuntu-rocm72 \
  --kernel=$RES/vmlinux-rocm72 \
  --app=$DIR/bash.sh \
  > output/output_run.txt
```

Notes:

- Use a different `-d` directory for each run if you want to keep old
  `m5out` contents.
- `--ipt-binary=<path>` overrides the IP discovery binary location.
- If you start gem5 without a terminal on stdin (from a script, `nohup`,
  etc.), add `--listener-mode=on` before the config path. Otherwise gem5
  does not open the terminal port.
- Fewer debug flags give a much smaller output file. Keep at least
  `AMDGPUDevice,SDMAEngine,PM4PacketProcessor` when looking at partitioning.
- The config sets `debug=True` on the MI300X. It also sets
  `log_buf_len=4M` and `com_1.device.outfile = "stdoutput"` so the guest
  console (dmesg and shell) is interleaved with the gem5 debug output in
  `output/output_run.txt`.

### A.4 Connect to the guest and run the partition commands

Booting the guest and loading amdgpu takes a while in simulation. Wait
for the shell prompt.

In a second shell on the same host:

```sh
cd gem5
make -C util/term          # first time only, builds gem5term
./util/term/gem5term 3456
```

`3456` is the first terminal port. If another simulation already uses it,
check the actual port in the line
`board.pc.com_1.device: Listening for connections on port ...`. gem5
prints this line to stderr, so it shows in your shell, not in the output
file.

When the `bash-5.2#` prompt appears, type:

```sh
rocm-smi --showcomputepartition
rocm-smi --showmemorypartition
rocm-smi --setmemorypartition NPS4      # answer Y at the [Y/N] prompt
rocm-smi --showmemorypartition
rocm-smi --setcomputepartition CPX
rocm-smi --showcomputepartition
exit
```

The `--setmemorypartition` command asks for confirmation. Answer `Y`
promptly. The driver reload that follows takes a long time in simulation.
`exit` ends the shell, and the simulation exits on its own.

**Without typing:** `drive_term.py` waits for the prompt, sends the
commands above in order, answers `Y`, and logs everything to a file.
Start it right after gem5 (it retries until the port opens):

```sh
python3 $DIR/drive_term.py 3456 output/term_run.log
```

To change the commands, edit the `COMMANDS` list in the script.

### A.5 Expected output (current state of the work)

This is what a correct reproduction looks like today. The NPS4 part is the
**open problem** being worked on.

1. `--showcomputepartition` reports `SPX` and `--showmemorypartition`
   reports `NPS1`.
2. `--setmemorypartition NPS4`: the gem5 output contains
   `PSP: GFX_CMD_ID_FB_NPS_MODE processed. Mode: 0x4`, and the guest prints
   `NPS mode change request done, reload driver`.
3. The driver unloads and reloads. All SDMA ring tests pass. KFD comes back
   with **one** node instead of four:
   ```
   Total number of KFD nodes to be created: 1
   KFD node 0 partition 0 size 16383M
   added device 1002:74a1
   ```
4. rocm-smi then prints
   `GPU[0] : Issue reloading driver, please check dmsg for errors`.
5. After that, `--showmemorypartition` prints
   `Failed to retrieve current memory partition, even though device supports it.`,
   `--setcomputepartition CPX` prints
   `Requested setting (CPX) is unavailable for current device`, and
   `--showcomputepartition` still reports `SPX`.

Useful greps on the output file:

```sh
grep -nE "FB_NPS_MODE|NPS mode change|KFD node|Issue reloading|ring .* test failed|probe with driver" output/output_run.txt
```

If you see `ring sdma0.X test failed (-110)` followed by
`probe with driver amdgpu failed with error -110`, your build is missing
the SDMA rptr fix (see A.2).

### A.6 Troubleshooting

- `TypeError: SimpleProcessor.__init__() missing 1 required positional
  argument: 'clk_freq'`: you are using an older copy of the config with a
  newer gem5. The config in this directory already passes `clk_freq`.
- No terminal port / `gem5term` cannot connect: add
  `--listener-mode=on` (see A.3) and check the port in the gem5 output.
- `Permission denied` on `/dev/kvm`: ask the admin to add you to the
  `kvm` group.
- KVM runs without perf counters by default. `--kvm-perf` enables them,
  which changes simulated timing. The expected output above was collected
  **without** `--kvm-perf`.

---

## Part B: modify and rebuild the amdgpu driver

Use this to add debug prints to the guest amdgpu driver and install the
rebuilt modules into the disk image. These steps need `sudo` on the host
for `systemd-nspawn`.

Paths used below (on eldin):

| What | Path |
|------|------|
| Linux 6.17 source tree that is built and installed | `/nobackup2/cassiesu/linux_debug_build` |
| Disk image directory | `/nobackup2/cassiesu/gpu_fs/gem5-resources/src/x86-ubuntu-gpu-ml/disk-image` |

> **Note:** there is also a separate tree at `/nobackup2/cassiesu/linux`.
> It is a different checkout, so edits made there are **not** picked up by
> the build in step B.3. Make your edits in `linux_debug_build`, the tree
> mounted into the container, or copy them over before building.

### B.1 Edit the driver (on the host)

```sh
cd /nobackup2/cassiesu/linux_debug_build
vim drivers/gpu/drm/amd/amdgpu/amdgpu_psp.c
```

Example print:

```c
dev_info(psp->adev->dev,
         "PSP DEBUG: new debug print");
```

Verify the edit:

```sh
grep -n "new debug print" drivers/gpu/drm/amd/amdgpu/amdgpu_psp.c
```

Expected output (the line number will vary):

```
838: dev_info(psp->adev->dev,"PSP DEBUG: new debug print");
```

### B.2 Enter the disk image environment

Make sure no gem5 simulation is using the image, then start the container
from the disk image directory:

```sh
cd /nobackup2/cassiesu/gpu_fs/gem5-resources/src/x86-ubuntu-gpu-ml/disk-image
sudo systemd-nspawn \
  -i x86-ubuntu-rocm72 \
  --bind=/nobackup2/cassiesu/linux_debug_build:/mnt
```

Inside the container, go to the source tree:

```sh
cd /mnt
```

### B.3 Use the correct kernel config

The gem5 guest kernel is Linux 6.17. Check:

```sh
grep UTS_RELEASE include/generated/utsrelease.h
```

Expected:

```
#define UTS_RELEASE "6.17.0"
```

Copy the Ubuntu config:

```sh
cp /boot/config-6.17.0-35-generic .config
```

### B.4 Fix the Ubuntu certificate issue

Building upstream Linux with an Ubuntu config can fail with:

```
No rule to make target 'debian/canonical-certs.pem'
```

Fix:

```sh
grep CONFIG_SYSTEM_TRUSTED_KEYS .config
sed -i 's|CONFIG_SYSTEM_TRUSTED_KEYS=.*|CONFIG_SYSTEM_TRUSTED_KEYS=""|' .config

grep CONFIG_SYSTEM_REVOCATION_KEYS .config
sed -i 's|CONFIG_SYSTEM_REVOCATION_KEYS=.*|CONFIG_SYSTEM_REVOCATION_KEYS=""|' .config
```

### B.5 Generate the config

```sh
yes "" | make oldconfig
```

This fills any missing kernel options with their defaults.

### B.6 Build the kernel and modules

Still inside `/mnt` in the container:

```sh
make -j$(nproc)
```

A successful build ends with many lines like:

```
INSTALL ...
SIGN ...
DEPMOD /lib/modules/6.17.0
```

### B.7 Install the modules into the disk image

**This step is critical.**

```sh
make modules_install
```

This installs `/lib/modules/6.17.0/` inside the disk image. Verify:

```sh
find /lib/modules -name 'amdgpu.ko*'
```

Expected:

```
/lib/modules/6.17.0/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
```

### B.8 Check that the installed module has your debug print

Inside the container:

```sh
strings /lib/modules/6.17.0/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko \
  | grep "new debug print"
```

Expected:

```
amdgpu: PSP DEBUG: new debug print
```

### B.9 Exit the container cleanly

**Important:** run

```sh
exit
```

This makes sure the disk image changes are synced. Do not start gem5
until the container has exited.

### B.10 Run gem5

From the gem5 root on the host, run the same command as in A.3. Use a new
output file name so you can compare it with earlier runs:

```sh
build/VEGA_X86/gem5.opt -d m5out \
  --debug-flags=GPUCommandProc,AMDGPUDevice,SDMAEngine,PM4PacketProcessor,HSAPacketProcessor \
  $DIR/x86-mi300x-gpu-partition.py \
  --image=$RES/disk-image/x86-ubuntu-rocm72 \
  --kernel=$RES/vmlinux-rocm72 \
  --app=$DIR/bash.sh \
  > output/output_new.txt
```

`vmlinux-rocm72` is the 6.17 kernel the modules are built against. If you
change the kernel config in a way that affects the kernel image itself,
and not just the amdgpu module, the guest kernel must be rebuilt and
replaced too. Otherwise the new modules may fail to load.

### B.11 Check that gem5 loaded the new driver

Inside the guest (through `gem5term`), check the kernel:

```sh
uname -r
```

Expected: `6.17.0`

Check the amdgpu module:

```sh
modinfo amdgpu | grep filename
```

Expected:

```
/lib/modules/6.17.0/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
```

and **not**

```
/lib/modules/6.8.0-136-generic/updates/dkms/amdgpu.ko.zst
```

### B.12 Check the debug output

On the host:

```sh
grep "new debug print" output/output_new.txt
```
