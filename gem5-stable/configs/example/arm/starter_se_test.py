# Copyright (c) 2016-2017, 2022-2023 Arm Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""This script is the syscall emulation example script from the ARM
Research Starter Kit on System Modeling. More information can be found
at: http://www.arm.com/ResearchEnablement/SystemModeling
"""

import argparse
import os
import shlex
from os.path import join as joinpath

import m5
from m5.objects import *
from m5.util import addToPath

m5.util.addToPath("../..")

import devices
from common import (
    MemConfig,
    ObjectList,
)
from common.cores.arm import (
    HPI,
    O3_ARM_v7a,
)

# Pre-defined CPU configurations. Each tuple must be ordered as : (cpu_class,
# l1_icache_class, l1_dcache_class, l2_Cache_class). Any of
# the cache class may be 'None' if the particular cache is not present.
cpu_types = {
    "atomic": (AtomicSimpleCPU, None, None, None),
    "minor": (MinorCPU, devices.L1I, devices.L1D, devices.L2),
    "hpi": (HPI.HPI, HPI.HPI_ICache, HPI.HPI_DCache, HPI.HPI_L2),
    "o3": (
        O3_ARM_v7a.O3_ARM_v7a_3,
        O3_ARM_v7a.O3_ARM_v7a_ICache,
        O3_ARM_v7a.O3_ARM_v7a_DCache,
        O3_ARM_v7a.O3_ARM_v7aL2,
    ),
}


def get_processes(cmd):
    """Interprets commands to run and returns a list of processes"""

    cwd = os.getcwd()
    multiprocesses = []
    for idx, c in enumerate(cmd):
        argv = shlex.split(c)

        process = Process(pid=100 + idx, cwd=cwd, cmd=argv, executable=argv[0])
        process.gid = os.getgid()

        print("info: %d. command and arguments: %s" % (idx + 1, process.cmd))
        multiprocesses.append(process)

    return multiprocesses

# Set up environment for taking SimPoint checkpoints
# Expecting SimPoint files generated by SimPoint 3.2
def parseSimpointAnalysisFile(options, testsys):
    import re

    (
        simpoint_filename,
        weight_filename,
        interval_length,
        warmup_length,
    ) = options.take_simpoint_checkpoints.split(",", 3)
    print("simpoint analysis file:", simpoint_filename)
    print("simpoint weight file:", weight_filename)
    print("interval length:", interval_length)
    print("warmup length:", warmup_length)

    interval_length = int(interval_length)
    warmup_length = int(warmup_length)

    # Simpoint analysis output starts interval counts with 0.
    simpoints = []
    simpoint_start_insts = []

    # Read in SimPoint analysis files
    simpoint_file = open(simpoint_filename)
    weight_file = open(weight_filename)
    while True:
        line = simpoint_file.readline()
        if not line:
            break
        m = re.match(r"(\d+)\s+(\d+)", line)
        if m:
            interval = int(m.group(1))
        else:
            fatal("unrecognized line in simpoint file!")

        line = weight_file.readline()
        if not line:
            fatal("not enough lines in simpoint weight file!")
        m = re.match(r"([0-9\.e\-]+)\s+(\d+)", line)
        if m:
            weight = float(m.group(1))
        else:
            fatal("unrecognized line in simpoint weight file!")

        if interval * interval_length - warmup_length > 0:
            starting_inst_count = interval * interval_length - warmup_length
            actual_warmup_length = warmup_length
        else:
            # Not enough room for proper warmup
            # Just starting from the beginning
            starting_inst_count = 0
            actual_warmup_length = interval * interval_length

        simpoints.append(
            (interval, weight, starting_inst_count, actual_warmup_length)
        )

    # Sort SimPoints by starting inst count
    simpoints.sort(key=lambda obj: obj[2])
    for s in simpoints:
        interval, weight, starting_inst_count, actual_warmup_length = s
        print(
            str(interval),
            str(weight),
            starting_inst_count,
            actual_warmup_length,
        )
        simpoint_start_insts.append(starting_inst_count)

    print("Total # of simpoints:", len(simpoints))
    print(simpoint_start_insts)
    for i in range(options.num_cores):
        testsys.cpu_cluster.cpus[i].simpoint_start_insts = simpoint_start_insts

    return (simpoints, interval_length)


def takeSimpointCheckpoints(simpoints, interval_length, cptdir):
    num_checkpoints = 0
    index = 0
    last_chkpnt_inst_count = -1
    for simpoint in simpoints:
        interval, weight, starting_inst_count, actual_warmup_length = simpoint
        if starting_inst_count == last_chkpnt_inst_count:
            # checkpoint starting point same as last time
            # (when warmup period longer than starting point)
            exit_cause = "simpoint starting point found"
            code = 0
        else:
            exit_event = m5.simulate()

            # print("sim stop")
            # skip checkpoint instructions should they exist
            while exit_event.getCause() == "checkpoint":
                print("Found 'checkpoint' exit event...ignoring...")
                exit_event = m5.simulate()

            exit_cause = exit_event.getCause()
            code = exit_event.getCode()

        if exit_cause == "simpoint starting point found":
            m5.checkpoint(
                joinpath(
                    cptdir,
                    "cpt.simpoint_%02d_inst_%d_weight_%f_interval_%d_warmup_%d"
                    % (
                        index,
                        starting_inst_count,
                        weight,
                        interval_length,
                        actual_warmup_length,
                    ),
                )
            )
            print(
                "Checkpoint #%d written. start inst:%d weight:%f"
                % (num_checkpoints, starting_inst_count, weight)
            )
            num_checkpoints += 1
            last_chkpnt_inst_count = starting_inst_count
        else:
            break
        index += 1

    print("Exiting @ tick %i because %s" % (m5.curTick(), exit_cause))
    print("%d checkpoints taken" % num_checkpoints)
    sys.exit(code)


def restoreSimpointCheckpoint():
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()

    if exit_cause == "simpoint starting point found":
        print("Warmed up! Dumping and resetting stats!")
        m5.stats.dump()
        m5.stats.reset()

        exit_event = m5.simulate()
        exit_cause = exit_event.getCause()

        if exit_cause == "simpoint starting point found":
            print("Done running SimPoint!")
            sys.exit(exit_event.getCode())

    print("Exiting @ tick %i because %s" % (m5.curTick(), exit_cause))
    sys.exit(exit_event.getCode())
    
def create(args):
    """Create and configure the system object."""

    cpu_class = cpu_types[args.cpu][0]
    mem_mode = cpu_class.memory_mode()
    # Only simulate caches when using a timing CPU (e.g., the HPI model)
    want_caches = True if mem_mode == "timing" else False

    system = devices.SimpleSeSystem(
        mem_mode=mem_mode,
    )

    # Add CPUs to the system. A cluster of CPUs typically have
    # private L1 caches and a shared L2 cache.
    system.cpu_cluster = devices.ArmCpuCluster(
        system,
        args.num_cores,
        args.cpu_freq,
        "1.2V",
        *cpu_types[args.cpu],
        tarmac_gen=args.tarmac_gen,
        tarmac_dest=args.tarmac_dest,
    )
    
    if args.maxinsts:
        for i in range(args.num_cores):
            system.cpu_cluster.cpus[i].max_insts_any_thread = args.maxinsts

    if args.simpoint_profile:
        for i in range(args.num_cores):
            system.cpu_cluster.cpus[i].addSimPointProbe(args.simpoint_interval)
    
    simpoints = []
    interval_length = -1
    if args.take_simpoint_checkpoints != None:
        simpoints, interval_length = parseSimpointAnalysisFile(
            args, system
        )
    
    # Create a cache hierarchy for the cluster. We are assuming that
    # clusters have core-private L1 caches and an L2 that's shared
    # within the cluster.
    system.addCaches(want_caches, last_cache_level=2)

    # Tell components about the expected physical memory ranges. This
    # is, for example, used by the MemConfig helper to determine where
    # to map DRAMs in the physical address space.
    system.mem_ranges = [AddrRange(start=0, size=args.mem_size)]

    # Configure the off-chip memory system.
    MemConfig.config_mem(args, system)

    # Wire up the system's memory system
    system.connect()

    # Parse the command line and get a list of Processes instances
    # that we can pass to gem5.
    processes = get_processes(args.commands_to_run)
    if len(processes) != args.num_cores:
        print(
            "Error: Cannot map %d command(s) onto %d CPU(s)"
            % (len(processes), args.num_cores)
        )
        sys.exit(1)

    system.workload = SEWorkload.init_compatible(processes[0].executable)

    # Assign one workload to each CPU
    for cpu, workload in zip(system.cpu_cluster.cpus, processes):
        cpu.workload = workload

    return system, simpoints, interval_length


def main():
    parser = argparse.ArgumentParser(epilog=__doc__)

    parser.add_argument(
        "commands_to_run",
        metavar="command(s)",
        nargs="*",
        help="Command(s) to run",
    )
    parser.add_argument(
        "--cpu",
        type=str,
        choices=list(cpu_types.keys()),
        default="atomic",
        help="CPU model to use",
    )
    parser.add_argument("--cpu-freq", type=str, default="4GHz")
    parser.add_argument(
        "--num-cores", type=int, default=1, help="Number of CPU cores"
    )
    parser.add_argument(
        "--mem-type",
        default="DDR3_1600_8x8",
        choices=ObjectList.mem_list.get_names(),
        help="type of memory to use",
    )
    parser.add_argument(
        "--mem-channels", type=int, default=2, help="number of memory channels"
    )
    parser.add_argument(
        "--mem-ranks",
        type=int,
        default=None,
        help="number of memory ranks per channel",
    )
    parser.add_argument(
        "--mem-size",
        action="store",
        type=str,
        default="4GB",
        help="Specify the physical memory size",
    )
    parser.add_argument(
        "--tarmac-gen",
        action="store_true",
        help="Write a Tarmac trace.",
    )
    parser.add_argument(
        "--tarmac-dest",
        choices=TarmacDump.vals,
        default="stdoutput",
        help="Destination for the Tarmac trace output. [Default: stdoutput]",
    )
    parser.add_argument(
        "-I",
        "--maxinsts",
        action="store",
        type=int,
        default=None,
        help="""Total number of instructions to
                                            simulate (default: run forever)""",
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        action="store",
        type=str,
        help="Place all checkpoints in this absolute directory",
        default="m5out",
    )
    
    # Simpoint options
    parser.add_argument(
        "--simpoint-profile",
        action="store_true",
        help="Enable basic block profiling for SimPoints",
    )
    parser.add_argument(
        "--simpoint-interval",
        type=int,
        default=10000000,
        help="SimPoint interval in num of instructions",
    )
    parser.add_argument(
        "--take-simpoint-checkpoints",
        action="store",
        type=str,
        help="<simpoint file,weight file,interval-length,warmup-length>",
    )
    parser.add_argument(
        "--restore-simpoint-checkpoint",
        action="store_true",
        default=False,
        help="restore from a simpoint checkpoint taken with "
        + "--take-simpoint-checkpoints",
    )

    args = parser.parse_args()

    # Create a single root node for gem5's object hierarchy. There can
    # only exist one root node in the simulator at any given
    # time. Tell gem5 that we want to use syscall emulation mode
    # instead of full system mode.
    root = Root(full_system=False)

    # Populate the root node with a system. A system corresponds to a
    # single node with shared memory.
    root.system, simpoints, interval_length = create(args)

    # Instantiate the C++ object hierarchy. After this point,
    # SimObjects can't be instantiated anymore.
    m5.instantiate()

    # Start the simulator. This gives control to the C++ world and
    # starts the simulator. The returned event tells the simulation
    # script why the simulator exited.
    
    cptdir = args.checkpoint_dir
        
    if args.take_simpoint_checkpoints != None:
        takeSimpointCheckpoints(simpoints, interval_length, cptdir)
    # Restore from SimPoint checkpoints
    elif args.restore_simpoint_checkpoint:
        restoreSimpointCheckpoint()
    else:    
        event = m5.simulate()

    # Print the reason for the simulation exit. Some exit codes are
    # requests for service (e.g., checkpoints) from the simulation
    # script. We'll just ignore them here and exit.
    print(f"{event.getCause()} ({event.getCode()}) @ {m5.curTick()}")


if __name__ == "__m5_main__":
    main()
