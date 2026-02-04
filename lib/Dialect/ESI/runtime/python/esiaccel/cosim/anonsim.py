#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .simulator import CosimCollateralDir, Simulator, SourceFiles


class AnonSim(Simulator):
  """Run and compile funcs for the AnonSim (vsim) simulator."""

  DefaultDriver = CosimCollateralDir / "driver.sv"

  # AnonSim doesn't use stderr for error messages. Everything goes to stdout.
  UsesStderr = False

  def __init__(
      self,
      sources: SourceFiles,
      run_dir: Path,
      debug: bool,
      run_stdout_callback: Optional[Callable[[str], None]] = None,
      run_stderr_callback: Optional[Callable[[str], None]] = None,
      compile_stdout_callback: Optional[Callable[[str], None]] = None,
      compile_stderr_callback: Optional[Callable[[str], None]] = None,
      make_default_logs: bool = True,
      macro_definitions: Optional[Dict[str, str]] = None,
      suppressed_vsim_errors: Optional[List[int]] = None,
  ):
    super().__init__(
        sources=sources,
        run_dir=run_dir,
        debug=debug,
        run_stdout_callback=run_stdout_callback,
        run_stderr_callback=run_stderr_callback,
        compile_stdout_callback=compile_stdout_callback,
        compile_stderr_callback=compile_stderr_callback,
        make_default_logs=make_default_logs,
        macro_definitions=macro_definitions,
    )
    self.suppressed_vsim_errors = suppressed_vsim_errors

  def internal_compile_commands(self) -> List[str]:
    cmds = [
        "onerror { quit -f -code 1 }",
    ]
    sources = self.sources.rtl_sources
    sources.append(AnonSim.DefaultDriver)

    # Format macro definition command.
    if self.macro_definitions:
      macro_definitions_cmd = " ".join([
          f"+define+{k}={v}" if v is not None else f"+define+{k}"
          for k, v in self.macro_definitions.items()
      ])
    else:
      macro_definitions_cmd = ""

    # Format warning/error suppression command.
    if self.suppressed_vsim_errors:
      suppressed_errors_cmd = " ".join(
          [f"-suppress {ec}" for ec in self.suppressed_vsim_errors])
    else:
      suppressed_errors_cmd = ""

    for src in sources:
      parts = [
          "vlog",
          "-incr",
          "+acc",
          "-sv",
      ]
      if macro_definitions_cmd:
        parts.append(macro_definitions_cmd)
      if suppressed_errors_cmd:
        parts.append(suppressed_errors_cmd)
      parts += [
          f"+define+TOP_MODULE={self.sources.top}",
          "+define+SIMULATION",
          src.as_posix(),
      ]
      cmds.append(" ".join(parts))

    cmds.append("vopt -incr driver -o driver_opt +acc")
    return cmds

  def compile_commands(self) -> List[List[str]]:
    with open("compile.do", "w") as f:
      for cmd in self.internal_compile_commands():
        f.write(cmd)
        f.write("\n")
      f.write("quit\n")
    return [
        ["vsim", "-batch", "-do", "compile.do"],
    ]

  def run_command(self, gui: bool) -> List[str]:
    vsim = "vsim"
    # Note: vsim exit codes say nothing about the test run's pass/fail even
    # if $fatal is encountered in the simulation.
    if gui:
      cmd = [
          vsim,
          "driver_opt",
      ]
    else:
      cmd = [
          vsim,
          "driver_opt",
          "-batch",
      ]

      if self.debug:
        # Create waveform dump .do file.
        wave_file = Path("wave.do")
        with wave_file.open("w") as f:
          f.write("log -r /*\n")
        cmd += [
            "-do",
            str(wave_file.resolve()),
        ]
        # AnonSim will by default log to a vsim.wlf file in the current
        # directory.
        print("Debug mode enabled - waveform will be in "
              f"{wave_file.resolve().parent / 'vsim.wlf'}")

      cmd += [
          "-do",
          "run -all",
      ]

    for lib in self.sources.dpi_so_paths():
      sv_lib = os.path.splitext(lib)[0]
      cmd.append("-sv_lib")
      cmd.append(sv_lib)
    return cmd

  def run(self,
          inner_command: str,
          gui: bool = False,
          server_only: bool = False) -> int:
    """Override the Simulator.run() to add a soft link in the run directory (to
    the work directory) before running the simulator."""

    # Create a soft link to the work directory.
    work_dir = self.run_dir / "work"
    if not work_dir.exists():
      os.symlink(Path(os.getcwd()) / "work", work_dir)

    # Run the simulation.
    return super().run(inner_command, gui, server_only=server_only)
