#!/usr/bin/env python3
"""Compile the Gated DeltaNet kernels with NVRTC without requiring a GPU.

This checks CUDA syntax and exported entry points, not numerical parity or
launch/resource safety. Pass a local NVRTC shared library with --nvrtc-library.
Only the Python standard library is required.
"""
import argparse
import ctypes as ct
import ctypes.util
import hashlib
import json
import re
from pathlib import Path


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with NVRTC status {status}")


def bind(library, name, arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ct.c_int
    return function


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvrtc-library", default=ctypes.util.find_library("nvrtc"))
    parser.add_argument("--arch", action="append", help="virtual CUDA target; repeatable")
    args = parser.parse_args()
    if not args.nvrtc_library:
        parser.error("NVRTC was not found; supply --nvrtc-library")
    library = ct.CDLL(args.nvrtc_library)
    version = bind(library, "nvrtcVersion", [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)])
    create = bind(library, "nvrtcCreateProgram", [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p])
    compile_program = bind(library, "nvrtcCompileProgram", [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)])
    log_size = bind(library, "nvrtcGetProgramLogSize", [ct.c_void_p, ct.POINTER(ct.c_size_t)])
    get_log = bind(library, "nvrtcGetProgramLog", [ct.c_void_p, ct.c_void_p])
    ptx_size = bind(library, "nvrtcGetPTXSize", [ct.c_void_p, ct.POINTER(ct.c_size_t)])
    get_ptx = bind(library, "nvrtcGetPTX", [ct.c_void_p, ct.c_void_p])
    destroy = bind(library, "nvrtcDestroyProgram", [ct.POINTER(ct.c_void_p)])
    major, minor = ct.c_int(), ct.c_int()
    check(version(ct.byref(major), ct.byref(minor)), "version")
    source_path = root / "crates/tenferro-gated-delta/src/cuda/kernels.cu"
    source = source_path.read_bytes()
    signatures = {
        "gated_delta_conv_silu": ["u64"] * 3 + ["u32"] * 3,
        "gated_delta_recurrent": ["u64"] * 6 + ["u32"] * 5,
        "gated_delta_norm_gate": ["u64"] * 4 + ["u32"] * 2 + ["f32"],
    }
    results = []
    for arch in args.arch or ["compute_70", "compute_80", "compute_90"]:
        program = ct.c_void_p()
        check(create(ct.byref(program), source, b"gated_delta.cu", 0, None, None), "create")
        try:
            flags = ["--std=c++14", f"--gpu-architecture={arch}", "--fmad=false"]
            options = (ct.c_char_p * len(flags))(*(flag.encode() for flag in flags))
            status = compile_program(program, len(flags), options)
            size = ct.c_size_t()
            check(log_size(program, ct.byref(size)), "log size")
            log = ct.create_string_buffer(size.value)
            check(get_log(program, log), "log")
            if status != 0:
                raise RuntimeError(f"NVRTC compilation for {arch} failed ({status}):\n{log.value.decode()}")
            check(ptx_size(program, ct.byref(size)), "PTX size")
            ptx = ct.create_string_buffer(size.value)
            check(get_ptx(program, ptx), "PTX")
            text = ptx.value.decode()
            for entry, expected in signatures.items():
                signature = re.search(rf"\.entry\s+{entry}\s*\((.*?)\)", text, re.DOTALL)
                if signature is None:
                    raise RuntimeError(f"missing PTX entry: {entry}")
                actual = re.findall(r"\.param\s+\.(u64|u32|f32)\s", signature.group(1))
                if actual != expected:
                    raise RuntimeError(f"ABI mismatch for {entry}: {actual} != {expected}")
            results.append({"arch": arch, "flags": flags, "ptx_bytes": len(ptx.value), "signatures": signatures, "log": log.value.decode()})
        finally:
            check(destroy(ct.byref(program)), "destroy")
    print(json.dumps({"nvrtc_version": f"{major.value}.{minor.value}", "source": str(source_path.relative_to(root)), "source_sha256": hashlib.sha256(source).hexdigest(), "targets": results, "gpu_execution_tested": False}, indent=2))


if __name__ == "__main__":
    main()
