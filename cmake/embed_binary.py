#!/usr/bin/env python3
"""Embed a binary file as a C++ byte array header (self-contained AE plugin build)."""

from __future__ import annotations

import argparse
import pathlib
import sys


def _format_byte_line(values: bytes) -> str:
    return "    " + ", ".join(f"0x{b:02x}" for b in values) + ","


def embed_binary(input_path: pathlib.Path, output_path: pathlib.Path, symbol_name: str) -> None:
    data = input_path.read_bytes()
    lines = ["#pragma once", "", "#include <cstddef>", "#include <cstdint>", "", "namespace bps_directx_embedded {", ""]

    lines.append(f"alignas(4) inline const std::uint8_t {symbol_name}[] = {{")
    for offset in range(0, len(data), 16):
        lines.append(_format_byte_line(data[offset : offset + 16]))
    lines.append("};")
    lines.append("")
    lines.append(f"inline const std::size_t {symbol_name}_size = {len(data)};")
    lines.append("")
    lines.append("}  // namespace bps_directx_embedded")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Embed a binary file as a C++ byte array.")
    parser.add_argument("-i", "--input", required=True, help="Input binary file")
    parser.add_argument("-o", "--output", required=True, help="Output header path")
    parser.add_argument("-n", "--name", required=True, help="C++ symbol name for the byte array")
    args = parser.parse_args(argv)

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not input_path.is_file():
        print(f"embed_binary: input file not found: {input_path}", file=sys.stderr)
        return 1

    embed_binary(input_path, output_path, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
