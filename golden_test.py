import base64
import contextlib
import io
import logging
import os
import re
import tempfile

import pytest

import machine
import translator

MAX_LOG_LENGTH = 4000


def sanitize_output(text: str) -> str:
    """Заменяет нулевые символы (\x00), которые запрещены в YAML, на читаемый плейсхолдер."""
    return text.replace("\x00", "<NUL>")


@pytest.mark.golden_test("golden/*.yml")
def test_translator_and_machine(golden, caplog):
    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdirname:
        source_path = os.path.join(tmpdirname, "source.f")
        input_path = os.path.join(tmpdirname, "input.txt")
        target_path = os.path.join(tmpdirname, "target.bin")

        with open(source_path, "w", encoding="utf-8") as f:
            f.write(golden["in_source"])
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(golden["in_stdin"])

        with contextlib.redirect_stdout(io.StringIO()) as stdout_io:
            translator.main(source_path, target_path)
            print("============================================================")
            limit = golden.get("in_limit", 500000)
            cache_size = golden.get("in_cache_size", 256)

            with open(target_path, "rb") as f_bin:
                binary_code = f_bin.read()

            output, ticks = machine.simulation(
                binary_code=binary_code,
                input_str=golden["in_stdin"],
                limit=limit,
                cache_size=cache_size,
            )
            print(f"\nSimulation output: '{output}'")
            print(f"Total ticks: {ticks}")

        with open(target_path, "rb") as f:
            binary_code_read = f.read()
        binary_code_b64 = base64.b64encode(binary_code_read).decode("utf-8")

        hex_listing_path = target_path + ".txt"
        with open(hex_listing_path, "r", encoding="utf-8", newline="") as f:
            hex_code_raw = f.read()
        hex_code_normalized = re.sub(
            r"; Source: .*", "; Source: <source_path>", hex_code_raw
        )

        stdout_raw = stdout_io.getvalue()
        stdout_normalized = re.sub(
            r"Successfully translated .*",
            "Successfully translated <source_path> to <target_path>",
            stdout_raw,
        )

        log_raw_text = caplog.text
        log_sanitized = sanitize_output(log_raw_text)
        if len(log_sanitized) > MAX_LOG_LENGTH:
            log_final = log_sanitized[:MAX_LOG_LENGTH] + "\n... (log truncated)"
        else:
            log_final = log_sanitized
        assert binary_code_b64 == golden.out["out_code"]
        assert hex_code_normalized == golden.out["out_code_hex"]
        assert stdout_normalized == golden.out["out_stdout"]
        assert log_final == golden.out["out_log"]
