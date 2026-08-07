from __future__ import annotations

import sys

from scripts.review_agent.common import run_cmd


def test_run_cmd_tee_output_streams_and_captures_both_streams(
    capsys,
) -> None:
    completed = run_cmd(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)",
        ],
        stream_stdout=False,
        stream_stderr=False,
        tee_output=True,
    )

    captured = capsys.readouterr()

    assert completed.stdout == "stdout-line\n"
    assert completed.stderr == "stderr-line\n"
    assert "stdout-line\n" in captured.out
    assert "stderr-line\n" in captured.err

