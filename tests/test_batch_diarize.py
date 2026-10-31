import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from podtrans import batch_diarize
from podtrans.profiles import AppleChip, Family, Machine, SPEAKER_ERES2NET_V2


def apple_machine() -> Machine:
    return Machine(
        Family.APPLE,
        "Apple M4",
        apple=AppleChip(4, 24, "Apple M4"),
    )


class BatchDiarizeTests(unittest.TestCase):
    def test_cpu_profile_runs_in_process_without_service(self) -> None:
        with patch("podtrans.diarize.main", return_value=0) as diarize_main:
            exit_code = batch_diarize.main(
                ["--profile", "cpu", "--", "--limit", "1"]
            )
        self.assertEqual(exit_code, 0)
        args = diarize_main.call_args.args[0]
        self.assertIn("cpu", args)
        self.assertEqual(args[-2:], ["--limit", "1"])

    def test_apple_profile_runs_in_process_without_service(self) -> None:
        with (
            patch("podtrans.diarize.main", return_value=0) as diarize_main,
            patch("podtrans.batch_diarize.subprocess.Popen") as popen,
        ):
            exit_code = batch_diarize.main(
                ["--profile", "apple-m4-16gb", "--", "--limit", "1"]
            )
        self.assertEqual(exit_code, 0)
        popen.assert_not_called()
        args = diarize_main.call_args.args[0]
        self.assertEqual(args[-2:], ["--limit", "1"])
        self.assertIn(SPEAKER_ERES2NET_V2, args)
        self.assertNotIn("--speaker-service-socket", args)
        self.assertEqual(args[args.index("--device") + 1], "cpu")

    def test_detected_apple_machine_is_announced_and_stays_in_process(self) -> None:
        stderr = io.StringIO()
        with (
            patch("podtrans.batch_diarize.detect_machine", return_value=apple_machine()),
            patch("podtrans.diarize.main", return_value=0) as diarize_main,
            patch("podtrans.batch_diarize.subprocess.Popen") as popen,
            redirect_stderr(stderr),
        ):
            exit_code = batch_diarize.main(["--", "--limit", "1"])
        self.assertEqual(exit_code, 0)
        popen.assert_not_called()
        diarize_main.assert_called_once()
        self.assertIn("podtrans: using profile apple-m4-16gb", stderr.getvalue())

    def test_apple_profile_rejects_more_than_one_worker(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            batch_diarize.main(["--profile", "apple-m3-8gb", "--workers", "2"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("exactly one in-process worker", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
