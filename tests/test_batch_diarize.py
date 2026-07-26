import unittest
from unittest.mock import patch

from podtrans import batch_diarize


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


if __name__ == "__main__":
    unittest.main()
