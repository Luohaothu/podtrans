import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diarize_funasr import SPEAKER_SERVICE_AUTHKEY
from speaker_embedding_service import AUTHKEY, DEFAULT_SOCKET


class SpeakerServiceContractTests(unittest.TestCase):
    def test_client_and_server_share_auth_key(self) -> None:
        self.assertEqual(AUTHKEY, SPEAKER_SERVICE_AUTHKEY)

    def test_default_socket_is_namespaced(self) -> None:
        self.assertEqual(DEFAULT_SOCKET.name, "podtrans-speaker-embedding.sock")


if __name__ == "__main__":
    unittest.main()
