import unittest

from podtrans.diarize import SPEAKER_SERVICE_AUTHKEY
from podtrans.speaker_service import AUTHKEY, DEFAULT_SOCKET


class SpeakerServiceContractTests(unittest.TestCase):
    def test_client_and_server_share_auth_key(self) -> None:
        self.assertEqual(AUTHKEY, SPEAKER_SERVICE_AUTHKEY)

    def test_default_socket_is_namespaced(self) -> None:
        self.assertEqual(DEFAULT_SOCKET.name, "podtrans-speaker-embedding.sock")


if __name__ == "__main__":
    unittest.main()
