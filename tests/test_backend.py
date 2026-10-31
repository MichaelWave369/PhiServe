import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from phiserve.backend import LlamaBackend


class BackendTests(unittest.TestCase):
    def test_missing_executable_fails_without_spawning(self):
        model = Path(tempfile.gettempdir()) / "example.gguf"
        backend = LlamaBackend("missing-llama-server-binary", model, "demo", 12345)
        with patch("phiserve.backend.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                backend.start()
        self.assertIsNone(backend.process)

    def test_backend_invocation_uses_fixed_local_argv_no_shell(self):
        with patch("phiserve.backend.shutil.which", return_value="/bin/llama-server"):
            with patch("phiserve.backend.socket.socket") as socket_cls:
                socket_cls.return_value.__enter__.return_value.connect_ex.return_value = 111
                with patch("phiserve.backend.subprocess.Popen") as popen:
                    process = MagicMock()
                    process.poll.return_value = None
                    popen.return_value = process
                    with patch("phiserve.backend.urllib.request.urlopen") as urlopen:
                        response = urlopen.return_value.__enter__.return_value
                        response.status = 200
                        with patch("phiserve.backend.json.load", return_value={"status": "ok"}):
                            backend = LlamaBackend("llama-server", Path("/local/my.gguf"), "demo", 19000, gpu_layers=12)
                            backend.start()
                            argv, kwargs = popen.call_args
                            self.assertEqual(argv[0][0], "/bin/llama-server")
                            self.assertIn("--alias", argv[0])
                            self.assertIn("127.0.0.1", argv[0])
                            self.assertEqual(kwargs["shell"], False)
                            backend.stop()
                            process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
