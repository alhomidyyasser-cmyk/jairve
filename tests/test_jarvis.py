import json
import errno
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

import jarvis


class JarvisMathTests(unittest.TestCase):
    def test_evaluates_basic_arithmetic(self):
        self.assertEqual(jarvis.evaluate_math("(2 + 3) * 4"), 20.0)

    def test_rejects_non_arithmetic_expressions(self):
        with self.assertRaises(ValueError):
            jarvis.evaluate_math("__import__('os').system('whoami')")

    def test_rejects_division_by_zero(self):
        with self.assertRaises(ValueError):
            jarvis.evaluate_math("1 / 0")


class JarvisCameraTests(unittest.TestCase):
    def test_tracker_reports_incompatible_mediapipe_version(self):
        incompatible_mediapipe = SimpleNamespace(__version__="1.2.2")
        with mock.patch.object(jarvis, "mp", incompatible_mediapipe):
            with self.assertRaisesRegex(RuntimeError, "MediaPipe 1.2.2"):
                jarvis.create_hand_tracker()

    def test_tracker_uses_solutions_hands_api(self):
        tracker = object()
        hands_api = SimpleNamespace(
            Hands=mock.Mock(return_value=tracker),
            HAND_CONNECTIONS=object(),
        )
        compatible_mediapipe = SimpleNamespace(
            __version__="0.10.21",
            solutions=SimpleNamespace(hands=hands_api),
        )
        with mock.patch.object(jarvis, "mp", compatible_mediapipe):
            actual_tracker, connections = jarvis.create_hand_tracker()
        self.assertIs(actual_tracker, tracker)
        self.assertIs(connections, hands_api.HAND_CONNECTIONS)
        hands_api.Hands.assert_called_once()


class JarvisHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = jarvis.ThreadedHTTPServer(
            ("127.0.0.1", 0), jarvis.TitanHTTPHandler
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def post(self, path, body, content_type="application/json"):
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as error:
            response = error
        return response.code, json.loads(response.read())

    def test_chat_endpoint_returns_a_local_reply(self):
        status, payload = self.post(
            "/api/chat",
            json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode(),
        )
        self.assertEqual(status, 200)
        self.assertIn("reply", payload)

    def test_chat_endpoint_rejects_malformed_json(self):
        status, payload = self.post("/api/chat", b"{")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_chat_endpoint_rejects_non_array_messages(self):
        status, payload = self.post(
            "/api/chat", json.dumps({"messages": "hello"}).encode()
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_oversized_request_is_rejected(self):
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=b"x" * (jarvis.MAX_REQUEST_BYTES + 1),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(error.exception.code, 413)

    def test_python_microphone_is_disabled_by_default(self):
        with mock.patch.object(jarvis, "ENABLE_PY_MIC", False):
            with mock.patch.object(jarvis, "sr") as speech_recognition:
                jarvis.background_mic_worker()
        speech_recognition.assert_not_called()

    def test_metrics_do_not_fabricate_values_without_psutil(self):
        with mock.patch.object(jarvis, "psutil", None):
            metrics = jarvis.collect_system_metrics()
        for key in ("cpu", "ram", "disk", "processes", "net_up_kbs", "net_down_kbs"):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics["cores"], [])

    def test_server_suppresses_expected_windows_client_disconnect(self):
        error = ConnectionAbortedError("client disconnected")
        error.winerror = 10053
        with mock.patch.object(jarvis.sys, "exc_info", return_value=(None, error, None)):
            with mock.patch.object(jarvis.LOG, "debug") as log_debug:
                self.server.handle_error(None, ("127.0.0.1", 12345))
        log_debug.assert_called_once()


if __name__ == "__main__":
    unittest.main()
