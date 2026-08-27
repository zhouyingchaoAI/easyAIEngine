import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import algorithm_manager


class TestAlgorithmManagerProcessTracking(unittest.TestCase):
    def test_zombie_process_is_not_running_instance(self):
        with patch("algorithm_manager.psutil.Process") as process_cls:
            process = process_cls.return_value
            process.is_running.return_value = True
            process.status.return_value = algorithm_manager.psutil.STATUS_ZOMBIE

            self.assertFalse(algorithm_manager.get_process_status(12345))

    def test_used_ports_includes_live_managed_instances(self):
        services = {
            "realtime": {"instances": [
                {"pid": 101, "config": {"port": 7911}},
                {"pid": 102, "config": {"port": 7912}},
            ]}
        }
        with patch("algorithm_manager.get_process_status", side_effect=lambda pid: pid == 101):
            self.assertEqual(algorithm_manager.get_live_instance_ports(services), {7911})

    def test_cleanup_on_exit_allows_multi_instance_services_without_legacy_pid(self):
        services = {
            "realtime": {"name": "实时检测服务", "instances": [{"pid": 101}]},
        }
        with patch.object(algorithm_manager, "SERVICES", services):
            algorithm_manager.cleanup_on_exit()

    def test_configure_manager_logging_uses_one_file_handler(self):
        with TemporaryDirectory() as directory:
            logger = algorithm_manager.configure_manager_logging(
                Path(directory) / "manager.log"
            )
            self.assertEqual(len(logger.handlers), 1)
            self.assertIsInstance(logger.handlers[0], algorithm_manager.logging.FileHandler)


if __name__ == "__main__":
    unittest.main()
