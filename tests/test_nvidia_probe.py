import subprocess
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))
import nvidia_probe


class NvidiaProbeTests(unittest.TestCase):
    @staticmethod
    def _runner(lspci='', listing='', listing_code=0):
        def run(command, **_kwargs):
            if command == ['lspci']:
                return subprocess.CompletedProcess(command, 0, lspci, '')
            return subprocess.CompletedProcess(command, listing_code, listing, '')
        return run

    def test_arbitrary_gpu_index_is_accepted(self):
        listing = 'GPU 17: NVIDIA RTX (UUID: GPU-test)\n'
        result = nvidia_probe.responding_gpu_listing(
            self._runner('NVIDIA controller', listing), which=lambda _name: '/bin/nvidia-smi')
        self.assertEqual(result, listing.strip())

    def test_multiple_gpus_are_accepted(self):
        listing = 'GPU 4: NVIDIA A\nGPU 12: NVIDIA B\n'
        self.assertEqual(nvidia_probe.responding_gpu_listing(
            self._runner('NVIDIA', listing), which=lambda _name: 'nvidia-smi'), listing.strip())

    def test_missing_hardware_is_rejected_before_driver_probe(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, 'Intel graphics', '')
        self.assertIsNone(nvidia_probe.responding_gpu_listing(run, lambda _name: 'nvidia-smi'))
        self.assertEqual(calls, [['lspci']])

    def test_failed_driver_is_rejected(self):
        self.assertIsNone(nvidia_probe.responding_gpu_listing(
            self._runner('NVIDIA', 'GPU 8: NVIDIA', listing_code=1),
            which=lambda _name: 'nvidia-smi'))


if __name__ == '__main__':
    unittest.main()
