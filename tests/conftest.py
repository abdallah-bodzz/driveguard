import pytest
from driveguard.detection import DetectionEngine

@pytest.fixture
def engine():
    eng = DetectionEngine()
    eng.cnn_available = False
    eng.cnn_enabled = False
    eng.head_nod_enabled = False
    return eng
