import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

class MockLeadOne:
  def __init__(self, status=0.0):
    self.status = status

class MockRadarState:
  def __init__(self, status=0.0):
    self.leadOne = MockLeadOne(status=status)

class MockCarState:
  def __init__(self, vEgo=0.0, vCruise=0.0, standstill=False):
    self.vEgo = vEgo
    self.vCruise = vCruise
    self.standstill = standstill

class MockModelData:
  def __init__(self, valid=True):
    size = 33 if valid else 10  # incomplete if invalid
    self.position = type("Pos", (), {"x": [0.0] * size})()
    self.orientation = type("Ori", (), {"x": [0.0] * size})()

class MockSelfDriveState:
  def __init__(self, experimentalMode=False):
    self.experimentalMode = experimentalMode

class MockParams:
  def __init__(self, sensitivity=0):
    self.sensitivity = sensitivity
    
  def get_bool(self, name):
    return True
    
  def get_int(self, name):
    if name == "DynamicExperimentalControlSensitivity":
      return self.sensitivity
    return 0

@pytest.fixture
def default_sm():
  sm = {
    'carState': MockCarState(vEgo=10.0, vCruise=20.0),
    'radarState': MockRadarState(status=1.0),
    'modelV2': MockModelData(valid=True),
    'selfdriveState': MockSelfDriveState(experimentalMode=True),
  }
  return sm

@pytest.fixture
def mock_cp():
  class CP:
    radarUnavailable = False
  return CP()

@pytest.fixture
def mock_mpc():
  class MPC:
    crash_cnt = 0
  return MPC()

# Fake Kalman Filter that always returns a given value
class FakeKalman:
  def __init__(self, value=1.0):
    self.value = value
  def add_data(self, v): pass
  def get_value(self): return self.value
  def get_confidence(self): return 1.0
  def reset_data(self): pass

def test_initial_mode_is_acc(mock_cp, mock_mpc):
  controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
  assert controller.mode() == "acc"

def test_standstill_triggers_blended(mock_cp, mock_mpc, default_sm):
  controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
  default_sm['carState'].standstill = True
  for _ in range(10):
    controller.update(default_sm)
  assert controller.mode() == "blended"

def test_emergency_blended_on_fcw(mock_cp, mock_mpc, default_sm):
  controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
  mock_mpc.crash_cnt = 1  # simulate FCW
  for _ in range(2):
    controller.update(default_sm)
  assert controller.mode() == "blended"

def test_radarless_slowdown_triggers_blended(mock_cp, mock_mpc, default_sm):
  mock_cp.radarUnavailable = True
  controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())

  # Force conditions to simulate slowdown
  controller._slow_down_filter = FakeKalman(value=1.0)  # Ensure urgency triggers slowdown
  controller._v_ego_kph = 35.0
  default_sm['modelV2'] = MockModelData(valid=False)  # Incomplete trajectory

  for _ in range(3):
    controller.update(default_sm)

  assert controller.mode() == "blended"

def test_default_sensitivity_settings(mock_cp, mock_mpc):
  controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams(sensitivity=0))
  assert controller._sensitivity == 0
  assert controller._urgency_emergency_threshold == 0.7
  assert controller._mode_manager.min_mode_duration == 10
  assert controller._mode_manager.confidence_threshold_change == 0.6
  assert controller._mode_manager.confidence_decay == 0.98

def test_low_sensitivity_settings(mock_cp, mock_mpc):
  controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams(sensitivity=1))
  assert controller._sensitivity == 1
  assert controller._urgency_emergency_threshold == 0.6
  assert controller._mode_manager.min_mode_duration == 5
  assert controller._mode_manager.confidence_threshold_change == 0.5
  assert controller._mode_manager.confidence_decay == 0.95

def test_sensitivity_affects_transition_speed(mock_cp, mock_mpc, default_sm):
  # Test that low sensitivity creates smoother transitions
  controller_default = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams(sensitivity=0))
  controller_low = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams(sensitivity=1))
  
  # Both should have same initial mode
  assert controller_default.mode() == "acc"
  assert controller_low.mode() == "acc"
  
  # Low sensitivity should have lower threshold
  assert controller_low._mode_manager.confidence_threshold_change < controller_default._mode_manager.confidence_threshold_change
