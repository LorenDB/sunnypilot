"""
Tests for the WebDAV uploader functionality.
"""
from openpilot.common.params import Params
from openpilot.sunnypilot.webdav.utils import get_webdav_config, webdav_ready, use_webdav_uploader


class TestWebDAVUtils:
  """Test the WebDAV utility functions."""

  def setup_method(self):
    self.params = Params()
    # Clear all WebDAV params before each test
    self.params.remove("WebDAVEnabled")
    self.params.remove("WebDAVEndpoint")
    self.params.remove("WebDAVUsername")
    self.params.remove("WebDAVPassword")
    self.params.remove("NetworkMetered")

  def test_get_webdav_config_disabled(self):
    """Test that config returns disabled when WebDAVEnabled is not set."""
    self.params.put_bool("WebDAVEnabled", False)
    is_enabled, endpoint, username, password = get_webdav_config(self.params)
    assert is_enabled is False
    assert endpoint is None
    assert username is None
    assert password is None

  def test_get_webdav_config_enabled_with_endpoint(self):
    """Test that config returns all fields when set."""
    self.params.put_bool("WebDAVEnabled", True)
    self.params.put("WebDAVEndpoint", "https://webdav.example.com/uploads/")
    self.params.put("WebDAVUsername", "testuser")
    self.params.put("WebDAVPassword", "testpass")

    is_enabled, endpoint, username, password = get_webdav_config(self.params)
    assert is_enabled is True
    assert endpoint == "https://webdav.example.com/uploads/"
    assert username == "testuser"
    assert password == "testpass"

  def test_webdav_ready_returns_false_when_disabled(self):
    """Test that webdav_ready returns False when not enabled."""
    self.params.put_bool("WebDAVEnabled", False)
    self.params.put("WebDAVEndpoint", "https://webdav.example.com/uploads/")
    assert webdav_ready(self.params) is False

  def test_webdav_ready_returns_false_when_no_endpoint(self):
    """Test that webdav_ready returns False when endpoint is not set."""
    self.params.put_bool("WebDAVEnabled", True)
    assert webdav_ready(self.params) is False

  def test_webdav_ready_returns_false_when_endpoint_empty(self):
    """Test that webdav_ready returns False when endpoint is empty."""
    self.params.put_bool("WebDAVEnabled", True)
    self.params.put("WebDAVEndpoint", "")
    assert webdav_ready(self.params) is False

  def test_webdav_ready_returns_true_when_properly_configured(self):
    """Test that webdav_ready returns True when properly configured."""
    self.params.put_bool("WebDAVEnabled", True)
    self.params.put("WebDAVEndpoint", "https://webdav.example.com/uploads/")
    assert webdav_ready(self.params) is True

  def test_use_webdav_uploader_returns_false_on_metered(self):
    """Test that use_webdav_uploader returns False on metered connection."""
    self.params.put_bool("WebDAVEnabled", True)
    self.params.put("WebDAVEndpoint", "https://webdav.example.com/uploads/")
    self.params.put_bool("NetworkMetered", True)
    assert use_webdav_uploader(self.params) is False

  def test_use_webdav_uploader_returns_false_when_not_ready(self):
    """Test that use_webdav_uploader returns False when not ready."""
    self.params.put_bool("WebDAVEnabled", False)
    self.params.put_bool("NetworkMetered", False)
    assert use_webdav_uploader(self.params) is False

  def test_use_webdav_uploader_returns_true_when_ready_and_unmetered(self):
    """Test that use_webdav_uploader returns True when ready and on unmetered connection."""
    self.params.put_bool("WebDAVEnabled", True)
    self.params.put("WebDAVEndpoint", "https://webdav.example.com/uploads/")
    self.params.put_bool("NetworkMetered", False)
    assert use_webdav_uploader(self.params) is True


class TestWebDAVUploaderUnit:
  """Unit tests for WebDAVUploader class."""

  def setup_method(self):
    self.params = Params()
    self.params.put("DongleId", "test_dongle_id")
    self.params.put_bool("WebDAVEnabled", True)
    self.params.put("WebDAVEndpoint", "https://webdav.example.com/uploads/")
    self.params.put("WebDAVUsername", "testuser")
    self.params.put("WebDAVPassword", "testpass")
    self.params.put_bool("NetworkMetered", False)

  def test_uploader_init(self):
    """Test that the WebDAVUploader initializes correctly."""
    from openpilot.sunnypilot.webdav.uploader import WebDAVUploader

    uploader = WebDAVUploader("test_dongle_id", "/tmp/test_root")
    assert uploader.dongle_id == "test_dongle_id"
    assert uploader.root == "/tmp/test_root"

  def test_ensure_remote_directory(self, mocker):
    """Test that _ensure_remote_directory creates parent directories."""
    from openpilot.sunnypilot.webdav.uploader import WebDAVUploader

    mock_response = mocker.MagicMock()
    mock_response.status_code = 201
    mocker.patch('openpilot.sunnypilot.webdav.uploader.requests.request', return_value=mock_response)

    uploader = WebDAVUploader("test_dongle_id", "/tmp/test_root")
    result = uploader._ensure_remote_directory(
      "https://webdav.example.com/uploads/",
      "dongle_id/segment_dir/file.log",
      ("user", "pass")
    )

    assert result is True
