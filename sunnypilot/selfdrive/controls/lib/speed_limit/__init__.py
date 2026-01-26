"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

LIMIT_ADAPT_ACC = -1.0  # m/s^2 Ideal acceleration for the adapting (braking) phase when approaching speed limits.
LIMIT_MAX_MAP_DATA_AGE = 10.0  # s Maximum time to hold to map data, then consider it invalid inside limits controllers.

# Speed Limit Assist constants
CONFIRM_SPEED_THRESHOLD = {
  True: 80,  # km/h
  False: 50,  # mph
}
