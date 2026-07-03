#ifndef STRINGS_EN_US_H
#define STRINGS_EN_US_H

#define AELOCALISE_HAS_EN_US

#include "LocKeys.h"

namespace AELocalise {
namespace EN_US {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_MODE_NAME:
    return "Mode";
  case LocKey::STR_MODE_ITEMS:
    return "Axis|Free Angle|Rotation|Radial";
  case LocKey::STR_DIRECTION_NAME:
    return "Direction";
  case LocKey::STR_DIRECTION_ITEMS:
    return "Horizontal|Vertical";
  case LocKey::STR_ANGLE_NAME:
    return "Angle";
  case LocKey::STR_CENTER_NAME:
    return "Centre";
  case LocKey::STR_SORT_CRITERION_NAME:
    return "Sort Criterion";
  case LocKey::STR_SORT_CRITERION_ITEMS:
    return "Luminance|RGB Average|RGB Product|RGB Minimum|RGB Maximum";
  case LocKey::STR_ORDER_NAME:
    return "Order";
  case LocKey::STR_ORDER_ITEMS:
    return "Ascending|Descending";
  case LocKey::STR_THRESHOLD_MIN:
    return "Threshold Min";
  case LocKey::STR_THRESHOLD_MAX:
    return "Threshold Max";
  case LocKey::STR_GPU_STATUS_NAME:
    return "GPU Acceleration";
  case LocKey::STR_GPU_STATUS_ACTIVE:
    return "GPU: Active";
  case LocKey::STR_GPU_STATUS_NO_BACKEND:
    return "GPU: Unavailable (no GPU backend)";
  case LocKey::STR_GPU_STATUS_HOST:
    return "GPU: Unavailable (host)";
  case LocKey::STR_GPU_STATUS_AXIS_TOO_LONG:
    return "GPU: CPU fallback (image too large)";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU: Inactive (enable Mercury GPU in Project Settings)";
  default:
    return "";
  }
}
} // namespace EN_US
} // namespace AELocalise

#endif // STRINGS_EN_US_H
