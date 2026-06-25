#ifndef STRINGS_ZH_CN_H
#define STRINGS_ZH_CN_H

#define AELOCALISE_HAS_ZH_CN

#include "LocKeys.h"

namespace AELocalise {
namespace ZH_CN {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_DIRECTION_NAME:
    return "方向";
  case LocKey::STR_DIRECTION_ITEMS:
    return "水平|垂直";
  case LocKey::STR_ORDER_NAME:
    return "排序";
  case LocKey::STR_ORDER_ITEMS:
    return "升序|降序";
  case LocKey::STR_THRESHOLD_MIN:
    return "阈值下限";
  case LocKey::STR_THRESHOLD_MAX:
    return "阈值上限";
  case LocKey::STR_GPU_STATUS_NAME:
    return "GPU 加速";
  case LocKey::STR_GPU_STATUS_ACTIVE:
    return "GPU：已启用";
  case LocKey::STR_GPU_STATUS_NO_BACKEND:
    return "GPU：不可用（无 GPU 后端）";
  case LocKey::STR_GPU_STATUS_HOST:
    return "GPU：不可用（宿主）";
  case LocKey::STR_GPU_STATUS_AXIS_TOO_LONG:
    return "GPU：CPU 回退（图像过大）";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU：未启用（请在项目设置中启用 Mercury GPU）";
  default:
    return "";
  }
}
} // namespace ZH_CN
} // namespace AELocalise

#endif // STRINGS_ZH_CN_H
