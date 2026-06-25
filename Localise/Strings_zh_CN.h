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
  default:
    return "";
  }
}
} // namespace ZH_CN
} // namespace AELocalise

#endif // STRINGS_ZH_CN_H
