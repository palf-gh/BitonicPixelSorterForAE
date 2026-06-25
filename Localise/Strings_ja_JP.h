#ifndef STRINGS_JA_JP_H
#define STRINGS_JA_JP_H

#define AELOCALISE_HAS_JA_JP

#include "LocKeys.h"

namespace AELocalise {
namespace JA_JP {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_DIRECTION_NAME:
    return "方向";
  case LocKey::STR_DIRECTION_ITEMS:
    return "水平|垂直";
  case LocKey::STR_ORDER_NAME:
    return "並び順";
  case LocKey::STR_ORDER_ITEMS:
    return "昇順|降順";
  case LocKey::STR_THRESHOLD_MIN:
    return "しきい値（下限）";
  case LocKey::STR_THRESHOLD_MAX:
    return "しきい値（上限）";
  default:
    return "";
  }
}
} // namespace JA_JP
} // namespace AELocalise

#endif // STRINGS_JA_JP_H
