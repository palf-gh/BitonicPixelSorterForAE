#ifndef STRINGS_EN_US_H
#define STRINGS_EN_US_H

#define AELOCALISE_HAS_EN_US

#include "LocKeys.h"

namespace AELocalise {
namespace EN_US {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_DIRECTION_NAME:
    return "Direction";
  case LocKey::STR_DIRECTION_ITEMS:
    return "Horizontal|Vertical";
  case LocKey::STR_ORDER_NAME:
    return "Order";
  case LocKey::STR_ORDER_ITEMS:
    return "Ascending|Descending";
  case LocKey::STR_THRESHOLD_MIN:
    return "Threshold Min";
  case LocKey::STR_THRESHOLD_MAX:
    return "Threshold Max";
  default:
    return "";
  }
}
} // namespace EN_US
} // namespace AELocalise

#endif // STRINGS_EN_US_H
