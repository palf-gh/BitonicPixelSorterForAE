#ifndef STRINGS_KO_KR_H
#define STRINGS_KO_KR_H

#define AELOCALISE_HAS_KO_KR

#include "LocKeys.h"

namespace AELocalise {
namespace KO_KR {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_DIRECTION_NAME:
    return "방향";
  case LocKey::STR_DIRECTION_ITEMS:
    return "수평|수직";
  case LocKey::STR_ORDER_NAME:
    return "정렬";
  case LocKey::STR_ORDER_ITEMS:
    return "오름차순|내림차순";
  case LocKey::STR_THRESHOLD_MIN:
    return "임계값 하한";
  case LocKey::STR_THRESHOLD_MAX:
    return "임계값 상한";
  default:
    return "";
  }
}
} // namespace KO_KR
} // namespace AELocalise

#endif // STRINGS_KO_KR_H
