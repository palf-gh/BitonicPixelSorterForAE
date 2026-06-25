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
  case LocKey::STR_GPU_STATUS_NAME:
    return "GPU 가속";
  case LocKey::STR_GPU_STATUS_ACTIVE:
    return "GPU: 활성";
  case LocKey::STR_GPU_STATUS_NO_BACKEND:
    return "GPU: 사용 불가(GPU 백엔드 없음)";
  case LocKey::STR_GPU_STATUS_HOST:
    return "GPU: 사용 불가(호스트)";
  case LocKey::STR_GPU_STATUS_AXIS_TOO_LONG:
    return "GPU: CPU 폴백(이미지가 너무 큼)";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU: 비활성(프로젝트 설정에서 Mercury GPU를 활성화하세요)";
  default:
    return "";
  }
}
} // namespace KO_KR
} // namespace AELocalise

#endif // STRINGS_KO_KR_H
