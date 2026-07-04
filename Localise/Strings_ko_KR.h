#ifndef STRINGS_KO_KR_H
#define STRINGS_KO_KR_H

#define AELOCALISE_HAS_KO_KR

#include "LocKeys.h"

namespace AELocalise {
namespace KO_KR {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_MODE_NAME:
    return "모드";
  case LocKey::STR_MODE_ITEMS:
    return "축 방향|자유 각도|회전|방사|나선|패스";
  case LocKey::STR_DIRECTION_NAME:
    return "방향";
  case LocKey::STR_DIRECTION_ITEMS:
    return "수평|수직";
  case LocKey::STR_ANGLE_NAME:
    return "각도";
  case LocKey::STR_CENTER_NAME:
    return "중심";
  case LocKey::STR_SWIRL_AMOUNT_NAME:
    return "나선량";
  case LocKey::STR_SWIRL_DIRECTION_NAME:
    return "회전 방향";
  case LocKey::STR_SWIRL_DIRECTION_ITEMS:
    return "시계 방향|반시계 방향";
  case LocKey::STR_PATH_NAME:
    return "패스";
  case LocKey::STR_PATH_DIRECTION_NAME:
    return "패스 방향";
  case LocKey::STR_PATH_DIRECTION_ITEMS:
    return "법선|접선";
  case LocKey::STR_SORT_CRITERION_NAME:
    return "정렬 기준";
  case LocKey::STR_SORT_CRITERION_ITEMS:
    return "휘도|RGB 평균|RGB 곱|RGB 최솟값|RGB 최댓값|빨강 채널|초록 채널|파랑 채널|알파 채널|색상|채도";
  case LocKey::STR_CRITERION_SOURCE_NAME:
    return "기준 소스 레이어";
  case LocKey::STR_SORT_TRIGGER_NAME:
    return "정렬 트리거";
  case LocKey::STR_SORT_TRIGGER_ITEMS:
    return "휘도|RGB 평균|RGB 곱|RGB 최솟값|RGB 최댓값|빨강 채널|초록 채널|파랑 채널|알파 채널|색상|채도";
  case LocKey::STR_TRIGGER_SOURCE_NAME:
    return "트리거 소스 레이어";
  case LocKey::STR_AFFECT_NAME:
    return "영향";
  case LocKey::STR_AFFECT_ITEMS:
    return "임계값 내부|임계값 외부";
  case LocKey::STR_CYCLE_NAME:
    return "순환";
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
  case LocKey::STR_GPU_STATUS_TRANSFORM_CPU_ONLY:
    return "GPU: CPU 폴백(안전한 변형 패스)";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU: 비활성(프로젝트 설정에서 Mercury GPU를 활성화하세요)";
  default:
    return "";
  }
}
} // namespace KO_KR
} // namespace AELocalise

#endif // STRINGS_KO_KR_H
