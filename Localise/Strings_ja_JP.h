#ifndef STRINGS_JA_JP_H
#define STRINGS_JA_JP_H

#define AELOCALISE_HAS_JA_JP

#include "LocKeys.h"

namespace AELocalise {
namespace JA_JP {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_MODE_NAME:
    return "モード";
  case LocKey::STR_MODE_ITEMS:
    return "軸方向|自由角度|回転|放射";
  case LocKey::STR_DIRECTION_NAME:
    return "方向";
  case LocKey::STR_DIRECTION_ITEMS:
    return "水平|垂直";
  case LocKey::STR_ANGLE_NAME:
    return "角度";
  case LocKey::STR_CENTER_NAME:
    return "中心";
  case LocKey::STR_SORT_CRITERION_NAME:
    return "ソート基準";
  case LocKey::STR_SORT_CRITERION_ITEMS:
    return "輝度|RGB平均|RGB積|RGB最小|RGB最大";
  case LocKey::STR_ORDER_NAME:
    return "並び順";
  case LocKey::STR_ORDER_ITEMS:
    return "昇順|降順";
  case LocKey::STR_THRESHOLD_MIN:
    return "しきい値（下限）";
  case LocKey::STR_THRESHOLD_MAX:
    return "しきい値（上限）";
  case LocKey::STR_GPU_STATUS_NAME:
    return "GPUアクセラレーション";
  case LocKey::STR_GPU_STATUS_ACTIVE:
    return "GPU: 有効";
  case LocKey::STR_GPU_STATUS_NO_BACKEND:
    return "GPU: 利用不可（GPUバックエンドなし）";
  case LocKey::STR_GPU_STATUS_HOST:
    return "GPU: 利用不可（ホスト）";
  case LocKey::STR_GPU_STATUS_AXIS_TOO_LONG:
    return "GPU: CPUフォールバック（画像が大きすぎます）";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU: 無効（プロジェクト設定でMercury GPUを有効にしてください）";
  default:
    return "";
  }
}
} // namespace JA_JP
} // namespace AELocalise

#endif // STRINGS_JA_JP_H
