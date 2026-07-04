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
    return "軸方向|自由角度|回転|放射|螺旋|パス";
  case LocKey::STR_DIRECTION_NAME:
    return "方向";
  case LocKey::STR_DIRECTION_ITEMS:
    return "水平|垂直";
  case LocKey::STR_ANGLE_NAME:
    return "角度";
  case LocKey::STR_CENTER_NAME:
    return "中心";
  case LocKey::STR_SWIRL_AMOUNT_NAME:
    return "螺旋量";
  case LocKey::STR_SWIRL_DIRECTION_NAME:
    return "回転方向";
  case LocKey::STR_SWIRL_DIRECTION_ITEMS:
    return "時計回り|反時計回り";
  case LocKey::STR_PATH_NAME:
    return "パス";
  case LocKey::STR_PATH_DIRECTION_NAME:
    return "パス方向";
  case LocKey::STR_PATH_DIRECTION_ITEMS:
    return "法線|タンジェント";
  case LocKey::STR_SORT_CRITERION_NAME:
    return "ソート基準";
  case LocKey::STR_SORT_CRITERION_ITEMS:
    return "輝度|RGB平均|RGB積|RGB最小|RGB最大|赤チャンネル|緑チャンネル|青チャンネル|アルファチャンネル|色相|彩度";
  case LocKey::STR_SORT_TRIGGER_NAME:
    return "ソートトリガー";
  case LocKey::STR_SORT_TRIGGER_ITEMS:
    return "輝度|RGB平均|RGB積|RGB最小|RGB最大|赤チャンネル|緑チャンネル|青チャンネル|アルファチャンネル|色相|彩度";
  case LocKey::STR_AFFECT_NAME:
    return "影響";
  case LocKey::STR_AFFECT_ITEMS:
    return "しきい値内|しきい値外";
  case LocKey::STR_CYCLE_NAME:
    return "循環";
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
  case LocKey::STR_GPU_STATUS_TRANSFORM_CPU_ONLY:
    return "GPU: CPUフォールバック（安全な変形パス）";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU: 無効（プロジェクト設定でMercury GPUを有効にしてください）";
  default:
    return "";
  }
}
} // namespace JA_JP
} // namespace AELocalise

#endif // STRINGS_JA_JP_H
