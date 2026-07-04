#ifndef STRINGS_ZH_CN_H
#define STRINGS_ZH_CN_H

#define AELOCALISE_HAS_ZH_CN

#include "LocKeys.h"

namespace AELocalise {
namespace ZH_CN {
inline const char *GetString(LocKey::Key key) {
  switch (key) {
  case LocKey::STR_MODE_NAME:
    return "模式";
  case LocKey::STR_MODE_ITEMS:
    return "轴向|自由角度|旋转|放射|螺旋|路径";
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
    return "旋转方向";
  case LocKey::STR_SWIRL_DIRECTION_ITEMS:
    return "顺时针|逆时针";
  case LocKey::STR_PATH_NAME:
    return "路径";
  case LocKey::STR_PATH_DIRECTION_NAME:
    return "路径方向";
  case LocKey::STR_PATH_DIRECTION_ITEMS:
    return "法线|切线";
  case LocKey::STR_SORT_CRITERION_NAME:
    return "排序标准";
  case LocKey::STR_SORT_CRITERION_ITEMS:
    return "亮度|RGB平均值|RGB乘积|RGB最小值|RGB最大值|红色通道|绿色通道|蓝色通道|Alpha通道|色相|饱和度";
  case LocKey::STR_SORT_TRIGGER_NAME:
    return "排序触发";
  case LocKey::STR_SORT_TRIGGER_ITEMS:
    return "亮度|RGB平均值|RGB乘积|RGB最小值|RGB最大值|红色通道|绿色通道|蓝色通道|Alpha通道|色相|饱和度";
  case LocKey::STR_AFFECT_NAME:
    return "影响";
  case LocKey::STR_AFFECT_ITEMS:
    return "阈值内|阈值外";
  case LocKey::STR_CYCLE_NAME:
    return "循环";
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
  case LocKey::STR_GPU_STATUS_TRANSFORM_CPU_ONLY:
    return "GPU：CPU 回退（安全变形路径）";
  case LocKey::STR_GPU_STATUS_HOST_INACTIVE:
    return "GPU：未启用（请在项目设置中启用 Mercury GPU）";
  default:
    return "";
  }
}
} // namespace ZH_CN
} // namespace AELocalise

#endif // STRINGS_ZH_CN_H
