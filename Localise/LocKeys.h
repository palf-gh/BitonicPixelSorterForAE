#ifndef LOC_KEYS_H
#define LOC_KEYS_H

// Localisation string keys. The popup "*_ITEMS" keys return a single
// "Item1|Item2" string (the form PF_ADD_POPUP expects), localised as a whole.
namespace LocKey {
enum Key {
  STR_DIRECTION_NAME,
  STR_DIRECTION_ITEMS,   // "Horizontal|Vertical"
  STR_ORDER_NAME,
  STR_ORDER_ITEMS,       // "Ascending|Descending"
  STR_THRESHOLD_MIN,
  STR_THRESHOLD_MAX,

  STR_NUM_KEYS
};
}

#endif // LOC_KEYS_H
