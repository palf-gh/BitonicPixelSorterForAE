#pragma once

namespace AELocalise {
namespace Internal {
// AE 26 requires UTF-8 but predates the SDK flag. Later versions require the
// effect's explicit declaration; earlier versions use the locale encoding.
inline bool UseUTF8ForAEDriver(long major, bool flagDeclared) {
  return major == 126 || (major >= 127 && flagDeclared);
}
} // namespace Internal
} // namespace AELocalise
