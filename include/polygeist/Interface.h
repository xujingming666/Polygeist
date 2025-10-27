#ifndef POLYGEIST_INTERFACE_H
#define POLYGEIST_INTERFACE_H

#include "polygeist/TilingInterface.h.inc"

namespace mlir {
namespace polygeist {
    void registerTilingInterfaceExternalModels(DialectRegistry &registry);
}
}

#endif //POLYGEIST_INTERFACE_H