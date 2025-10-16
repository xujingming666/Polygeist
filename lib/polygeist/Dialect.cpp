//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Support/TypeID.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "polygeist/PolygeistOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "polygeist/PolygeistAttributes.cpp.inc"
    >();
}

#include "polygeist/PolygeistOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "polygeist/PolygeistAttributes.cpp.inc"
