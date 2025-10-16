//===- BFVDialect.h - BFV dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BFV_BFVDIALECT_H
#define BFV_BFVDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/TypeID.h"

#include "polygeist/PolygeistOpsDialect.h.inc"


#define GET_ATTRDEF_CLASSES
#include "polygeist/PolygeistAttributes.h.inc"

#endif // BFV_BFVDIALECT_H
