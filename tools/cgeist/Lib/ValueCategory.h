//===- ValueCategory.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_VALUE_CATEGORY
#define CLANG_MLIR_VALUE_CATEGORY

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

// Represents a rhs or lhs value.
class ValueCategory {
public:
  mlir::Value val;
  bool isReference;
  mlir::Type ptrType;

public:
  ValueCategory() : val(nullptr), isReference(false), ptrType(nullptr) {};
  ValueCategory(std::nullptr_t) : val(nullptr), isReference(false), ptrType(nullptr) {};
  ValueCategory(mlir::Value val, bool isReference);
  ValueCategory(mlir::Value val, bool isReference, mlir::Type type) : ValueCategory(val, isReference)  { ptrType = type; }

  // TODO: rename to 'loadVariable'? getValue seems to generic.
  mlir::Value getValue(mlir::Location loc, mlir::OpBuilder &builder) const;
  void store(mlir::Location loc, mlir::OpBuilder &builder,
             ValueCategory toStore, bool isArray) const;
  // TODO: rename to storeVariable?
  void store(mlir::Location loc, mlir::OpBuilder &builder,
             mlir::Value toStore) const;
  ValueCategory dereference(mlir::Location loc, mlir::OpBuilder &builder) const;
};

#endif
