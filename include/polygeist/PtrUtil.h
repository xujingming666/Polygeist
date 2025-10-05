#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

static inline mlir::Type getValuePtrType(mlir::Value value) {
  assert(mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()) && " value type should be LLVMPointerType \n");
  if (auto op = value.getDefiningOp<mlir::LLVM::AllocaOp>()) {
    return op.getElemType();
  }
  if (auto op = value.getDefiningOp<mlir::polygeist::Memref2PointerOp>()) {
    return mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType()).getElementType();
  }

  if (auto op = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
    getValuePtrType(op.getArgMutable().get());
  }

  value.dump();
  assert(false && "getValuePtrType, can not get element type ");

  return nullptr;
}
