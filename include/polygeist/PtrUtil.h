#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"


static inline mlir::Type traverseStructType(mlir::Type baseType, mlir::ArrayRef<int> indices) {
  mlir::Type currentType = baseType;
  
  for (auto index : llvm::enumerate(indices)) {
    if (index.index() == 0)
      continue;
  
    if (index.value() == mlir::ShapedType::kDynamic) {
      return nullptr;
    }
    
    if (auto structType = currentType.dyn_cast<mlir::LLVM::LLVMStructType>()) {
      if (index.value() < 0 || index.value() >= structType.getBody().size()) {
        llvm::errs() << "Warning: Struct index out of bounds: " << index.value() << "\n";
        return nullptr;
      }
      currentType = structType.getBody()[index.value()];
    } else if (auto arrayType = currentType.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
      currentType = arrayType.getElementType();
    }  else {
      return currentType;
    }
  }
  return currentType;
}

static inline mlir::Type getGEPResultType(mlir::LLVM::GEPOp gepOp) {
  mlir::Type elemType = gepOp.getElemType();
  if (!elemType) {
    return nullptr;
  }
  auto indices = gepOp.getRawConstantIndices();
  if (!indices.size()) {
    return elemType;
  }
  return traverseStructType(elemType, indices);
}

static inline mlir::Type getValuePtrType(mlir::Value value) {
  assert(mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()) && " value type should be LLVMPointerType \n");
  if (auto op = value.getDefiningOp<mlir::LLVM::AllocaOp>()) {
    return op.getElemType();
  }
  if (auto op = value.getDefiningOp<mlir::polygeist::Memref2PointerOp>()) {
    return mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType()).getElementType();
  }
  if (auto op = value.getDefiningOp<mlir::LLVM::GlobalOp>()) {
    return op.getType();
  }
  // if (auto op = value.getDefiningOp<mlir::LLVM::LLVMFuncOp>()) {
  //   return op.getFunctionType();
  // }
  if (auto op = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
    return getValuePtrType(op.getArgMutable().get());
  }
  if (auto op = value.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto globalOp = mlir::SymbolTable::lookupSymbolIn(moduleOp, op.getGlobalName());
    if (mlir::isa<mlir::LLVM::GlobalOp>(globalOp))
      return mlir::dyn_cast<mlir::LLVM::GlobalOp>(globalOp).getType();
  }

  if (auto op = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
    if (auto type = getGEPResultType(op))
      return type;
  }

  value.dump();
  assert(false && "getValuePtrType, can not get element type ");

  return nullptr;
}
