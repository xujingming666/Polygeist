//===---- IfScope.cc - Create an if statement to guard loop boundaries ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "IfScope.h"
#include "clang-mlir.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

IfScope::IfScope(MLIRScanner &scanner, bool isRealScope, bool isIfScope) : scanner(scanner), prevBlock(nullptr), isRealScope(isRealScope), isIfScope(isIfScope) {
  if (scanner.loops.size() && scanner.loops.back().keepRunning) {
    auto loc = scanner.builder.getUnknownLoc();
    auto lop = scanner.builder.create<memref::LoadOp>(
        loc, scanner.loops.back().keepRunning);
    ifOp = scanner.builder.create<scf::IfOp>(loc, lop,
                                                  /*hasElse*/ false);
    prevBlock = scanner.builder.getInsertionBlock();
    prevIterator = scanner.builder.getInsertionPoint();
    ifOp.getThenRegion().back().clear();
    scanner.builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
    er = scanner.builder.create<scf::ExecuteRegionOp>(
        loc, ArrayRef<mlir::Type>());
    scanner.builder.create<scf::YieldOp>(loc);
    er.getRegion().push_back(new Block());
    scanner.builder.setInsertionPointToStart(&er.getRegion().back());
  }
  scanner.ifScopeStacks.push_back(this);
}

llvm::SmallVector<mlir::Value, 4> IfScope::getDynamicValues(mlir::Location &loc, mlir::OpBuilder & builder, mlir::RankedTensorType type) {
  llvm::SmallVector<mlir::Value, 4> dynamicValues;
  for (int64_t shape : type.getShape()) {
    if (shape == ShapedType::kDynamic) {
      dynamicValues.push_back(
        builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(builder.getI32Type(), 0))
      );
    }
  }
  return dynamicValues;
}

IfScope::~IfScope() {
  mlir::OpBuilder &builder = scanner.builder;
  auto loc = builder.getUnknownLoc();
  if (scanner.loops.size() && scanner.loops.back().keepRunning) {
    if (yieldParams.size() > 0) {
      llvm::SmallVector<mlir::Type> types;
      llvm::SmallVector<mlir::Value> yiledValues;
      for (auto it : yieldParams) {
        auto yieldValue = it.second.getValue(loc, builder);
        types.push_back(yieldValue.getType());
        yiledValues.push_back(yieldValue);
      }
      builder.create<scf::YieldOp>(loc, yiledValues);

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(ifOp);
      auto newIfOp = scanner.builder.create<scf::IfOp>(loc, types, ifOp.getCondition(),
                                                  /*hasElse*/ true);
      newIfOp.getThenRegion().back().clear();
      builder.setInsertionPointToStart(&newIfOp.getThenRegion().back());
      auto newer = builder.create<scf::ExecuteRegionOp>(loc, types);
      mlir::IRMapping mapping;
      er.getRegion().cloneInto(&(newer.getRegion()), newer.getRegion().begin(), mapping);
      auto thenYieldOp = builder.create<scf::YieldOp>(loc, newer.getResults());

      newIfOp.getElseRegion().back().clear();
      builder.setInsertionPointToStart(&newIfOp.getElseRegion().back());

      llvm::SmallVector<mlir::Value, 4> retValues;
      for (auto result : newer.getResults()) {
        mlir::RankedTensorType type = dyn_cast<mlir::RankedTensorType>(result.getType());
        llvm::SmallVector<mlir::Value, 4> dynamicValues = getDynamicValues(loc, builder, type);
        retValues.push_back(builder.create<tensor::EmptyOp>(loc, type, dynamicValues));
      }
      builder.create<scf::YieldOp>(loc, retValues);

      for (auto it : llvm::enumerate(yieldParams)) {
        auto decl = it.value().first;
        auto newValue = ValueCategory(newIfOp.getResults()[it.index()], true);
        if (scanner.ifScopeStacks.size() > 1) {
          // update ifscope include this ifscope, so updateRedefinedLocalParams can not be used directly. 
          auto it = scanner.ifScopeStacks.rbegin() + 1;
          if ((*it)->localParams.count(decl) > 0) {
            if ((*it)->isRealScope) {
              (*it)->localParams[decl] = newValue;
              if ((*it)->yieldParams.count(decl) > 0)
                (*it)->yieldParams[decl] = newValue;
            } else {
              (*it)->yieldParams[decl] = newValue;
            }
          } else {
            (*it)->yieldParams[decl] = newValue;
            bool isModified = false;
            for (; it != scanner.ifScopeStacks.rend(); it++) {
              if (!(*it)->isRealScope)
                continue;
              if ((*it)->isIfScope) {
                (*it)->localParams[decl] = newValue;
                isModified = true;
                break;
              }
              if ((*it)->localParams.count(decl) > 0) {
                (*it)->localParams[decl] = newValue;
                isModified = true;
                break;
              }
            }

            if (!isModified)
              scanner.params[decl] = newValue;
          }
        } else
          scanner.params[decl] = newValue;
      }
      ifOp->erase();
    } else {
      builder.create<scf::YieldOp>(loc);
    }
    scanner.builder.setInsertionPoint(prevBlock, prevIterator);
  }

  localParams.clear();
  yieldParams.clear();

  scanner.ifScopeStacks.pop_back();
}
