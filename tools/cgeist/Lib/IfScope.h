//===---- IfScope.h - Create an if statement to guard loop boundaries  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef IF_SCOPE_H_
#define IF_SCOPE_H_

#include "ValueCategory.h"

#include "mlir/IR/Block.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/StmtVisitor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

class MLIRScanner;
class MLIRASTConsumer;

class IfScope {
public:
  bool isRealScope;
  bool isIfScope;
  MLIRScanner &scanner;
  mlir::Block *prevBlock;
  mlir::Block::iterator prevIterator;
  mlir::scf::IfOp  ifOp;
  mlir::scf::ExecuteRegionOp er;
  
  std::map<const clang::ValueDecl *, ValueCategory> localParams;
  std::map<const clang::ValueDecl *, ValueCategory> yieldParams;
  std::map<const clang::ValueDecl *, ValueCategory> entryParams;

  llvm::SmallVector<mlir::Value, 4> getDynamicValues(mlir::Location &loc, mlir::OpBuilder & builder, mlir::RankedTensorType type);
  IfScope(MLIRScanner &scanner, bool isRealScope = true, bool isIfScope = false);
  void collectEntryInfo(clang::Stmt *stmt, MLIRASTConsumer &astContext);
  ~IfScope();
};

#endif
