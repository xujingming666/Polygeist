#include "TypeUtils.h"
#include "clang-mlir.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "utils.h"
#include "polygeist/Passes/Utils.h"
#include "polygeist/PtrUtil.h"
#include "clang/Basic/Builtins.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"

using namespace std;
using namespace clang;
using namespace llvm;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;
using namespace mlir::arith;
using namespace mlirclang;

ValueCategory MLIRScanner::VisitTensorCXXOperatorCallExpr(clang::CXXOperatorCallExpr *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());
  auto retType = getMLIRType(BO->getType());

  if (BO->getNumArgs() == 2) {
    auto lhs = Visit(BO->getArgs()[0]);
    auto rhs = Visit(BO->getArgs()[1]);
    mlir::Value lhsValue = lhs.getValue(loc, builder);
    mlir::Value rhsValue = rhs.getValue(loc, builder);

    if (BO->getOperator() == clang::OverloadedOperatorKind::OO_Equal) {
      if (auto lhsVarRef = dyn_cast<clang::DeclRefExpr>(BO->getArgs()[0])) {
        updateRedefinedLocalParams(lhsVarRef->getDecl(), ValueCategory(rhsValue, true));
      }
      return ValueCategory(rhsValue, true);
    }

    if (BO->getOperator() == clang::OverloadedOperatorKind::OO_PlusEqual) {
      auto lhsType = dyn_cast<mlir::RankedTensorType>(lhsValue.getType());
      auto lhsShape = lhsType.getShape();
      auto elementType = lhsType.getElementType();

      llvm::SmallVector<mlir::Value> dynamicShapes;
      for (auto it : llvm::enumerate(lhsShape)) {
        if (it.value() == ShapedType::kDynamic) {
          auto mValue = builder.create<tensor::DimOp>(loc, lhsValue, it.index());
          dynamicShapes.push_back(mValue);
        }
      }
      auto tensorType = RankedTensorType::get(lhsShape, elementType, builder.getI64IntegerAttr(0));
      auto allocTensor = builder.create<tensor::EmptyOp>(loc, tensorType, dynamicShapes);
      auto matmul_result = builder.create<linalg::ElemwiseBinaryOp>(loc, TypeRange{lhsType},
                                      ValueRange{lhsValue, rhsValue}, ValueRange{allocTensor},
                                      linalg::BinaryFnAttr::get(builder.getContext(), linalg::BinaryFn::add),
                                      linalg::TypeFnAttr::get(builder.getContext(), linalg::TypeFn::cast_signed));
      auto resultValue = ValueCategory(matmul_result.getResult(0), true);
      if (auto lhsVarRef = dyn_cast<clang::DeclRefExpr>(BO->getArgs()[0])) {
        updateRedefinedLocalParams(lhsVarRef->getDecl(), resultValue);
      }
      return resultValue;
    }
  }

  BO->dump();
  llvm_unreachable("unhandled CXXOperatorCallExpr \n");
}

