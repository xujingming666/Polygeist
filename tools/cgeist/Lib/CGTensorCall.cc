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


mlir::Value getConstantIndexValue(OpBuilder &builder, Location loc, int64_t value) {
  auto indexValue = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value));
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), indexValue);
}

llvm::SmallVector<int64_t> getStaticShape(mlir::Value value, int dim) {
  llvm::SmallVector<int64_t> shapes(dim, ShapedType::kDynamic);
  if (auto fromElem = value.getDefiningOp<tensor::FromElementsOp>()) {
    for (auto it : llvm::enumerate(fromElem.getOperands())) {
      if (auto shapeElem = it.value().getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = shapeElem.getValue().dyn_cast<IntegerAttr>();
        shapes[it.index()] = intAttr.getValue().getSExtValue();
      }
    }
  }
  return shapes;
}

const clang::FunctionDecl *getCallee(const clang::Expr *E) {
  E = E->IgnoreParens();
  // Look through function-to-pointer decay.
  if (auto ICE = dyn_cast<clang::ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == clang::CK_FunctionToPointerDecay ||
        ICE->getCastKind() == clang::CK_BuiltinFnToFnPtr) {
      return getCallee(ICE->getSubExpr());
    }

    // Resolve direct calls.
  } else if (auto DRE = dyn_cast<clang::DeclRefExpr>(E)) {
    if (auto FD = dyn_cast<clang::FunctionDecl>(DRE->getDecl())) {
      return FD;
    }

  } else if (auto ME = dyn_cast<clang::MemberExpr>(E)) {
    if (auto FD = dyn_cast<clang::FunctionDecl>(ME->getMemberDecl())) {
      // TODO EmitIgnoredExpr(ME->getBase());
      return FD;
    }

    // Look through template substitutions.
  } else if (auto NTTP = dyn_cast<clang::SubstNonTypeTemplateParmExpr>(E)) {
    return getCallee(NTTP->getReplacement());
  }

  return nullptr;  
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitTensorBinaryOps(clang::CallExpr *expr, OpBuilder &builder, Location loc, linalg::BinaryFn fn) {
  llvm::SmallVector<mlir::Value, 4> argValues;
  for (clang::Expr *argExpr : llvm::ArrayRef<clang::Expr *>(expr->getArgs(), expr->getNumArgs())) {
    argValues.push_back(
      Visit(argExpr).getValue(loc, builder));
  }
  auto lhsType = dyn_cast<mlir::RankedTensorType>(argValues[0].getType());
  auto lhsShape = lhsType.getShape();
  auto elementType = lhsType.getElementType();

  llvm::SmallVector<mlir::Value> dynamicShapes;
  for (auto it : llvm::enumerate(lhsShape)) {
    if (it.value() == ShapedType::kDynamic) {
      auto mValue = builder.create<tensor::DimOp>(loc, argValues[0], it.index());
      dynamicShapes.push_back(mValue);
    }
  }
  auto tensorType = RankedTensorType::get(lhsShape, elementType);
  auto allocTensor = builder.create<tensor::EmptyOp>(loc, tensorType, dynamicShapes);
  auto matmul_result = builder.create<linalg::ElemwiseBinaryOp>(loc, TypeRange{lhsType},
                                  ValueRange{argValues[0], argValues[1]}, ValueRange{allocTensor},
                                  linalg::BinaryFnAttr::get(builder.getContext(), fn),
                                  linalg::TypeFnAttr::get(builder.getContext(), linalg::TypeFn::cast_signed));
  return make_pair(ValueCategory(matmul_result.getResults()[0], true), true);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitTensorUnaryOps(clang::CallExpr *expr, OpBuilder &builder, Location loc, linalg::UnaryFn fn) {
  llvm::SmallVector<mlir::Value, 4> argValues;
  for (clang::Expr *argExpr : llvm::ArrayRef<clang::Expr *>(expr->getArgs(), expr->getNumArgs())) {
    argValues.push_back(
      Visit(argExpr).getValue(loc, builder));
  }
  auto lhsType = dyn_cast<mlir::RankedTensorType>(argValues[0].getType());
  auto lhsShape = lhsType.getShape();
  auto elementType = lhsType.getElementType();

  llvm::SmallVector<mlir::Value> dynamicShapes;
  for (auto it : llvm::enumerate(lhsShape)) {
    if (it.value() == ShapedType::kDynamic) {
      auto mValue = builder.create<tensor::DimOp>(loc, argValues[0], it.index());
      dynamicShapes.push_back(mValue);
    }
  }
  auto tensorType = RankedTensorType::get(lhsShape, elementType);
  auto allocTensor = builder.create<tensor::EmptyOp>(loc, tensorType, dynamicShapes);
  auto matmul_result = builder.create<linalg::ElemwiseUnaryOp>(loc, TypeRange{lhsType},
                                  ValueRange{argValues[0]}, ValueRange{allocTensor},
                                  linalg::UnaryFnAttr::get(builder.getContext(), fn),
                                  linalg::TypeFnAttr::get(builder.getContext(), linalg::TypeFn::cast_signed));
  return make_pair(ValueCategory(matmul_result.getResults()[0], true), true);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitTensorCallOps(clang::CallExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  auto fd = getCallee(expr->getCallee());

  if (!fd || !fd->getIdentifier()) {
    return make_pair(ValueCategory(), false);
  }

  if (fd->getName() == "to_tensor") {
    llvm::SmallVector<mlir::Value, 4> argValues;
    for (clang::Expr *argExpr : llvm::ArrayRef<clang::Expr *>(expr->getArgs(), expr->getNumArgs())) {
      argValues.push_back(
        Visit(argExpr).getValue(loc, builder));
    }

    auto memrefType = dyn_cast<mlir::UnrankedMemRefType>(argValues[0].getType());
    auto unrankedTensorType = UnrankedTensorType::get(memrefType.getElementType());
    argValues[0] = builder.create<mlir::bufferization::ToTensorOp>(loc, unrankedTensorType, argValues[0], true, true);

    int dim = dyn_cast<mlir::RankedTensorType>(argValues[1].getType()).getShape()[0];
    llvm::SmallVector<int64_t> shape = getStaticShape(argValues[1], dim);
    auto tensorType = RankedTensorType::get(shape, memrefType.getElementType(),
                        builder.getI64IntegerAttr(memrefType.getMemorySpaceAsInt()));
    mlir::Value reshapeTensor = builder.create<mlir::tensor::ReshapeOp>(loc, tensorType, argValues[0], argValues[1]);
    
    auto retValue = ValueCategory(reshapeTensor, true);

    auto funcArg = expr->getArgs()[0];
    if (auto refDecl = getRelVarDecl(funcArg)) {
      retValue.setDecl(refDecl);
    }
    // reference value would not trigger alloc.
    return make_pair(retValue, true);
  }

  if (fd->getName() == "mac_load") {
    llvm::SmallVector<mlir::Value, 4> argValues;
    for (clang::Expr *argExpr : llvm::ArrayRef<clang::Expr *>(expr->getArgs(), expr->getNumArgs())) {
      argValues.push_back(
        Visit(argExpr).getValue(loc, builder));
    }

    auto tensorType = dyn_cast<mlir::RankedTensorType>(argValues[0].getType());
    int dim = tensorType.getShape().size();
    auto elementType = tensorType.getElementType();

    SmallVector<int64_t> resultShape = getStaticShape(argValues[1], dim);
    SmallVector<int64_t> strideValueStatic(dim, 1);
    SmallVector<int64_t> sizeValueStatic(dim, ShapedType::kDynamic),
                         offsetValueStatic(dim, ShapedType::kDynamic);
    auto resultType = mlir::RankedTensorType::get(resultShape, elementType);
    SmallVector<mlir::Value> sizeValueDynamic, offsetValueDynamic;
    for (int i = 0; i < dim; i++) {
      auto indexValue = getConstantIndexValue(builder, loc, i);
      sizeValueDynamic.push_back(
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
          builder.create<tensor::ExtractOp>(loc, argValues[1], ValueRange{indexValue})));
      offsetValueDynamic.push_back(
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
          builder.create<tensor::ExtractOp>(loc, argValues[2], ValueRange{indexValue})));
    }

    auto loadTensor = builder.create<tensor::ExtractSliceOp>(loc, resultType,
                          argValues[0], offsetValueDynamic, sizeValueDynamic, ValueRange{},
                                        offsetValueStatic, sizeValueStatic, strideValueStatic);
    return make_pair(ValueCategory(loadTensor, true), true);
  }

  if (fd->getName() == "mac_store") {
    llvm::SmallVector<mlir::Value, 4> argValues;
    for (clang::Expr *argExpr : llvm::ArrayRef<clang::Expr *>(expr->getArgs(), expr->getNumArgs())) {
      argValues.push_back(
        Visit(argExpr).getValue(loc, builder));
    }

    auto sizeValueStatic = dyn_cast<RankedTensorType>(argValues[1].getType()).getShape();
    auto tensorType = dyn_cast<mlir::RankedTensorType>(argValues[0].getType());
    int dim = tensorType.getShape().size();
    auto elementType = tensorType.getElementType();

    SmallVector<int64_t> resultShape(dim, ShapedType::kDynamic);
    SmallVector<int64_t> strideValueStatic(dim, 1), offsetValueStatic(dim, ShapedType::kDynamic);
    auto resultType = mlir::RankedTensorType::get(resultShape, elementType);
    SmallVector<mlir::Value> offsetValueDynamic, sizeValueDynamic;
    for (int i = 0; i < dim; i++) {
      auto indexValue = getConstantIndexValue(builder, loc, i);
      offsetValueDynamic.push_back(
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
          builder.create<tensor::ExtractOp>(loc, argValues[2], ValueRange{indexValue}))
      );
      if (sizeValueStatic[i] == ShapedType::kDynamic)
        sizeValueDynamic.push_back(builder.create<tensor::DimOp>(loc, argValues[1], i));
    }

    auto storeTensor = builder.create<tensor::InsertSliceOp>(loc, argValues[0].getType(), argValues[1], argValues[0],
                                      offsetValueDynamic, sizeValueDynamic, ValueRange{},
                                      offsetValueStatic, sizeValueStatic, strideValueStatic);
    auto storeValue = ValueCategory(storeTensor, true);
    if (auto dstVarRef = dyn_cast<clang::DeclRefExpr>(expr->getArgs()[0])) {
        updateRedefinedLocalParams(dstVarRef->getDecl(), storeValue);
    }
    return make_pair(storeValue, true);
  }

  //emit binary ops
  if (fd->getName() == "mac_add") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::add);
  }
  if (fd->getName() == "mac_sub") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::sub);
  }
  if (fd->getName() == "mac_mul") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::mul);
  }
  if (fd->getName() == "mac_div") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::div);
  }
  if (fd->getName() == "mac_div_unsigned") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::div_unsigned);
  }
  if (fd->getName() == "mac_min_signed") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::min_signed);
  }
  if (fd->getName() == "mac_max_signed") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::max_signed);
  }
  if (fd->getName() == "mac_min_unsigned") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::min_unsigned);
  }
  if (fd->getName() == "mac_max_unsigned") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::max_unsigned);
  }
  if (fd->getName() == "mac_powf") {
    return EmitTensorBinaryOps(expr, builder, loc, linalg::BinaryFn::powf);
  }

  // emit unary ops
  if (fd->getName() == "mac_exp") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::exp);
  }
  if (fd->getName() == "mac_log") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::log);
  }
  if (fd->getName() == "mac_abs") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::abs);
  }
  if (fd->getName() == "mac_ceil") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::ceil);
  }
  if (fd->getName() == "mac_floor") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::floor);
  }
  if (fd->getName() == "mac_negf") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::negf);
  }
  if (fd->getName() == "mac_reciprocal") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::reciprocal);
  }
  if (fd->getName() == "mac_round") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::round);
  }
  if (fd->getName() == "mac_sqrt") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::sqrt);
  }
  if (fd->getName() == "mac_rsqrt") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::rsqrt);
  }
  if (fd->getName() == "mac_square") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::square);
  }
  if (fd->getName() == "mac_tanh") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::tanh);
  }
  if (fd->getName() == "mac_erf") {
    return EmitTensorUnaryOps(expr, builder, loc, linalg::UnaryFn::erf);
  }

  if (fd->getName() == "mac_fill") {
    llvm::SmallVector<mlir::Value, 4> argValues;
    for (clang::Expr *argExpr : llvm::ArrayRef<clang::Expr *>(expr->getArgs(), expr->getNumArgs())) {
      argValues.push_back(
        Visit(argExpr).getValue(loc, builder));
    }

    auto elementType = argValues[0].getType();
    int dim = dyn_cast<mlir::RankedTensorType>(argValues[1].getType()).getShape()[0];
    SmallVector<int64_t> resultShape = getStaticShape(argValues[1], dim);
    auto resultType = RankedTensorType::get(resultShape, elementType);
    
    SmallVector<mlir::Value> dynamicShapes;
    for (int i = 0; i < dim; i++) {
      if (resultShape[i] == ShapedType::kDynamic) {
        auto indexValue = getConstantIndexValue(builder, loc, i);
        dynamicShapes.push_back(
          builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
            builder.create<tensor::ExtractOp>(loc, argValues[1], ValueRange{indexValue}))
        );
      }
    }

    auto allocTensor = builder.create<tensor::EmptyOp>(loc, resultType, dynamicShapes);
    auto fillOp = builder.create<mlir::linalg::FillOp>(loc, 
                    TypeRange{resultType}, ValueRange{argValues[0]}, ValueRange{allocTensor});

    return make_pair(ValueCategory(fillOp.getResults()[0], true), true);
  }

  return make_pair(ValueCategory(), false);
}