#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Interface.h"
#include "llvm/Support/Debug.h"

#include <numeric>

using namespace mlir;
using namespace polygeist;
using namespace mlir::affine;

#include "polygeist/TilingInterface.cpp.inc"

namespace {

static int64_t mergeTileSize(int tile1, int tile2) {
  if (tile1 == 0 || tile2 == 0) {
    return 0;
  }
  return std::lcm(tile1, tile2);
}

static SmallVector<int64_t> getValueTileSize(const mlir::Value &v) {
  auto *defOp = v.getDefiningOp();
  if (defOp) {
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(defOp))
      return getValueTileSize(expandOp.getSrc());
    auto interface = dyn_cast<mlir::polygeist::TilingInterface>(defOp);
    if (interface)
      return interface.getFuseTileSize();
  }
  return {};
}

template <typename OpType>
struct LinalgOpTilingInterface
    : public mlir::polygeist::TilingInterface::ExternalModel<LinalgOpTilingInterface<OpType>, OpType> {
  llvm::SmallVector<int64_t> getTileSize(mlir::Operation *op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    llvm::ArrayRef<int64_t> resultShape = returnType.getShape();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;

    if constexpr (std::disjunction_v<
        std::is_same<OpType, linalg::ElemwiseBinaryOp>,
        std::is_same<OpType, linalg::ElemwiseUnaryOp>,
        std::is_same<OpType, linalg::FillOp>
      >) {
      llvm::SmallVector<int64_t> tileSize{rank, 1};
      tileSize[rank - 1] = 64/bpe;
      return tileSize;
    }

    if constexpr (std::is_same_v<OpType, linalg::MatmulOp>) {
      return {8, 8, 64/bpe};
    }

    if constexpr (std::is_same_v<OpType, linalg::Conv2DOp>) {
      return {1, 64/bpe, 1, 8, 8, 1, 1};
    }

    // reduce op should not tile reduce axis.
    if constexpr (std::is_same_v<OpType, linalg::ReduceOp>) {
      auto inputType = dyn_cast<mlir::RankedTensorType>(op->getOperandTypes()[0]);
      int64_t inputRank = inputType.getRank();
      auto reduceOp = dyn_cast<linalg::ReduceOp>(op);
      auto reduceDims = reduceOp.getDimensions();
      assert(reduceDims.size() == 1 && "only 1 reduce dim is supported");
      
      if (std::all_of(resultShape.begin(), resultShape.end(), 
                      [&](const auto& element) {return element == 1;})) {
        llvm::SmallVector<int64_t> tileSize(inputRank, 0);
        return tileSize;
      } else {
        llvm::SmallVector<int64_t> tileSize(inputRank, 1);
        tileSize[reduceDims[0]] = 0;
        if (reduceDims[0] == (rank - 1)) {
          tileSize[rank - 2] = 64/bpe;
        } else {
          tileSize[rank - 1] = 64/bpe;
        }
        return tileSize;
      }
    }

    // broadcast op should not tile broadcast axis.
    if constexpr (std::is_same_v<OpType, linalg::BroadcastOp>) {
      auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op);
      auto broadcastDims = broadcastOp.getDimensions();
      assert(broadcastDims.size() == 1 && "only 1 reduce dim is supported");
      llvm::SmallVector<int64_t> tileSize{rank, 1};
      tileSize[broadcastDims[0]] = 0;
      if (broadcastDims[0] == (rank - 1)) {
        tileSize[rank - 2] = 64/bpe;
      } else {
        tileSize[rank - 1] = 64/bpe;
      }
      return tileSize;
    }
  }
  
  llvm::SmallVector<int64_t> getMatmulFuseTileSize(mlir::Operation *op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhsTileSize = getValueTileSize(lhs);
    auto rhsTileSize = getValueTileSize(rhs);
    llvm::SmallVector<int64_t> opTileSize = {8, 64/bpe};
    if (lhsTileSize.size() > 0) {
      opTileSize[0] = mergeTileSize(lhsTileSize[0], opTileSize[0]);
    }
    if (rhsTileSize.size() > 0) {
      opTileSize[1] = mergeTileSize(rhsTileSize[1], opTileSize[1]);
    }
    return opTileSize;
  }

  llvm::SmallVector<int64_t> getConv2DFuseTileSize(mlir::Operation *op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;

    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhsTileSize = getValueTileSize(lhs);
    auto rhsTileSize = getValueTileSize(rhs);
    llvm::SmallVector<int64_t> opTileSize = {1, 64/bpe, 1, 8};
    if (lhsTileSize.size() > 0) {
      opTileSize[0] = mergeTileSize(lhsTileSize[0], opTileSize[0]);
    }
    if (rhsTileSize.size() > 0) {
      opTileSize[1] = mergeTileSize(rhsTileSize[1], opTileSize[1]);
    }
    return opTileSize;
  }

  llvm::SmallVector<int64_t> getElemwiseFuseTileSize(mlir::linalg::ElemwiseBinaryOp op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;
    
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhsTileSize = getValueTileSize(lhs);
    auto rhsTileSize = getValueTileSize(rhs);
    llvm::SmallVector<int64_t> opTileSize(rank, 1);
    opTileSize[rank - 1] = 64/bpe;
    if (lhsTileSize.size() > 0) {
      for (int i = 0; i < rank; i++)
        opTileSize[i] = mergeTileSize(lhsTileSize[i], opTileSize[i]);
    }
    if (rhsTileSize.size() > 0) {
      for (int i = 0; i < rank; i++)
        opTileSize[i] = mergeTileSize(rhsTileSize[i], opTileSize[i]);
    }
    return opTileSize;
  }

  llvm::SmallVector<int64_t> getElemwiseFuseTileSize(mlir::linalg::ElemwiseUnaryOp op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;
    
    auto input = op->getOperand(0);
    auto inputTileSize = getValueTileSize(input);
    llvm::SmallVector<int64_t> opTileSize(rank, 1);
    opTileSize[rank - 1] = 64/bpe;
    if (inputTileSize.size() > 0) {
      for (int i = 0; i < rank; i++)
        opTileSize[i] = mergeTileSize(inputTileSize[i], opTileSize[i]);
    }
    return opTileSize;
  }

  llvm::SmallVector<int64_t> getFillTileSize(mlir::Operation *op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;

    llvm::SmallVector<int64_t> opTileSize(rank, 1);
    opTileSize[rank - 1] = 64/bpe;
    return opTileSize;
  }

  llvm::SmallVector<int64_t> getReduceFuseTileSize(mlir::linalg::ReduceOp op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;
    
    auto reduceDims = op.getDimensions();
    assert(reduceDims.size() == 1 && "only 1 reduce dim is supported");

    auto input = op->getOperand(0);
    auto inputTileSize = getValueTileSize(input);
    llvm::SmallVector<int64_t> opTileSize(rank, 1);
    if (inputTileSize.size() > 0) {
      for (int i = 0; i < rank; i++) {
        opTileSize[i] = mergeTileSize(inputTileSize[(i < reduceDims[0]) ? i : (i + 1)], opTileSize[i]);
      }
    }
    return opTileSize;
  }
  
  llvm::SmallVector<int64_t> getBroadcastFuseTileSize(mlir::linalg::BroadcastOp op) const {
    auto returnType = dyn_cast<mlir::RankedTensorType>(op->getResultTypes()[0]);
    int64_t rank = returnType.getRank();
    auto elementType = returnType.getElementType();
    int bpe = elementType.getIntOrFloatBitWidth()/8;
    
    auto broadcastDims = op.getDimensions();

    auto input = op->getOperand(0);
    auto inputTileSize = getValueTileSize(input);

    llvm::SmallVector<int64_t> opTileSize(rank, 1);
    for (auto bDim : broadcastDims) {
      opTileSize[bDim] = 0;
    }
    for (int i = rank - 1; i >= 0; i--) {
      if (std::find(broadcastDims.begin(), broadcastDims.end(), i) == broadcastDims.end()) {
        opTileSize[i] = 64/bpe;
        break;
      }
    }

    if (inputTileSize.size() > 0) {
      int bIdx = 0;
      for (int i = 0; i < opTileSize.size(); i++) {
        if (std::find(broadcastDims.begin(), broadcastDims.end(), i) != broadcastDims.end()) {
          bIdx++;
          continue;
        }
        opTileSize[i] = mergeTileSize(inputTileSize[i - bIdx], opTileSize[i]);
      }
    }
    return opTileSize;
  }

  // broadcast&reduce's reduce dim should not be tiled in Fuse Tiling.
  llvm::SmallVector<int64_t> getFuseTileSize(mlir::Operation *op) const {
    if constexpr (std::is_same_v<OpType, linalg::MatmulOp>) {
      return getMatmulFuseTileSize(op);
    }
    if constexpr (std::is_same_v<OpType, linalg::Conv2DOp>) {
      return getConv2DFuseTileSize(op);
    }
    if constexpr (std::is_same_v<OpType, linalg::Conv2DNchwFchwOp>) {
      return getConv2DFuseTileSize(op);
    }
    if constexpr (std::is_same_v<OpType, mlir::linalg::ElemwiseBinaryOp>) {
      return getElemwiseFuseTileSize(dyn_cast<mlir::linalg::ElemwiseBinaryOp>(op));
    }
    if constexpr (std::is_same_v<OpType, mlir::linalg::ElemwiseUnaryOp>) {
      return getElemwiseFuseTileSize(dyn_cast<mlir::linalg::ElemwiseUnaryOp>(op));
    }
    if constexpr (std::is_same_v<OpType, mlir::linalg::FillOp>) {
      return getFillTileSize(op);
    }
    if constexpr (std::is_same_v<OpType, mlir::linalg::ReduceOp>) {
      return getReduceFuseTileSize(dyn_cast<mlir::linalg::ReduceOp>(op));
    }
    if constexpr (std::is_same_v<OpType, mlir::linalg::BroadcastOp>) {
      return getBroadcastFuseTileSize(dyn_cast<mlir::linalg::BroadcastOp>(op));
    }
    assert(false && " Not Supported Linalg Op \n");
    return {};
  }
};

template <typename ...OpTypes>
void attachTilingInterface(MLIRContext *ctx) {
  (void)llvm::ArrayRef<int> {0, 
    (OpTypes::template attachInterface<LinalgOpTilingInterface<OpTypes>>(*ctx), 0)...};
}

static constexpr llvm::StringRef tileAttrName = "tile";

struct TilingPass : public TilingBase<TilingPass> {
  
  void runOnOperation() override;

  void applyTilingTransform(mlir::Operation *rootOp, llvm::SmallVector<int64_t> tileSize) {
    auto funcOp = getOperation();
    mlir::MLIRContext *context = &getContext();

    mlir::OpBuilder builder(context);
    rootOp->setAttr(tileAttrName, builder.getBoolAttr(true));
    llvm::StringRef tileOpName = rootOp->getName().getStringRef();

    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(builder.getUnknownLoc()));
    Location loc = module->getLoc();
    builder.setInsertionPointToStart(module->getBody());

    std::vector<int64_t> tileInterchange(tileSize.size());
    std::iota(tileInterchange.begin(), tileInterchange.end(), 0);

    auto anyType = transform::AnyOpType::get(builder.getContext());
    llvm::SmallVector<mlir::Type> loopTypes(
      std::count_if(tileSize.begin(), tileSize.end(), 
                    [](int64_t x) { return x != 0; }), 
      anyType);
    auto sequenceOp = builder.create<transform::SequenceOp>(
        loc, TypeRange{}, transform::FailurePropagationMode::Propagate, anyType,
        [&](OpBuilder &b, Location loc, mlir::BlockArgument root) {
          mlir::MLIRContext *ctx = b.getContext();
          auto matchFuncOp = b.create<transform::MatchOp>(loc,
              mlir::TypeRange{anyType}, root,
              b.getArrayAttr(llvm::ArrayRef<Attribute>{b.getStringAttr(tileOpName.data())}),
              nullptr,
              b.getDictionaryAttr(llvm::ArrayRef<NamedAttribute>{b.getNamedAttr(tileAttrName, b.getBoolAttr(true))}),
              nullptr,
              nullptr);

          auto fuseOp = b.create<transform::FuseOp>(loc, anyType, loopTypes,
                                      matchFuncOp->getResult(0),
                                      builder.getI64ArrayAttr(llvm::ArrayRef<int64_t>{tileSize}),
                                      builder.getI64ArrayAttr(llvm::ArrayRef<int64_t>{tileInterchange}),
                                      builder.getBoolAttr(true));
          b.create<transform::ApplyPatternsOp>(loc, root,
            [&](OpBuilder &b, Location loc) {
              b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
            });
          b.create<transform::ApplyCommonSubexpressionEliminationOp>(loc, root);

          b.create<transform::YieldOp>(loc);
        });
    
    if (failed(transform::applyTransforms(funcOp, sequenceOp)))
      return signalPassFailure();
  }

  void mergeOpIntoAnchorOps(llvm::SmallVector<mlir::Operation *> &anchorOps, Operation *op) {
    bool shouldIn = true;
    for (mlir::Operation * anchorOp : anchorOps) {
      llvm::SetVector<mlir::Operation *> anchorOpPreds;
      mlir::getBackwardSlice(anchorOp, &anchorOpPreds);

      if (anchorOpPreds.count(op) > 0)
        return;

      llvm::SetVector<mlir::Operation *> opPreds;
      mlir::getBackwardSlice(op, &opPreds);

      if (opPreds.count(anchorOp) > 0) {
        anchorOps.erase(std::find(anchorOps.begin(), anchorOps.end(), anchorOp));
        anchorOps.push_back(op);
        return;
      }
    }
    anchorOps.push_back(op);
  }
};

} // end namespace.

void TilingPass::runOnOperation() {
  mlir::Operation *funcOp = getOperation();
  if (!funcOp->hasAttr("mpu"))
    return;
  llvm::SmallVector<mlir::Operation *> rootOps;
  funcOp->walk([&](func::ReturnOp retOp) {
    for (auto opreand : retOp->getOperands()) {
      if (auto defOp = opreand.getDefiningOp<linalg::LinalgOp>()) {
        mergeOpIntoAnchorOps(rootOps, defOp.getOperation());
      }
    }
  });

  for (mlir::Operation *op : rootOps) {
    auto interface = dyn_cast<mlir::polygeist::TilingInterface>(op);
    if (interface) {
      auto tileSize = interface.getFuseTileSize();
      applyTilingTransform(op, tileSize);
    }
  }
}

void mlir::polygeist::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    attachTilingInterface<
      linalg::ElemwiseBinaryOp,
      linalg::ElemwiseUnaryOp,
      linalg::FillOp,
      linalg::MatmulOp,
      linalg::Conv2DOp,
      linalg::Conv2DNchwFchwOp,
      linalg::ReduceOp,
      linalg::BroadcastOp
    >(ctx);
  });
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> tilingPass() {
  return std::make_unique<TilingPass>();
}
} // namespace polygeist
} // namespace mlir
