#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace polygeist;
using namespace mlir::affine;

namespace {
struct GroupAnnotationPass : public GroupAnnotationBase<GroupAnnotationPass> {

  static int64_t groupId;

  void runOnOperation() override;
  
  void markAsGrouped(linalg::LinalgOp op, int64_t groupId) {
    OpBuilder builder(op);
    op->setAttr("group_id", builder.getI64IntegerAttr(groupId));
  }

  int64_t getGroupId(linalg::LinalgOp currentOp) {
    Operation* prevOp = currentOp->getPrevNode();
    while (prevOp && !isa<linalg::LinalgOp>(prevOp)) {
      if (isa<tensor::EmptyOp>(prevOp) ||
          isa<tensor::ExpandShapeOp>(prevOp) ||
          prevOp->getName().getStringRef().starts_with("arith."))
        prevOp = prevOp->getPrevNode();
      else
        break;
    }
    if (prevOp != nullptr &&
        isa<linalg::LinalgOp>(prevOp) &&
        prevOp->hasAttr("group_id")) {
      groupId = dyn_cast<IntegerAttr>(prevOp->getAttr("group_id")).getInt();
    } else 
      ++groupId;
    return groupId;
  }
};

int64_t GroupAnnotationPass::groupId = 1;

} // end namespace.

void GroupAnnotationPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  moduleOp.walk([&](linalg::LinalgOp linalgOp) {
    markAsGrouped(linalgOp, getGroupId(linalgOp));
  });
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> groupAnnotationPass() {
  return std::make_unique<GroupAnnotationPass>();
}
} // namespace polygeist
} // namespace mlir
