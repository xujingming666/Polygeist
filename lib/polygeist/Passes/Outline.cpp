#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace polygeist;
using namespace mlir::affine;

namespace {
struct OutlinePass : public OutlineBase<OutlinePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto &func : module.getBody()->getOperations()) {
      if (auto f = dyn_cast<func::FuncOp>(func)) {
        outlineOperationsInFunction(f, module);
      }
    }
  }

private:
  void outlineOperationsInFunction(func::FuncOp parentFunc, ModuleOp module) {
    std::map<int64_t, SmallVector<Operation*>> groupOps;
    parentFunc.walk([&](linalg::LinalgOp linalgOp) {
      int64_t groupId = dyn_cast<IntegerAttr>(linalgOp->getAttr("group_id")).getInt();
      if (groupOps.count(groupId) > 0) {
        groupOps[groupId].push_back(linalgOp);
      } else {
        groupOps[groupId] = {linalgOp};
      }
      linalgOp->removeAttr("group_id");
    });
    
    for(auto it : groupOps) {
      func::FuncOp outlinedFunc = createOutlinedFunction(it.second, module, parentFunc);
      replaceOpsWithCall(it.second, outlinedFunc, parentFunc);
    }
  }
  
  func::FuncOp createOutlinedFunction(ArrayRef<Operation*> ops,
                ModuleOp module, func::FuncOp parentFunc) {
    static int funcIdx = 0;

    OpBuilder builder(module.getBodyRegion());
    
    SmallVector<Type> inputTypes, outputTypes;
    SmallVector<Value> inputs, outputs;
    
    collectInputsOutputs(ops, inputs, outputs);
    for (Value input : inputs) inputTypes.push_back(input.getType());
    for (Value output : outputs) outputTypes.push_back(output.getType());
    
    SmallVector<Type> resultTypes(outputTypes.begin(), outputTypes.end());
    FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
    
    auto outlinedFunc = builder.create<func::FuncOp>(module.getLoc(),
          "mpu_" + parentFunc.getSymName().str() + "_" + std::to_string(funcIdx++), funcType);
    
    Block *entryBlock = outlinedFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    IRMapping mapping;
    for (unsigned i = 0; i < inputs.size(); ++i) {
      mapping.map(inputs[i], outlinedFunc.getArgument(i));
    }
    
    for (Operation *op : ops) {
      builder.clone(*op, mapping);
    }
    
    SmallVector<Value> returnValues;
    for (Value output : outputs) {
      returnValues.push_back(mapping.lookup(output));
    }
    builder.create<func::ReturnOp>(module.getLoc(), returnValues);
    
    return outlinedFunc;
  }
  
  void collectInputsOutputs(ArrayRef<Operation*> ops,
                           SmallVectorImpl<Value> &inputs,
                           SmallVectorImpl<Value> &outputs) {
    DenseSet<Value> inputSet, outputSet;
    DenseSet<Operation*> opSet(ops.begin(), ops.end());
    
    for (Operation *op : ops) {
      for (Value operand : op->getOperands()) {
        if (!opSet.contains(operand.getDefiningOp())) {
          inputSet.insert(operand);
        }
      }
      
      for (Value result : op->getResults()) {
        bool isOutput = false;
        for (Operation *user : result.getUsers()) {
          if (!opSet.contains(user)) {
            isOutput = true;
            break;
          }
        }
        if (isOutput) {
          outputSet.insert(result);
        }
      }
    }
    
    inputs.append(inputSet.begin(), inputSet.end());
    outputs.append(outputSet.begin(), outputSet.end());
  }
  
  void replaceOpsWithCall(ArrayRef<Operation*> ops,
                         func::FuncOp outlinedFunc,
                         func::FuncOp parentFunc) {
    OpBuilder builder(parentFunc.getBody());
    builder.setInsertionPointAfter(ops.back());
    
    SmallVector<Value> inputs, outputs;
    collectInputsOutputs(ops, inputs, outputs);
    
    auto callOp = builder.create<func::CallOp>(
        parentFunc.getLoc(), outlinedFunc, inputs);
    
    for (unsigned i = 0; i < outputs.size(); ++i) {
      outputs[i].replaceAllUsesWith(callOp.getResult(i));
    }
    
    for (Operation *op : llvm::reverse(ops)) {
      op->erase();
    }
  }
};

}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> outlinePass() {
  return std::make_unique<OutlinePass>();
}
} // namespace polygeist
} // namespace mlir
