
#ifndef POLYGEIST_INTERFACE
#define POLYGEIST_INTERFACE

include "mlir/IR/OpBase.td"

def TilingInterface : OpInterface<"TilingInterface"> {
  let description = [{
    tiling interface
  }];
  let cppNamespace = "::mlir::polygeist";
  let methods = [
    InterfaceMethod<" infer tiling info ", "llvm::SmallVector<int64_t>", "getFuseTileSize",
      (ins)
    >,
    InterfaceMethod<" infer tiling info ", "llvm::SmallVector<int64_t>", "getTileSize",
      (ins)
    >
  ];
}

#endif // POLYGEIST_INTERFACE
