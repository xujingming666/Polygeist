add_mlir_dialect(PolygeistOps polygeist)
add_mlir_doc(PolygeistDialect -gen-dialect-doc PolygeistDialect Polygeist/)
add_mlir_doc(PolygeistOps -gen-op-doc PolygeistOps Polygeist/)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(PolygeistAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(PolygeistAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS Interface.td)
mlir_tablegen(TilingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TilingInterface.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(PolygeistAttributesIncGen)

add_subdirectory(Passes)