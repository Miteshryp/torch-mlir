
#ifndef MLIR_CONVERSION_GEN_PARAM_METADATA_H
#define MLIR_CONVERSION_GEN_PARAM_METADATA_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <memory>

namespace mlir {

#define GEN_PASS_DECL_GENPARAMMETADATA
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createGenParamMetadataPass();

} // namespace mlir
#endif // MLIR_CONVERSION_UBTOSPIRV_UBSPIRV_H
