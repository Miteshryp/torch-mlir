#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Conversion/IsolateTorchOps/Isolator.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {

class ConvolutionIsolator : public Isolator {
public:
  virtual void applyIsolation(ModuleOp &module, MLIRContext *ctx,
                              Operation *gop, int idx) override;
};
} // namespace mlir
