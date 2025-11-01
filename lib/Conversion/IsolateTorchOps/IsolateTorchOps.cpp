//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

#include "torch-mlir/Conversion/IsolateTorchOps/GenericIsolator.h"
#include "torch-mlir/Conversion/IsolateTorchOps/IsolateTorchOps.h"
#include "torch-mlir/Conversion/IsolateTorchOps/IsolatorUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {
struct IsolateTorchOps
    : public PassWrapper<IsolateTorchOps, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsolateTorchOps)

  Option<std::string> outputFolderPath{
      *this, "output-path",
      llvm::cl::desc("Path of output director to write isolated kernels into"),
      llvm::cl::init("./lowerings")};

  IsolateTorchOps() = default;
  IsolateTorchOps(const IsolateTorchOps &pass) {}
  IsolateTorchOps(IsolateTorchOps &&pass) {}

  StringRef getArgument() const final { return "isolate-torch-ops"; }

  StringRef getDescription() const final {
    return "Isolate each torch.named_op into a separate MLIR file for "
           "inspection";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    if (outputFolderPath.getValue().empty()) {
      llvm::errs() << "Output Folder not passed: Exiting Isolation Pass\n";
      exit(2);
    }

    fs::path output_folder(outputFolderPath.getValue());

    // llvm::outs() << "This is running?\n";
    // index for naming files
    unsigned idx = 0;

    SmallVector<Operation *, 8> ops;

    module.walk([&](mlir::Operation *op) {
      llvm::TypeSwitch<Operation *, void>(op)
          .Case<mlir::linalg::GenericOp>([&](mlir::linalg::GenericOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,
                                            "linalg/outlined_linalg_");
          })
          .Case<torch::Torch::AtenReluOp>([&](torch::Torch::AtenReluOp) {
            GenericIsolator::applyIsolation(
                module, ctx, op, idx++, output_folder, "relu/outlined_relu_");
          })
          .Case<torch::Torch::AtenConv2dOp>([&](Torch::AtenConv2dOp constOp) {
            GenericIsolator::applyIsolation(
                module, ctx, op, idx++, output_folder, "conv/outlined_conv2d_");
          })
          .Case<torch::Torch::AtenConv3dOp>([&](Torch::AtenConv3dOp constOp) {
            GenericIsolator::applyIsolation(
                module, ctx, op, idx++, output_folder, "conv/outlined_conv3d_");
          })
          .Case<torch::Torch::AtenConvolutionOp>(
              [&](Torch::AtenConvolutionOp constOp) {
                GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                                output_folder,

                                                "conv/outlined_conv_");
              })
          .Case<torch::Torch::AtenMatmulOp>([&](auto constOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,

                                            "matmul/outlined_matmul_");
            llvm::outs() << "Matmul Operation: Pushed\n";
          })
          .Case<torch::Torch::AtenMmOp>([&](auto mmOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,

                                            "matmul/outlined_mm_");
            llvm::outs() << "MM Operation: Pushed\n";
          })
          .Case<torch::Torch::AtenTransposeIntOp>(
              [&](torch::Torch::AtenTransposeIntOp) {
                GenericIsolator::applyIsolation(
                    module, ctx, op, idx++, output_folder,
                    "transpose/outlined_transpose_");
              })
          .Case<torch::Torch::AtenSqueezeOp>([&](auto constOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,

                                            "squeeze/outlined_squeeze_");
            llvm::outs() << "Squeeze Operation: Pushed\n";
          })
          .Case<torch::Torch::AtenSqueezeDimOp>([&](auto constOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,

                                            "squeeze/outlined_dim_squeeze_");
            llvm::outs() << "Dim Squeeze Operation: Pushed\n";
          })
          .Case<torch::Torch::AtenLinearOp>([&](auto linearOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,

                                            "linear/outlined_linear_");
            llvm::outs() << "Linear Operation: Pushed\n";
          })
          .Case<torch::Torch::AtenAvgPool2dOp>([&](auto constOp) {
            GenericIsolator::applyIsolation(module, ctx, op, idx++,
                                            output_folder,

                                            "pooling/outlined_avg_2d_pool_");
            llvm::outs() << "Average 2D Pool Operation: Pushed\n";
          })
          .Default([](mlir::Operation *op) {});
    });
  }
};

std::unique_ptr<Pass> createIsolateTorchOps() {
  return std::make_unique<IsolateTorchOps>();
}

} // namespace mlir
