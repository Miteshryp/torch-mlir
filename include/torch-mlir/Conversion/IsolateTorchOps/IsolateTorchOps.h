//===- IsolateLinalgGenerics.h -  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ISOLATE_TORCH_OPS_H
#define MLIR_CONVERSION_ISOLATE_TORCH_OPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <memory>

namespace mlir {

#define GEN_PASS_DECL_ISOLATETORCHOPS
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createIsolateTorchOps();

} // namespace mlir
#endif // MLIR_CONVERSION_UBTOSPIRV_UBSPIRV_H
