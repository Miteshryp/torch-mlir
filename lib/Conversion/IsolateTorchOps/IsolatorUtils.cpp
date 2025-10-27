#include "torch-mlir/Conversion/IsolateTorchOps/IsolatorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h" // Dialect resource handle APIs
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir::torch;

namespace mlir {

std::optional<int64_t> matchConstantInt(Value value) {
  if (auto cst = value.getDefiningOp<Torch::ConstantIntOp>()) {
    return cst.getValue();
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t>> matchConstantIntList(Value value) {
  auto listConstruct = value.getDefiningOp<Torch::PrimListConstructOp>();
  if (!listConstruct)
    return std::nullopt;

  SmallVector<int64_t> intValues;
  for (Value operand : listConstruct.getOperands()) {
    std::optional<int64_t> intVal = matchConstantInt(operand);
    if (!intVal)
      return std::nullopt; // One of the elements is not a constant
    intValues.push_back(*intVal);
  }
  return intValues;
}

std::optional<bool> matchConstantBool(Value value) {
  if (auto cst = value.getDefiningOp<Torch::ConstantBoolOp>()) {
    return cst.getValue();
  }
  return std::nullopt;
}

bool collectStaticDependencies(
    Value val, llvm::DenseSet<Operation *> &staticOps,
    llvm::SmallSet<StringRef, 4> &requiredResources) {

  // A block argument is a runtime value, so it's not static.
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp)
    return false;

  // If we've already processed this op, we're done with this branch.
  if (staticOps.count(definingOp))
    return true;

  // --- Define what operations are considered "static-able" ---
  bool isStaticable =
      isa<Torch::ConstantBoolOp, Torch::ConstantFloatOp, Torch::ConstantIntOp,
          Torch::ConstantNoneOp, Torch::ConstantStrOp,
          Torch::PrimListConstructOp, Torch::PrimTupleConstructOp,
          Torch::ValueTensorLiteralOp>(definingOp);

  if (!isStaticable) {
    // This operation is not something we can move, so the value is dynamic.
    return false;
  }

  // --- Recursively check all operands of this operation ---
  for (Value operand : definingOp->getOperands()) {
    if (!collectStaticDependencies(operand, staticOps, requiredResources)) {
      // If any dependency is dynamic, this whole chain is dynamic.
      return false;
    }
  }

  // --- Success! This op and its dependencies are static. ---
  // Add it to our set of ops to be cloned.
  staticOps.insert(definingOp);

  // If it's a literal, also record its resource dependency.
  if (auto literalOp = dyn_cast<Torch::ValueTensorLiteralOp>(definingOp)) {
    if (auto denseResourceAttr =
            mlir::dyn_cast<mlir::DenseResourceElementsAttr>(
                literalOp.getValueAttr())) {
      requiredResources.insert(denseResourceAttr.getRawHandle().getKey());
    }
  }

  return true;
}

} // namespace mlir
