#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {

// Helper to resolve a !torch.int Value to a constant integer.
// It checks if the value is defined by a torch.constant.int.
std::optional<int64_t> matchConstantInt(Value value);

// Helper to resolve a !torch.list<int> Value to a vector of constant
// integers. It checks for a torch.prim.ListConstruct op whose inputs are all
// constant ints.
std::optional<SmallVector<int64_t>> matchConstantIntList(Value value);

// Helper to resolve a !torch.bool Value to a constant boolean.
std::optional<bool> matchConstantBool(Value value);

// Helper to segregate static and dynamic dependencies for a function
// NOTE: This should be called on all arguments of the operator that is being
// outlined
bool collectStaticDependencies(Value val,
                               llvm::DenseSet<Operation *> &staticOps,
                               llvm::SmallSet<StringRef, 4> &requiredResources);
} // namespace mlir
