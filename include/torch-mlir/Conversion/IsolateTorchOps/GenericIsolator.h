#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/IsolateTorchOps/IsolatorUtils.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/SmallSet.h"

#include "../PassDetail.h"
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

#include "torch-mlir/Conversion/IsolateTorchOps/IsolateTorchOps.h"
#include "torch-mlir/Conversion/IsolateTorchOps/IsolatorUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/RegionUtils.h"
#include <filesystem>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace fs = std::filesystem;

namespace mlir {

class GenericIsolator {
public:
  static void applyIsolation(ModuleOp &module, MLIRContext *ctx, Operation *gop,
                             int idx, const fs::path &outputFolderPath,
                             const std::string &file_prefix) {
    // SymbolTable symbolTable(module);
    OpBuilder builder(ctx);
    ModuleOp fileModule = ModuleOp::create(builder.getUnknownLoc());

    // Extracting Operators in the module
    llvm::SetVector<Value> externalOperands;

    // Collect SSA Arguments to the operation
    // Extracting the type of all arguments needed in the operator
    // (Including Dyamic operands + SSA arguments needed in the region
    // of operation)
    externalOperands.insert(gop->getOperands().begin(),
                            gop->getOperands().end());

    // Finding and storing SSA values used in the region of operation
    if (gop->getNumRegions() > 0) {
      Region &region = gop->getRegion(0);
      getUsedValuesDefinedAbove(region, externalOperands);
    }

    // Segregating collected operand into categories for seperate handling logic
    // argOperands -> Comes from the user
    // staticDefinitions -> SSA values statically declared in the input program
    // requiredResourceDefinitions -> Defined in the dialect resources. Need to
    // be added in outlined kernel
    SmallVector<Value> argOperands;
    llvm::DenseSet<Operation *> staticDefinitions;
    llvm::SmallSet<StringRef, 4> requiredResourceDefinitions;

    // Parsing all selected external operands to see which one of them are
    // static, resource dependent or dynamic
    // This will reduce the number of arguments needed from wrapper code to a
    // bare minimum
    for (Value val : externalOperands) {
      // argTypes.push_back(val.getType());

      llvm::DenseSet<Operation *> currentStaticDefinitions;
      llvm::SmallSet<StringRef, 4> currentRequiredResourceDefinitions;

      if (collectStaticDependencies(val, currentStaticDefinitions,
                                    currentRequiredResourceDefinitions)) {
        // The operand can be induced from source file itself.
        // Need not pass this burden to the wrapper
        staticDefinitions.insert(currentStaticDefinitions.begin(),
                                 currentStaticDefinitions.end());
        requiredResourceDefinitions.insert(
            currentRequiredResourceDefinitions.begin(),
            currentRequiredResourceDefinitions.end());
      } else {
        // Operand is Dynamic, need it from wrapper code
        argOperands.push_back(val);
      }
    }

    // Extracting type of user input operands for function signature creation
    SmallVector<Type> argTypes;
    for (Value &op : argOperands) {
      argTypes.push_back(op.getType());
    }

    // Extracting return type information from operation
    ValueTypeRange<ResultRange> resTypes = gop->getResultTypes();

    // Create function signature with valid input and return types
    mlir::FunctionType funcType = builder.getFunctionType(argTypes, resTypes);
    func::FuncOp outlinedFunc = func::FuncOp::create(
        builder.getUnknownLoc(),
        // The method name is the same in all outlined kernels for easier
        // integration with the benchmarking tool
        "kernel_call", funcType);

    // Create code block in the outlined kernel
    mlir::Block &entryBlock = *outlinedFunc.addEntryBlock();

    // Mapping extracted signatures to SSAs in the module
    IRMapping mapping;
    for (size_t i = 0; i < argOperands.size(); i++) {
      mapping.map(argOperands[i], entryBlock.getArgument(i));
    }

    // Create builder to create outlined module
    OpBuilder funcBuilder(&entryBlock, entryBlock.begin());

    // Clone static definitions into the outlined function in a topological
    // order using the same IRMapping we used for function arguments. This
    // ensures that when a static op depends on another static op, the
    // dependency is remapped to the cloned version rather than left
    // referencing the original op's SSA values. Cloning out-of-order can
    // produce ops that reference original values which may later be erased
    // producing "operation destroyed but still has uses" errors.
    SmallVector<Operation *> pendingStaticOps(staticDefinitions.begin(),
                                              staticDefinitions.end());
    // Debug: print the static ops we're planning to clone.
    // llvm::outs() << "Static definitions to clone: \n";
    for (Operation *op : pendingStaticOps) {
      llvm::outs() << "  Op: " << op->getName() << " (" << op << ")\n";
      // for (Value res : op->getResults()) {
      // llvm::outs() << "    Result: " << res << " uses=" <<
      // res.getUses().size() << "\n";
      // }
    }
    while (!pendingStaticOps.empty()) {
      bool progress = false;
      for (auto it = pendingStaticOps.begin(); it != pendingStaticOps.end();) {
        Operation *op = *it;

        // Check if all operands that are produced by other static ops have
        // already been cloned (i.e. are present in the mapping). If so, we
        // can safely clone this op and let the mapping remap its operands to
        // the cloned results.
        bool ready = true;
        for (Value operand : op->getOperands()) {
          // Require that every operand that the op references is already
          // present in the mapping. If an operand isn't mapped yet we must
          // wait until it is (it might be an argument, a static dependency
          // that will be cloned earlier, or another mapped value).
          if (!mapping.lookupOrNull(operand)) {
            ready = false;
            break;
          }
        }

        if (ready) {
          funcBuilder.clone(*op, mapping);
          progress = true;
          it = pendingStaticOps.erase(it);
        } else {
          ++it;
        }
      }

      // If we couldn't make progress (cycle or unexpected dependency), just
      // clone the remaining ops with the mapping as best-effort to avoid an
      // infinite loop. This may leave some operands referencing original
      // values, but it avoids getting stuck; the caller should ensure
      // staticDefinitions are acyclic where possible.
      if (!progress) {
        for (Operation *op : pendingStaticOps)
          funcBuilder.clone(*op, mapping);
        break;
      }
    }

    // Ensure all operands (and their static dependencies) are present in the
    // mapping before cloning the main op. If any required value isn't yet
    // mapped, recursively clone its defining op (if it's static) so the
    // mapping gets populated. This prevents IRMapping::lookup assertions
    // inside the clone implementation.
    std::function<void(Value)> ensureMapped = [&](Value v) {
      if (mapping.lookupOrNull(v))
        return;
      // If this value is a block argument it should already be mapped to an
      // entry argument; otherwise, try to clone its defining op if it's a
      // static dependency.
      if (auto defOp = v.getDefiningOp()) {
        if (staticDefinitions.count(defOp)) {
          // Ensure all operands of the defining op are mapped first.
          for (Value opnd : defOp->getOperands())
            ensureMapped(opnd);
          // Clone the defining op now that its operands are mapped and
          // update the mapping.
          funcBuilder.clone(*defOp, mapping);
          return;
        }
      }
      // If we reach here the value is neither mapped nor clonable as a
      // static dependency. Emit a diagnostic to help debugging.
      llvm::errs() << "Warning: value not mapped and not a static dep: ";
      v.print(llvm::errs());
      llvm::errs() << "\n";
    };

    // Ensure every operand that the outlined op references is mapped.
    for (Value opnd : gop->getOperands())
      ensureMapped(opnd);

    // Clone op into the new outlined function's block, based on the
    // arguments we defined
    Operation *cloned = funcBuilder.clone(*gop, mapping);

    // Create function with valid argument and return type signatures, as well
    // as input SSAs mapped as arguments to the function containing the op
    funcBuilder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                       cloned->getResults());

    fileModule.push_back(outlinedFunc);

    // Print the module to a file
    // fs::path output_filepath =
    //     fs::path(fs::current_path().append("lowerings"))
    //         .append((file_prefix + Twine(idx) + ".mlir").str());

    // std::string filename =
    //     "lowerings/" + (file_prefix + Twine(idx) + ".mlir").str();

    std::string filename =
        fs::path(outputFolderPath)
            .append((file_prefix + Twine(idx) + ".mlir").str())
            .generic_string();

    llvm::errs() << "Output Filename:  " << filename << '\n';

    fs::path filepath = fs::current_path().append(filename);
    fs::path dir_path = filepath.parent_path();

    if (!fs::exists(dir_path)) {
      fs::create_directories(dir_path);
    }

    std::error_code ec;
    llvm::raw_fd_ostream out(filename, ec);
    if (ec) {
      llvm::errs() << "Failed to open " << filename
                   << " for writing: " << ec.message() << "\n";
    } else {
      fileModule.print(out);
      llvm::outs() << "Wrote " << filename << "\n";
    }
  }
};
} // namespace mlir
