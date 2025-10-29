//===- RecordVtensorMetadataPass.cpp - Record vtensor metadata ------------===//
//
// This file implements a simple MLIR pass that records tensor/vtensor
// argument metadata (rank, element type, static sizes if available) for
// each function in a module. The metadata is attached as a function
// attribute named "vtensor.metadata" (an ArrayAttr of DictionaryAttr) and
// (optionally) emitted as a JSON file.
//
// Build/Usage notes:
//  - Add this .cpp to an MLIR-based tool or compile it as a plugin for
//    mlir-opt. The pass registers itself as
//    `record-vtensor-metadata`.
//  - Example invocation:
//      mlir-opt --record-vtensor-metadata[=out.json] input.mlir -o out.mlir
//    If an output filename is supplied (e.g.
//    --record-vtensor-metadata=meta.json) the pass will also write a JSON file
//    with the same metadata.
//
// Compatibility: targets MLIR C++ APIs (LLVM-style), should compile with
// recent MLIR/LLVM releases (may require small fixes across versions).
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {

std::string typeToString(Type ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  os.flush();
  return s;
}

struct GenParamMetadataPassOptions
    : public mlir::PassPipelineOptions<GenParamMetadataPassOptions> {};

struct GenParamMetadata
    : public PassWrapper<GenParamMetadata, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenParamMetadata)
  /// Optionally write out a JSON file with metadata. If empty, no file is
  /// written. The pass can be invoked with `--record-vtensor-metadata=foo.json`
  /// or just `--record-vtensor-metadata` to skip JSON emission (only attach
  /// attributes).

  Option<std::string> outputJson{
      *this, "output-json",
      llvm::cl::desc("Path to write tensor metadata JSON file"),
      llvm::cl::init("")};

  GenParamMetadata() = default;
  GenParamMetadata(const GenParamMetadata &pass) {
    // this->options.copyOptionValuesFrom(pass.options);
  }
  GenParamMetadata(GenParamMetadata &&pass) {
    // this->options.copyOptionValuesFrom(pass.options);
  }

  // GenParamMetadataPassOptions options;

  StringRef getArgument() const final { return "generate-param-metadata"; }

  StringRef getDescription() const final {
    return "Isolate each torch.named_op into a separate MLIR file for "
           "inspection";
  }

  // Will not use this right now since this is not yet tested properly.
  // TODO:
  // I don't have the time right now, but this could potentially lead to
  // a more general result production in the future.
  llvm::json::Object collectTensorOrMemRefInfo(Type type) {
    BaseTensorType torchTensorType = mlir::dyn_cast<BaseTensorType>(type);
    MemRefType memrefTy = mlir::dyn_cast<MemRefType>(type);

    llvm::json::Object jsonOut;

    if (torchTensorType) {
      SmallVector<int64_t> staticShape;
      jsonOut["rank"] = static_cast<int64_t>(
          torchTensorType.getSizes().size()); // Get the rank of tensor
      jsonOut["dtype"] = typeToString(torchTensorType.getDtype());
      staticShape.assign(torchTensorType.getSizes().begin(),
                         torchTensorType.getSizes().end());

      llvm::json::Array shapeArr;
      for (int64_t dim : staticShape)
        shapeArr.push_back(dim);
      jsonOut["shape"] = std::move(shapeArr);

    } else if (memrefTy) {
      jsonOut["rank"] = static_cast<int64_t>(memrefTy.getRank());
      jsonOut["dtype"] = stringifyType(memrefTy.getElementType());
      if (memrefTy.hasStaticShape()) {
        llvm::json::Array shapeArr;
        for (int64_t dim : memrefTy.getShape())
          shapeArr.push_back(dim);
        jsonOut["shape"] = std::move(shapeArr);
      }
      if (memrefTy.getMemorySpace())
        jsonOut["memory_space"] = stringifyAttr(memrefTy.getMemorySpace());
    } else {
      jsonOut["dtype"] = "non-tensor";
      jsonOut["rank"] = -1;
    }

    return jsonOut;
  }

  std::string stringifyType(Type ty) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ty.print(ss);
    return ss.str();
  }

  std::string stringifyAttr(Attribute attr) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    attr.print(ss);
    return ss.str();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    Builder builder(ctx);

    // {
    //     *this, "output-json", llvm::cl::desc("Optional JSON output file"),
    //     llvm::cl::value_desc("filename"), llvm::cl::init("")};

    llvm::json::Object rootJson;
    llvm::json::Array functionsJsonArray;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {

      SmallVector<Attribute> entries;
      llvm::json::Array funcJsonArray;

      for (BlockArgument arg : func.getArguments()) {
        Type ty = arg.getType();

        // Default entry values
        // int64_t rank = -1;
        SmallVector<int64_t> staticShape;
        std::string elemTypeStr = "unknown";

        if (!mlir::dyn_cast<torch::Torch::BaseTensorType>(ty)) {
          llvm::outs() << "This is not a Tensor type: Skipping\n";
          continue;
        }

        // auto baseTensor = mlir::dyn_cast<BaseTensorType>(ty);
        // if (baseTensor) {
        //   llvm::outs() << "Detected Ranked Tensor\n";
        //   rank = baseTensor.getSizes().size();
        //   elemTypeStr = typeToString(baseTensor.getDtype());
        //   staticShape.assign(baseTensor.getSizes().begin(),
        //                      baseTensor.getSizes().end());
        // } else {
        //   rank = -1;
        // }
        //
        //
        //
        //
        //
        //
        //
        // // Build MLIR attributes for the function attribute
        // NamedAttribute argIndexAttr = builder.getNamedAttr(
        //     "arg_index", builder.getI64IntegerAttr(arg.getArgNumber()));
        // NamedAttribute rankAttr =
        //     builder.getNamedAttr("rank", builder.getI64IntegerAttr(rank));
        //
        // // shape attribute: array of ints (use -1 for dynamic dims)
        // SmallVector<Attribute> shapeAttrs;
        // if (rank > 0) {
        //   for (int64_t d : staticShape)
        //     shapeAttrs.push_back(builder.getI64IntegerAttr(d));
        // }
        // NamedAttribute shapeNamed =
        //     builder.getNamedAttr("shape", builder.getArrayAttr(shapeAttrs));
        //
        // NamedAttribute dtypeAttr =
        //     builder.getNamedAttr("dtype",
        //     builder.getStringAttr(elemTypeStr));
        //
        // // Pack into a dictionary attribute
        // SmallVector<NamedAttribute> dictAttrs = {argIndexAttr, rankAttr,
        //                                          shapeNamed, dtypeAttr};
        // DictionaryAttr dict = builder.getDictionaryAttr(dictAttrs);
        // entries.push_back(dict);
        //
        //
        //
        //
        //
        //

        // Also create JSON entry for optional file output
        // llvm::json::Object jentry;
        // jentry["arg_index"] = (int64_t)arg.getArgNumber();
        // jentry["rank"] = rank;
        // if (rank > 0) {
        //   llvm::json::Array jshape;
        //   for (int64_t d : staticShape)
        //     jshape.push_back(d);
        //   jentry["shape"] = std::move(jshape);
        // } else {
        //   jentry["shape"] = llvm::json::Array();
        // }
        // jentry["dtype"] = elemTypeStr;
        llvm::json::Object jentry = collectTensorOrMemRefInfo(ty);
        funcJsonArray.push_back(std::move(jentry));
      }

      // Attach attribute to function
      // ArrayAttr arr = builder.getArrayAttr(entries);
      // func->setAttr("vtensor.metadata", arr);

      // Add to module-level JSON under function name
      std::string fname = func.getName().str();

      // llvm::outs() << "Fine till here?\n";

      // rootJson[fname] = std::move(funcJsonArray);
      llvm::json::Object funcJson;
      funcJson["args"] = std::move(funcJsonArray);

      // Add the return argument information
      llvm::json::Array returnArgumentArray;
      for (Type res : func.getFunctionType().getResults()) {
        returnArgumentArray.push_back(collectTensorOrMemRefInfo(res));
      }
      funcJson["returns"] = std::move(returnArgumentArray);

      // llvm::outs() << "IS this even working??\n";

      rootJson[fname] = std::move(funcJson);
    }

    // Optionally write JSON file
    if (!this->outputJson.getValue().empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(this->outputJson.getValue(), ec);
      if (ec) {
        // Emit a warning but do not fail the pass
        llvm::outs() << "Could not open JSON output file: " << ec.message();
      } else {
        llvm::json::OStream jos(os, 2);
        jos.value(llvm::json::Value(std::move(rootJson)));
      }
      os.flush();
    }
  }
}; // end anonymous namespace

std::unique_ptr<Pass> createGenParamMetadataPass() {
  return std::make_unique<GenParamMetadata>();
}

} // namespace mlir
