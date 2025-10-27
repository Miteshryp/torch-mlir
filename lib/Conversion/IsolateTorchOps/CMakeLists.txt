add_mlir_conversion_library(
  TorchMLIRIsolateTorchOps
  IsolateTorchOps.cpp
  IsolatorUtils.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/IsolateTorchOps
  DEPENDS
  TorchMLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  TorchMLIRTorchDialect)

torch_mlir_target_includes(TorchMLIRIsolateTorchOps)
