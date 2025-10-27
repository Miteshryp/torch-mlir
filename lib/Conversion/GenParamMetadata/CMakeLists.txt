add_mlir_conversion_library(
  TorchMLIRGenParamMetadata
  GenParameterData.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/GenParamMetadata
  DEPENDS
  TorchMLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  TorchMLIRTorchDialect)

torch_mlir_target_includes(TorchMLIRGenParamMetadata)
