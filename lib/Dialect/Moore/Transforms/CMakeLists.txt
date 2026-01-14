add_circt_dialect_library(CIRCTMooreTransforms
InlineCalls.cpp
LowerConcatRef.cpp
SimplifyProcedures.cpp
CreateVTables.cpp

  DEPENDS
  CIRCTMooreTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
