
#ifndef SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_
#define SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_


namespace spvtools {
namespace fuzz {

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"


class TransformationSwapFunctionVariables : public Transformation
{

};


}  // namespace fuzz
}  // namespace spvtools

#endif // SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_