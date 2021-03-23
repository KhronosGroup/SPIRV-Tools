
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
    public:
    explicit TransformationSwapFunctionVariables(
      const protobufs::TransformationSwapFunctionVariables& message);

    TransformationSwapFunctionVariables(uint32_t var_id1,
                                    uint32_t var_id2);

    bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

    void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

    protobufs::Transformation ToMessage() const override;
    
    std::unordered_set<uint32_t> GetFreshIds() const override;
    
    private:
    protobufs::TransformationSwapFunctionVariables message_;

};


}  // namespace fuzz
}  // namespace spvtools

#endif // SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTION_VARIABLES_H_