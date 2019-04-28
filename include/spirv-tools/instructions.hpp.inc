// THIS FILE IS GENERATED

struct INop : public Instruction {
  static constexpr SpvOp Opcode = SpvOpNop;
  INop(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IUndef : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUndef;
  IUndef(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISourceContinued : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSourceContinued;
  ISourceContinued(const spv_parsed_instruction_t *i) : Instruction(i) {}
  const char * GetContinuedSource() const { return getstr(0); }
};
struct ISource : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSource;
  ISource(const spv_parsed_instruction_t *i) : Instruction(i) {}
  SpvSourceLanguage GetSourceLanguage() const { return getval<SpvSourceLanguage>(0); }
uint32_t GetVersion() const { return getval<uint32_t>(1); }
uint32_t GetFile() const { return getval<uint32_t>(2); }
const char * GetSource() const { return getstr(3); }
};
struct ISourceExtension : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSourceExtension;
  ISourceExtension(const spv_parsed_instruction_t *i) : Instruction(i) {}
  const char * GetExtension() const { return getstr(0); }
};
struct IName : public Instruction {
  static constexpr SpvOp Opcode = SpvOpName;
  IName(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTarget() const { return getval<uint32_t>(0); }
const char * GetName() const { return getstr(1); }
};
struct IMemberName : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMemberName;
  IMemberName(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetType() const { return getval<uint32_t>(0); }
uint32_t GetMember() const { return getval<uint32_t>(1); }
const char * GetName() const { return getstr(2); }
};
struct IString : public Instruction {
  static constexpr SpvOp Opcode = SpvOpString;
  IString(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
const char * GetString() const { return getstr(1); }
};
struct ILine : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLine;
  ILine(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetFile() const { return getval<uint32_t>(0); }
uint32_t GetLine() const { return getval<uint32_t>(1); }
uint32_t GetColumn() const { return getval<uint32_t>(2); }
};
struct IExtension : public Instruction {
  static constexpr SpvOp Opcode = SpvOpExtension;
  IExtension(const spv_parsed_instruction_t *i) : Instruction(i) {}
  const char * GetName() const { return getstr(0); }
};
struct IExtInstImport : public Instruction {
  static constexpr SpvOp Opcode = SpvOpExtInstImport;
  IExtInstImport(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
const char * GetName() const { return getstr(1); }
};
struct IExtInst : public Instruction {
  static constexpr SpvOp Opcode = SpvOpExtInst;
  IExtInst(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSet() const { return getval<uint32_t>(2); }
uint32_t GetInstruction() const { return getval<uint32_t>(3); }
const std::vector<uint32_t> GetOperands() const { return getvec<uint32_t>(4); }
};
struct IMemoryModel : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMemoryModel;
  IMemoryModel(const spv_parsed_instruction_t *i) : Instruction(i) {}
  SpvAddressingModel GetAddressingModel() const { return getval<SpvAddressingModel>(0); }
SpvMemoryModel GetMemoryModel() const { return getval<SpvMemoryModel>(1); }
};
struct IEntryPoint : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEntryPoint;
  IEntryPoint(const spv_parsed_instruction_t *i) : Instruction(i) {}
  SpvExecutionModel GetExecutionModel() const { return getval<SpvExecutionModel>(0); }
uint32_t GetEntryPoint() const { return getval<uint32_t>(1); }
const char * GetName() const { return getstr(2); }
const std::vector<uint32_t> GetInterface() const { return getvec<uint32_t>(3); }
};
struct IExecutionMode : public Instruction {
  static constexpr SpvOp Opcode = SpvOpExecutionMode;
  IExecutionMode(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetEntryPoint() const { return getval<uint32_t>(0); }
SpvExecutionMode GetMode() const { return getval<SpvExecutionMode>(1); }
};
struct ICapability : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCapability;
  ICapability(const spv_parsed_instruction_t *i) : Instruction(i) {}
  SpvCapability GetCapability() const { return getval<SpvCapability>(0); }
};
struct ITypeVoid : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeVoid;
  ITypeVoid(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeBool : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeBool;
  ITypeBool(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeInt : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeInt;
  ITypeInt(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetWidth() const { return getval<uint32_t>(1); }
uint32_t GetSignedness() const { return getval<uint32_t>(2); }
};
struct ITypeFloat : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeFloat;
  ITypeFloat(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetWidth() const { return getval<uint32_t>(1); }
};
struct ITypeVector : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeVector;
  ITypeVector(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetComponentType() const { return getval<uint32_t>(1); }
uint32_t GetComponentCount() const { return getval<uint32_t>(2); }
};
struct ITypeMatrix : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeMatrix;
  ITypeMatrix(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetColumnType() const { return getval<uint32_t>(1); }
uint32_t GetColumnCount() const { return getval<uint32_t>(2); }
};
struct ITypeImage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeImage;
  ITypeImage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetSampledType() const { return getval<uint32_t>(1); }
SpvDim GetDim() const { return getval<SpvDim>(2); }
uint32_t GetDepth() const { return getval<uint32_t>(3); }
uint32_t GetArrayed() const { return getval<uint32_t>(4); }
uint32_t GetMS() const { return getval<uint32_t>(5); }
uint32_t GetSampled() const { return getval<uint32_t>(6); }
SpvImageFormat GetImageFormat() const { return getval<SpvImageFormat>(7); }
SpvAccessQualifier GetAccessQualifier() const { return getval<SpvAccessQualifier>(8); }
};
struct ITypeSampler : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeSampler;
  ITypeSampler(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeSampledImage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeSampledImage;
  ITypeSampledImage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetImageType() const { return getval<uint32_t>(1); }
};
struct ITypeArray : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeArray;
  ITypeArray(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetElementType() const { return getval<uint32_t>(1); }
uint32_t GetLength() const { return getval<uint32_t>(2); }
};
struct ITypeRuntimeArray : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeRuntimeArray;
  ITypeRuntimeArray(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetElementType() const { return getval<uint32_t>(1); }
};
struct ITypeStruct : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeStruct;
  ITypeStruct(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
const std::vector<uint32_t> GetMembers() const { return getvec<uint32_t>(1); }
};
struct ITypeOpaque : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeOpaque;
  ITypeOpaque(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
const char * GetTypeName() const { return getstr(1); }
};
struct ITypePointer : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypePointer;
  ITypePointer(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
SpvStorageClass GetStorageClass() const { return getval<SpvStorageClass>(1); }
uint32_t GetType() const { return getval<uint32_t>(2); }
};
struct ITypeFunction : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeFunction;
  ITypeFunction(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetReturnType() const { return getval<uint32_t>(1); }
const std::vector<uint32_t> GetParameterTypes() const { return getvec<uint32_t>(2); }
};
struct ITypeEvent : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeEvent;
  ITypeEvent(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeDeviceEvent : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeDeviceEvent;
  ITypeDeviceEvent(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeReserveId : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeReserveId;
  ITypeReserveId(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeQueue : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeQueue;
  ITypeQueue(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypePipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypePipe;
  ITypePipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
SpvAccessQualifier GetQualifier() const { return getval<SpvAccessQualifier>(1); }
};
struct ITypeForwardPointer : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeForwardPointer;
  ITypeForwardPointer(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointerType() const { return getval<uint32_t>(0); }
SpvStorageClass GetStorageClass() const { return getval<SpvStorageClass>(1); }
};
struct IConstantTrue : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstantTrue;
  IConstantTrue(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct IConstantFalse : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstantFalse;
  IConstantFalse(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct IConstant : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstant;
  IConstant(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
};
struct IConstantComposite : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstantComposite;
  IConstantComposite(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
const std::vector<uint32_t> GetConstituents() const { return getvec<uint32_t>(2); }
};
struct IConstantSampler : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstantSampler;
  IConstantSampler(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
SpvSamplerAddressingMode GetSamplerAddressingMode() const { return getval<SpvSamplerAddressingMode>(2); }
uint32_t GetParam() const { return getval<uint32_t>(3); }
SpvSamplerFilterMode GetSamplerFilterMode() const { return getval<SpvSamplerFilterMode>(4); }
};
struct IConstantNull : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstantNull;
  IConstantNull(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISpecConstantTrue : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSpecConstantTrue;
  ISpecConstantTrue(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISpecConstantFalse : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSpecConstantFalse;
  ISpecConstantFalse(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISpecConstant : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSpecConstant;
  ISpecConstant(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
};
struct ISpecConstantComposite : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSpecConstantComposite;
  ISpecConstantComposite(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
const std::vector<uint32_t> GetConstituents() const { return getvec<uint32_t>(2); }
};
struct ISpecConstantOp : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSpecConstantOp;
  ISpecConstantOp(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOpcode() const { return getval<uint32_t>(2); }
};
struct IFunction : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFunction;
  IFunction(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetFunctionControl() const { return getval<uint32_t>(2); }
uint32_t GetFunctionType() const { return getval<uint32_t>(3); }
};
struct IFunctionParameter : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFunctionParameter;
  IFunctionParameter(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct IFunctionEnd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFunctionEnd;
  IFunctionEnd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IFunctionCall : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFunctionCall;
  IFunctionCall(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetFunction() const { return getval<uint32_t>(2); }
const std::vector<uint32_t> GetArguments() const { return getvec<uint32_t>(3); }
};
struct IVariable : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVariable;
  IVariable(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
SpvStorageClass GetStorageClass() const { return getval<SpvStorageClass>(2); }
uint32_t GetInitializer() const { return getval<uint32_t>(3); }
};
struct IImageTexelPointer : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageTexelPointer;
  IImageTexelPointer(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetSample() const { return getval<uint32_t>(4); }
};
struct ILoad : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLoad;
  ILoad(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemoryAccess() const { return getval<uint32_t>(3); }
};
struct IStore : public Instruction {
  static constexpr SpvOp Opcode = SpvOpStore;
  IStore(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointer() const { return getval<uint32_t>(0); }
uint32_t GetObject() const { return getval<uint32_t>(1); }
uint32_t GetMemoryAccess() const { return getval<uint32_t>(2); }
};
struct ICopyMemory : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCopyMemory;
  ICopyMemory(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTarget() const { return getval<uint32_t>(0); }
uint32_t GetSource() const { return getval<uint32_t>(1); }
uint32_t GetMemoryAccess() const { return getval<uint32_t>(2); }
uint32_t GetSourceMemoryAccess() const { return getval<uint32_t>(3); }
};
struct ICopyMemorySized : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCopyMemorySized;
  ICopyMemorySized(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTarget() const { return getval<uint32_t>(0); }
uint32_t GetSource() const { return getval<uint32_t>(1); }
uint32_t GetSize() const { return getval<uint32_t>(2); }
uint32_t GetMemoryAccess() const { return getval<uint32_t>(3); }
uint32_t GetSourceMemoryAccess() const { return getval<uint32_t>(4); }
};
struct IAccessChain : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAccessChain;
  IAccessChain(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
const std::vector<uint32_t> GetIndexes() const { return getvec<uint32_t>(3); }
};
struct IInBoundsAccessChain : public Instruction {
  static constexpr SpvOp Opcode = SpvOpInBoundsAccessChain;
  IInBoundsAccessChain(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
const std::vector<uint32_t> GetIndexes() const { return getvec<uint32_t>(3); }
};
struct IPtrAccessChain : public Instruction {
  static constexpr SpvOp Opcode = SpvOpPtrAccessChain;
  IPtrAccessChain(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetElement() const { return getval<uint32_t>(3); }
const std::vector<uint32_t> GetIndexes() const { return getvec<uint32_t>(4); }
};
struct IArrayLength : public Instruction {
  static constexpr SpvOp Opcode = SpvOpArrayLength;
  IArrayLength(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetStructure() const { return getval<uint32_t>(2); }
uint32_t GetArraymember() const { return getval<uint32_t>(3); }
};
struct IGenericPtrMemSemantics : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGenericPtrMemSemantics;
  IGenericPtrMemSemantics(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
};
struct IInBoundsPtrAccessChain : public Instruction {
  static constexpr SpvOp Opcode = SpvOpInBoundsPtrAccessChain;
  IInBoundsPtrAccessChain(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetElement() const { return getval<uint32_t>(3); }
const std::vector<uint32_t> GetIndexes() const { return getvec<uint32_t>(4); }
};
struct IDecorate : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDecorate;
  IDecorate(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTarget() const { return getval<uint32_t>(0); }
SpvDecoration GetDecoration() const { return getval<SpvDecoration>(1); }
};
struct IMemberDecorate : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMemberDecorate;
  IMemberDecorate(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetStructureType() const { return getval<uint32_t>(0); }
uint32_t GetMember() const { return getval<uint32_t>(1); }
SpvDecoration GetDecoration() const { return getval<SpvDecoration>(2); }
};
struct IDecorationGroup : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDecorationGroup;
  IDecorationGroup(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct IGroupDecorate : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupDecorate;
  IGroupDecorate(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetDecorationGroup() const { return getval<uint32_t>(0); }
const std::vector<uint32_t> GetTargets() const { return getvec<uint32_t>(1); }
};
struct IGroupMemberDecorate : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupMemberDecorate;
  IGroupMemberDecorate(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetDecorationGroup() const { return getval<uint32_t>(0); }
const std::vector<std::pair<uint32_t, uint32_t>> GetTargets() const { return getvec<std::pair<uint32_t, uint32_t>>(1); }
};
struct IVectorExtractDynamic : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVectorExtractDynamic;
  IVectorExtractDynamic(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector() const { return getval<uint32_t>(2); }
uint32_t GetIndex() const { return getval<uint32_t>(3); }
};
struct IVectorInsertDynamic : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVectorInsertDynamic;
  IVectorInsertDynamic(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector() const { return getval<uint32_t>(2); }
uint32_t GetComponent() const { return getval<uint32_t>(3); }
uint32_t GetIndex() const { return getval<uint32_t>(4); }
};
struct IVectorShuffle : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVectorShuffle;
  IVectorShuffle(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector1() const { return getval<uint32_t>(2); }
uint32_t GetVector2() const { return getval<uint32_t>(3); }
const std::vector<uint32_t> GetComponents() const { return getvec<uint32_t>(4); }
};
struct ICompositeConstruct : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCompositeConstruct;
  ICompositeConstruct(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
const std::vector<uint32_t> GetConstituents() const { return getvec<uint32_t>(2); }
};
struct ICompositeExtract : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCompositeExtract;
  ICompositeExtract(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetComposite() const { return getval<uint32_t>(2); }
const std::vector<uint32_t> GetIndexes() const { return getvec<uint32_t>(3); }
};
struct ICompositeInsert : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCompositeInsert;
  ICompositeInsert(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetObject() const { return getval<uint32_t>(2); }
uint32_t GetComposite() const { return getval<uint32_t>(3); }
const std::vector<uint32_t> GetIndexes() const { return getvec<uint32_t>(4); }
};
struct ICopyObject : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCopyObject;
  ICopyObject(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct ITranspose : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTranspose;
  ITranspose(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetMatrix() const { return getval<uint32_t>(2); }
};
struct ISampledImage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSampledImage;
  ISampledImage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetSampler() const { return getval<uint32_t>(3); }
};
struct IImageSampleImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleImplicitLod;
  IImageSampleImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSampleExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleExplicitLod;
  IImageSampleExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSampleDrefImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleDrefImplicitLod;
  IImageSampleDrefImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSampleDrefExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleDrefExplicitLod;
  IImageSampleDrefExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSampleProjImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleProjImplicitLod;
  IImageSampleProjImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSampleProjExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleProjExplicitLod;
  IImageSampleProjExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSampleProjDrefImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleProjDrefImplicitLod;
  IImageSampleProjDrefImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSampleProjDrefExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleProjDrefExplicitLod;
  IImageSampleProjDrefExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageFetch : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageFetch;
  IImageFetch(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageGather : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageGather;
  IImageGather(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetComponent() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageDrefGather : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageDrefGather;
  IImageDrefGather(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageRead : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageRead;
  IImageRead(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageWrite : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageWrite;
  IImageWrite(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetImage() const { return getval<uint32_t>(0); }
uint32_t GetCoordinate() const { return getval<uint32_t>(1); }
uint32_t GetTexel() const { return getval<uint32_t>(2); }
uint32_t GetImageOperands() const { return getval<uint32_t>(3); }
};
struct IImage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImage;
  IImage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
};
struct IImageQueryFormat : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQueryFormat;
  IImageQueryFormat(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
};
struct IImageQueryOrder : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQueryOrder;
  IImageQueryOrder(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
};
struct IImageQuerySizeLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQuerySizeLod;
  IImageQuerySizeLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetLevelofDetail() const { return getval<uint32_t>(3); }
};
struct IImageQuerySize : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQuerySize;
  IImageQuerySize(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
};
struct IImageQueryLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQueryLod;
  IImageQueryLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
};
struct IImageQueryLevels : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQueryLevels;
  IImageQueryLevels(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
};
struct IImageQuerySamples : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageQuerySamples;
  IImageQuerySamples(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
};
struct IConvertFToU : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConvertFToU;
  IConvertFToU(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetFloatValue() const { return getval<uint32_t>(2); }
};
struct IConvertFToS : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConvertFToS;
  IConvertFToS(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetFloatValue() const { return getval<uint32_t>(2); }
};
struct IConvertSToF : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConvertSToF;
  IConvertSToF(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSignedValue() const { return getval<uint32_t>(2); }
};
struct IConvertUToF : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConvertUToF;
  IConvertUToF(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetUnsignedValue() const { return getval<uint32_t>(2); }
};
struct IUConvert : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUConvert;
  IUConvert(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetUnsignedValue() const { return getval<uint32_t>(2); }
};
struct ISConvert : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSConvert;
  ISConvert(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSignedValue() const { return getval<uint32_t>(2); }
};
struct IFConvert : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFConvert;
  IFConvert(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetFloatValue() const { return getval<uint32_t>(2); }
};
struct IQuantizeToF16 : public Instruction {
  static constexpr SpvOp Opcode = SpvOpQuantizeToF16;
  IQuantizeToF16(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
};
struct IConvertPtrToU : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConvertPtrToU;
  IConvertPtrToU(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
};
struct ISatConvertSToU : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSatConvertSToU;
  ISatConvertSToU(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSignedValue() const { return getval<uint32_t>(2); }
};
struct ISatConvertUToS : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSatConvertUToS;
  ISatConvertUToS(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetUnsignedValue() const { return getval<uint32_t>(2); }
};
struct IConvertUToPtr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConvertUToPtr;
  IConvertUToPtr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetIntegerValue() const { return getval<uint32_t>(2); }
};
struct IPtrCastToGeneric : public Instruction {
  static constexpr SpvOp Opcode = SpvOpPtrCastToGeneric;
  IPtrCastToGeneric(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
};
struct IGenericCastToPtr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGenericCastToPtr;
  IGenericCastToPtr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
};
struct IGenericCastToPtrExplicit : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGenericCastToPtrExplicit;
  IGenericCastToPtrExplicit(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
SpvStorageClass GetStorage() const { return getval<SpvStorageClass>(3); }
};
struct IBitcast : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitcast;
  IBitcast(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct ISNegate : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSNegate;
  ISNegate(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct IFNegate : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFNegate;
  IFNegate(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct IIAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIAdd;
  IIAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFAdd;
  IFAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IISub : public Instruction {
  static constexpr SpvOp Opcode = SpvOpISub;
  IISub(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFSub : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFSub;
  IFSub(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IIMul : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIMul;
  IIMul(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFMul : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFMul;
  IFMul(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IUDiv : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUDiv;
  IUDiv(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISDiv : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSDiv;
  ISDiv(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFDiv : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFDiv;
  IFDiv(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IUMod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUMod;
  IUMod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISRem : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSRem;
  ISRem(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISMod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSMod;
  ISMod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFRem : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFRem;
  IFRem(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFMod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFMod;
  IFMod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IVectorTimesScalar : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVectorTimesScalar;
  IVectorTimesScalar(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector() const { return getval<uint32_t>(2); }
uint32_t GetScalar() const { return getval<uint32_t>(3); }
};
struct IMatrixTimesScalar : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMatrixTimesScalar;
  IMatrixTimesScalar(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetMatrix() const { return getval<uint32_t>(2); }
uint32_t GetScalar() const { return getval<uint32_t>(3); }
};
struct IVectorTimesMatrix : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVectorTimesMatrix;
  IVectorTimesMatrix(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector() const { return getval<uint32_t>(2); }
uint32_t GetMatrix() const { return getval<uint32_t>(3); }
};
struct IMatrixTimesVector : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMatrixTimesVector;
  IMatrixTimesVector(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetMatrix() const { return getval<uint32_t>(2); }
uint32_t GetVector() const { return getval<uint32_t>(3); }
};
struct IMatrixTimesMatrix : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMatrixTimesMatrix;
  IMatrixTimesMatrix(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetLeftMatrix() const { return getval<uint32_t>(2); }
uint32_t GetRightMatrix() const { return getval<uint32_t>(3); }
};
struct IOuterProduct : public Instruction {
  static constexpr SpvOp Opcode = SpvOpOuterProduct;
  IOuterProduct(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector1() const { return getval<uint32_t>(2); }
uint32_t GetVector2() const { return getval<uint32_t>(3); }
};
struct IDot : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDot;
  IDot(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector1() const { return getval<uint32_t>(2); }
uint32_t GetVector2() const { return getval<uint32_t>(3); }
};
struct IIAddCarry : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIAddCarry;
  IIAddCarry(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IISubBorrow : public Instruction {
  static constexpr SpvOp Opcode = SpvOpISubBorrow;
  IISubBorrow(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IUMulExtended : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUMulExtended;
  IUMulExtended(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISMulExtended : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSMulExtended;
  ISMulExtended(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IAny : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAny;
  IAny(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector() const { return getval<uint32_t>(2); }
};
struct IAll : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAll;
  IAll(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetVector() const { return getval<uint32_t>(2); }
};
struct IIsNan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIsNan;
  IIsNan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
};
struct IIsInf : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIsInf;
  IIsInf(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
};
struct IIsFinite : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIsFinite;
  IIsFinite(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
};
struct IIsNormal : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIsNormal;
  IIsNormal(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
};
struct ISignBitSet : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSignBitSet;
  ISignBitSet(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
};
struct ILessOrGreater : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLessOrGreater;
  ILessOrGreater(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
uint32_t Gety() const { return getval<uint32_t>(3); }
};
struct IOrdered : public Instruction {
  static constexpr SpvOp Opcode = SpvOpOrdered;
  IOrdered(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
uint32_t Gety() const { return getval<uint32_t>(3); }
};
struct IUnordered : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUnordered;
  IUnordered(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t Getx() const { return getval<uint32_t>(2); }
uint32_t Gety() const { return getval<uint32_t>(3); }
};
struct ILogicalEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLogicalEqual;
  ILogicalEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ILogicalNotEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLogicalNotEqual;
  ILogicalNotEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ILogicalOr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLogicalOr;
  ILogicalOr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ILogicalAnd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLogicalAnd;
  ILogicalAnd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ILogicalNot : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLogicalNot;
  ILogicalNot(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct ISelect : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSelect;
  ISelect(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetCondition() const { return getval<uint32_t>(2); }
uint32_t GetObject1() const { return getval<uint32_t>(3); }
uint32_t GetObject2() const { return getval<uint32_t>(4); }
};
struct IIEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIEqual;
  IIEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IINotEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpINotEqual;
  IINotEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IUGreaterThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUGreaterThan;
  IUGreaterThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISGreaterThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSGreaterThan;
  ISGreaterThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IUGreaterThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUGreaterThanEqual;
  IUGreaterThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISGreaterThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSGreaterThanEqual;
  ISGreaterThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IULessThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpULessThan;
  IULessThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISLessThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSLessThan;
  ISLessThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IULessThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpULessThanEqual;
  IULessThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISLessThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSLessThanEqual;
  ISLessThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFOrdEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFOrdEqual;
  IFOrdEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFUnordEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFUnordEqual;
  IFUnordEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFOrdNotEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFOrdNotEqual;
  IFOrdNotEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFUnordNotEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFUnordNotEqual;
  IFUnordNotEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFOrdLessThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFOrdLessThan;
  IFOrdLessThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFUnordLessThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFUnordLessThan;
  IFUnordLessThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFOrdGreaterThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFOrdGreaterThan;
  IFOrdGreaterThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFUnordGreaterThan : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFUnordGreaterThan;
  IFUnordGreaterThan(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFOrdLessThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFOrdLessThanEqual;
  IFOrdLessThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFUnordLessThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFUnordLessThanEqual;
  IFUnordLessThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFOrdGreaterThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFOrdGreaterThanEqual;
  IFOrdGreaterThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IFUnordGreaterThanEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFUnordGreaterThanEqual;
  IFUnordGreaterThanEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IShiftRightLogical : public Instruction {
  static constexpr SpvOp Opcode = SpvOpShiftRightLogical;
  IShiftRightLogical(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetShift() const { return getval<uint32_t>(3); }
};
struct IShiftRightArithmetic : public Instruction {
  static constexpr SpvOp Opcode = SpvOpShiftRightArithmetic;
  IShiftRightArithmetic(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetShift() const { return getval<uint32_t>(3); }
};
struct IShiftLeftLogical : public Instruction {
  static constexpr SpvOp Opcode = SpvOpShiftLeftLogical;
  IShiftLeftLogical(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetShift() const { return getval<uint32_t>(3); }
};
struct IBitwiseOr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitwiseOr;
  IBitwiseOr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IBitwiseXor : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitwiseXor;
  IBitwiseXor(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IBitwiseAnd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitwiseAnd;
  IBitwiseAnd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct INot : public Instruction {
  static constexpr SpvOp Opcode = SpvOpNot;
  INot(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct IBitFieldInsert : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitFieldInsert;
  IBitFieldInsert(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetInsert() const { return getval<uint32_t>(3); }
uint32_t GetOffset() const { return getval<uint32_t>(4); }
uint32_t GetCount() const { return getval<uint32_t>(5); }
};
struct IBitFieldSExtract : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitFieldSExtract;
  IBitFieldSExtract(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetOffset() const { return getval<uint32_t>(3); }
uint32_t GetCount() const { return getval<uint32_t>(4); }
};
struct IBitFieldUExtract : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitFieldUExtract;
  IBitFieldUExtract(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
uint32_t GetOffset() const { return getval<uint32_t>(3); }
uint32_t GetCount() const { return getval<uint32_t>(4); }
};
struct IBitReverse : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitReverse;
  IBitReverse(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
};
struct IBitCount : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBitCount;
  IBitCount(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBase() const { return getval<uint32_t>(2); }
};
struct IDPdx : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDPdx;
  IDPdx(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IDPdy : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDPdy;
  IDPdy(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IFwidth : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFwidth;
  IFwidth(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IDPdxFine : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDPdxFine;
  IDPdxFine(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IDPdyFine : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDPdyFine;
  IDPdyFine(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IFwidthFine : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFwidthFine;
  IFwidthFine(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IDPdxCoarse : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDPdxCoarse;
  IDPdxCoarse(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IDPdyCoarse : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDPdyCoarse;
  IDPdyCoarse(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IFwidthCoarse : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFwidthCoarse;
  IFwidthCoarse(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetP() const { return getval<uint32_t>(2); }
};
struct IEmitVertex : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEmitVertex;
  IEmitVertex(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IEndPrimitive : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEndPrimitive;
  IEndPrimitive(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IEmitStreamVertex : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEmitStreamVertex;
  IEmitStreamVertex(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetStream() const { return getval<uint32_t>(0); }
};
struct IEndStreamPrimitive : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEndStreamPrimitive;
  IEndStreamPrimitive(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetStream() const { return getval<uint32_t>(0); }
};
struct IControlBarrier : public Instruction {
  static constexpr SpvOp Opcode = SpvOpControlBarrier;
  IControlBarrier(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetExecution() const { return getval<uint32_t>(0); }
uint32_t GetMemory() const { return getval<uint32_t>(1); }
uint32_t GetSemantics() const { return getval<uint32_t>(2); }
};
struct IMemoryBarrier : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMemoryBarrier;
  IMemoryBarrier(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetMemory() const { return getval<uint32_t>(0); }
uint32_t GetSemantics() const { return getval<uint32_t>(1); }
};
struct IAtomicLoad : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicLoad;
  IAtomicLoad(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
};
struct IAtomicStore : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicStore;
  IAtomicStore(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointer() const { return getval<uint32_t>(0); }
uint32_t GetMemory() const { return getval<uint32_t>(1); }
uint32_t GetSemantics() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct IAtomicExchange : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicExchange;
  IAtomicExchange(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicCompareExchange : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicCompareExchange;
  IAtomicCompareExchange(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetEqual() const { return getval<uint32_t>(4); }
uint32_t GetUnequal() const { return getval<uint32_t>(5); }
uint32_t GetValue() const { return getval<uint32_t>(6); }
uint32_t GetComparator() const { return getval<uint32_t>(7); }
};
struct IAtomicCompareExchangeWeak : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicCompareExchangeWeak;
  IAtomicCompareExchangeWeak(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetEqual() const { return getval<uint32_t>(4); }
uint32_t GetUnequal() const { return getval<uint32_t>(5); }
uint32_t GetValue() const { return getval<uint32_t>(6); }
uint32_t GetComparator() const { return getval<uint32_t>(7); }
};
struct IAtomicIIncrement : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicIIncrement;
  IAtomicIIncrement(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
};
struct IAtomicIDecrement : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicIDecrement;
  IAtomicIDecrement(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
};
struct IAtomicIAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicIAdd;
  IAtomicIAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicISub : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicISub;
  IAtomicISub(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicSMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicSMin;
  IAtomicSMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicUMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicUMin;
  IAtomicUMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicSMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicSMax;
  IAtomicSMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicUMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicUMax;
  IAtomicUMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicAnd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicAnd;
  IAtomicAnd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicOr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicOr;
  IAtomicOr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IAtomicXor : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicXor;
  IAtomicXor(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
uint32_t GetValue() const { return getval<uint32_t>(5); }
};
struct IPhi : public Instruction {
  static constexpr SpvOp Opcode = SpvOpPhi;
  IPhi(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
const std::vector<std::pair<uint32_t, uint32_t>> GetOptionalImageOperands() const { return getvec<std::pair<uint32_t, uint32_t>>(2); }
};
struct ILoopMerge : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLoopMerge;
  ILoopMerge(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetMergeBlock() const { return getval<uint32_t>(0); }
uint32_t GetContinueTarget() const { return getval<uint32_t>(1); }
uint32_t GetLoopControl() const { return getval<uint32_t>(2); }
};
struct ISelectionMerge : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSelectionMerge;
  ISelectionMerge(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetMergeBlock() const { return getval<uint32_t>(0); }
uint32_t GetSelectionControl() const { return getval<uint32_t>(1); }
};
struct ILabel : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLabel;
  ILabel(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct IBranch : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBranch;
  IBranch(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTargetLabel() const { return getval<uint32_t>(0); }
};
struct IBranchConditional : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBranchConditional;
  IBranchConditional(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetCondition() const { return getval<uint32_t>(0); }
uint32_t GetTrueLabel() const { return getval<uint32_t>(1); }
uint32_t GetFalseLabel() const { return getval<uint32_t>(2); }
const std::vector<uint32_t> GetBranchweights() const { return getvec<uint32_t>(3); }
};
struct ISwitch : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSwitch;
  ISwitch(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetSelector() const { return getval<uint32_t>(0); }
uint32_t GetDefault() const { return getval<uint32_t>(1); }
const std::vector<std::pair<uint32_t, uint32_t>> GetTarget() const { return getvec<std::pair<uint32_t, uint32_t>>(2); }
};
struct IKill : public Instruction {
  static constexpr SpvOp Opcode = SpvOpKill;
  IKill(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IReturn : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReturn;
  IReturn(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IReturnValue : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReturnValue;
  IReturnValue(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetValue() const { return getval<uint32_t>(0); }
};
struct IUnreachable : public Instruction {
  static constexpr SpvOp Opcode = SpvOpUnreachable;
  IUnreachable(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct ILifetimeStart : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLifetimeStart;
  ILifetimeStart(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointer() const { return getval<uint32_t>(0); }
uint32_t GetSize() const { return getval<uint32_t>(1); }
};
struct ILifetimeStop : public Instruction {
  static constexpr SpvOp Opcode = SpvOpLifetimeStop;
  ILifetimeStop(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointer() const { return getval<uint32_t>(0); }
uint32_t GetSize() const { return getval<uint32_t>(1); }
};
struct IGroupAsyncCopy : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupAsyncCopy;
  IGroupAsyncCopy(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetDestination() const { return getval<uint32_t>(3); }
uint32_t GetSource() const { return getval<uint32_t>(4); }
uint32_t GetNumElements() const { return getval<uint32_t>(5); }
uint32_t GetStride() const { return getval<uint32_t>(6); }
uint32_t GetEvent() const { return getval<uint32_t>(7); }
};
struct IGroupWaitEvents : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupWaitEvents;
  IGroupWaitEvents(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetExecution() const { return getval<uint32_t>(0); }
uint32_t GetNumEvents() const { return getval<uint32_t>(1); }
uint32_t GetEventsList() const { return getval<uint32_t>(2); }
};
struct IGroupAll : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupAll;
  IGroupAll(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPredicate() const { return getval<uint32_t>(3); }
};
struct IGroupAny : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupAny;
  IGroupAny(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPredicate() const { return getval<uint32_t>(3); }
};
struct IGroupBroadcast : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupBroadcast;
  IGroupBroadcast(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetLocalId() const { return getval<uint32_t>(4); }
};
struct IGroupIAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupIAdd;
  IGroupIAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupFAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupFAdd;
  IGroupFAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupFMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupFMin;
  IGroupFMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupUMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupUMin;
  IGroupUMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupSMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupSMin;
  IGroupSMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupFMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupFMax;
  IGroupFMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupUMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupUMax;
  IGroupUMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupSMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupSMax;
  IGroupSMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IReadPipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReadPipe;
  IReadPipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetPointer() const { return getval<uint32_t>(3); }
uint32_t GetPacketSize() const { return getval<uint32_t>(4); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(5); }
};
struct IWritePipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpWritePipe;
  IWritePipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetPointer() const { return getval<uint32_t>(3); }
uint32_t GetPacketSize() const { return getval<uint32_t>(4); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(5); }
};
struct IReservedReadPipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReservedReadPipe;
  IReservedReadPipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetReserveId() const { return getval<uint32_t>(3); }
uint32_t GetIndex() const { return getval<uint32_t>(4); }
uint32_t GetPointer() const { return getval<uint32_t>(5); }
uint32_t GetPacketSize() const { return getval<uint32_t>(6); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(7); }
};
struct IReservedWritePipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReservedWritePipe;
  IReservedWritePipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetReserveId() const { return getval<uint32_t>(3); }
uint32_t GetIndex() const { return getval<uint32_t>(4); }
uint32_t GetPointer() const { return getval<uint32_t>(5); }
uint32_t GetPacketSize() const { return getval<uint32_t>(6); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(7); }
};
struct IReserveReadPipePackets : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReserveReadPipePackets;
  IReserveReadPipePackets(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetNumPackets() const { return getval<uint32_t>(3); }
uint32_t GetPacketSize() const { return getval<uint32_t>(4); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(5); }
};
struct IReserveWritePipePackets : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReserveWritePipePackets;
  IReserveWritePipePackets(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetNumPackets() const { return getval<uint32_t>(3); }
uint32_t GetPacketSize() const { return getval<uint32_t>(4); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(5); }
};
struct ICommitReadPipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCommitReadPipe;
  ICommitReadPipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPipe() const { return getval<uint32_t>(0); }
uint32_t GetReserveId() const { return getval<uint32_t>(1); }
uint32_t GetPacketSize() const { return getval<uint32_t>(2); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(3); }
};
struct ICommitWritePipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCommitWritePipe;
  ICommitWritePipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPipe() const { return getval<uint32_t>(0); }
uint32_t GetReserveId() const { return getval<uint32_t>(1); }
uint32_t GetPacketSize() const { return getval<uint32_t>(2); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(3); }
};
struct IIsValidReserveId : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIsValidReserveId;
  IIsValidReserveId(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetReserveId() const { return getval<uint32_t>(2); }
};
struct IGetNumPipePackets : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetNumPipePackets;
  IGetNumPipePackets(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetPacketSize() const { return getval<uint32_t>(3); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(4); }
};
struct IGetMaxPipePackets : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetMaxPipePackets;
  IGetMaxPipePackets(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipe() const { return getval<uint32_t>(2); }
uint32_t GetPacketSize() const { return getval<uint32_t>(3); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(4); }
};
struct IGroupReserveReadPipePackets : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupReserveReadPipePackets;
  IGroupReserveReadPipePackets(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPipe() const { return getval<uint32_t>(3); }
uint32_t GetNumPackets() const { return getval<uint32_t>(4); }
uint32_t GetPacketSize() const { return getval<uint32_t>(5); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(6); }
};
struct IGroupReserveWritePipePackets : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupReserveWritePipePackets;
  IGroupReserveWritePipePackets(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPipe() const { return getval<uint32_t>(3); }
uint32_t GetNumPackets() const { return getval<uint32_t>(4); }
uint32_t GetPacketSize() const { return getval<uint32_t>(5); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(6); }
};
struct IGroupCommitReadPipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupCommitReadPipe;
  IGroupCommitReadPipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetExecution() const { return getval<uint32_t>(0); }
uint32_t GetPipe() const { return getval<uint32_t>(1); }
uint32_t GetReserveId() const { return getval<uint32_t>(2); }
uint32_t GetPacketSize() const { return getval<uint32_t>(3); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(4); }
};
struct IGroupCommitWritePipe : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupCommitWritePipe;
  IGroupCommitWritePipe(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetExecution() const { return getval<uint32_t>(0); }
uint32_t GetPipe() const { return getval<uint32_t>(1); }
uint32_t GetReserveId() const { return getval<uint32_t>(2); }
uint32_t GetPacketSize() const { return getval<uint32_t>(3); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(4); }
};
struct IEnqueueMarker : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEnqueueMarker;
  IEnqueueMarker(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetQueue() const { return getval<uint32_t>(2); }
uint32_t GetNumEvents() const { return getval<uint32_t>(3); }
uint32_t GetWaitEvents() const { return getval<uint32_t>(4); }
uint32_t GetRetEvent() const { return getval<uint32_t>(5); }
};
struct IEnqueueKernel : public Instruction {
  static constexpr SpvOp Opcode = SpvOpEnqueueKernel;
  IEnqueueKernel(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetQueue() const { return getval<uint32_t>(2); }
uint32_t GetFlags() const { return getval<uint32_t>(3); }
uint32_t GetNDRange() const { return getval<uint32_t>(4); }
uint32_t GetNumEvents() const { return getval<uint32_t>(5); }
uint32_t GetWaitEvents() const { return getval<uint32_t>(6); }
uint32_t GetRetEvent() const { return getval<uint32_t>(7); }
uint32_t GetInvoke() const { return getval<uint32_t>(8); }
uint32_t GetParam() const { return getval<uint32_t>(9); }
uint32_t GetParamSize() const { return getval<uint32_t>(10); }
uint32_t GetParamAlign() const { return getval<uint32_t>(11); }
const std::vector<uint32_t> GetLocalSize() const { return getvec<uint32_t>(12); }
};
struct IGetKernelNDrangeSubGroupCount : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetKernelNDrangeSubGroupCount;
  IGetKernelNDrangeSubGroupCount(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetNDRange() const { return getval<uint32_t>(2); }
uint32_t GetInvoke() const { return getval<uint32_t>(3); }
uint32_t GetParam() const { return getval<uint32_t>(4); }
uint32_t GetParamSize() const { return getval<uint32_t>(5); }
uint32_t GetParamAlign() const { return getval<uint32_t>(6); }
};
struct IGetKernelNDrangeMaxSubGroupSize : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetKernelNDrangeMaxSubGroupSize;
  IGetKernelNDrangeMaxSubGroupSize(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetNDRange() const { return getval<uint32_t>(2); }
uint32_t GetInvoke() const { return getval<uint32_t>(3); }
uint32_t GetParam() const { return getval<uint32_t>(4); }
uint32_t GetParamSize() const { return getval<uint32_t>(5); }
uint32_t GetParamAlign() const { return getval<uint32_t>(6); }
};
struct IGetKernelWorkGroupSize : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetKernelWorkGroupSize;
  IGetKernelWorkGroupSize(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetInvoke() const { return getval<uint32_t>(2); }
uint32_t GetParam() const { return getval<uint32_t>(3); }
uint32_t GetParamSize() const { return getval<uint32_t>(4); }
uint32_t GetParamAlign() const { return getval<uint32_t>(5); }
};
struct IGetKernelPreferredWorkGroupSizeMultiple : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetKernelPreferredWorkGroupSizeMultiple;
  IGetKernelPreferredWorkGroupSizeMultiple(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetInvoke() const { return getval<uint32_t>(2); }
uint32_t GetParam() const { return getval<uint32_t>(3); }
uint32_t GetParamSize() const { return getval<uint32_t>(4); }
uint32_t GetParamAlign() const { return getval<uint32_t>(5); }
};
struct IRetainEvent : public Instruction {
  static constexpr SpvOp Opcode = SpvOpRetainEvent;
  IRetainEvent(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetEvent() const { return getval<uint32_t>(0); }
};
struct IReleaseEvent : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReleaseEvent;
  IReleaseEvent(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetEvent() const { return getval<uint32_t>(0); }
};
struct ICreateUserEvent : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCreateUserEvent;
  ICreateUserEvent(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct IIsValidEvent : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIsValidEvent;
  IIsValidEvent(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetEvent() const { return getval<uint32_t>(2); }
};
struct ISetUserEventStatus : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSetUserEventStatus;
  ISetUserEventStatus(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetEvent() const { return getval<uint32_t>(0); }
uint32_t GetStatus() const { return getval<uint32_t>(1); }
};
struct ICaptureEventProfilingInfo : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCaptureEventProfilingInfo;
  ICaptureEventProfilingInfo(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetEvent() const { return getval<uint32_t>(0); }
uint32_t GetProfilingInfo() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
};
struct IGetDefaultQueue : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetDefaultQueue;
  IGetDefaultQueue(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct IBuildNDRange : public Instruction {
  static constexpr SpvOp Opcode = SpvOpBuildNDRange;
  IBuildNDRange(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetGlobalWorkSize() const { return getval<uint32_t>(2); }
uint32_t GetLocalWorkSize() const { return getval<uint32_t>(3); }
uint32_t GetGlobalWorkOffset() const { return getval<uint32_t>(4); }
};
struct IImageSparseSampleImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleImplicitLod;
  IImageSparseSampleImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSparseSampleExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleExplicitLod;
  IImageSparseSampleExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSparseSampleDrefImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleDrefImplicitLod;
  IImageSparseSampleDrefImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSparseSampleDrefExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleDrefExplicitLod;
  IImageSparseSampleDrefExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSparseSampleProjImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleProjImplicitLod;
  IImageSparseSampleProjImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSparseSampleProjExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleProjExplicitLod;
  IImageSparseSampleProjExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSparseSampleProjDrefImplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleProjDrefImplicitLod;
  IImageSparseSampleProjDrefImplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSparseSampleProjDrefExplicitLod : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseSampleProjDrefExplicitLod;
  IImageSparseSampleProjDrefExplicitLod(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSparseFetch : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseFetch;
  IImageSparseFetch(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct IImageSparseGather : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseGather;
  IImageSparseGather(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetComponent() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSparseDrefGather : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseDrefGather;
  IImageSparseDrefGather(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetOptionalImageOperands() const { return getval<uint32_t>(4); }
uint32_t GetImageOperands() const { return getval<uint32_t>(5); }
};
struct IImageSparseTexelsResident : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseTexelsResident;
  IImageSparseTexelsResident(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetResidentCode() const { return getval<uint32_t>(2); }
};
struct INoLine : public Instruction {
  static constexpr SpvOp Opcode = SpvOpNoLine;
  INoLine(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct IAtomicFlagTestAndSet : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicFlagTestAndSet;
  IAtomicFlagTestAndSet(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetMemory() const { return getval<uint32_t>(3); }
uint32_t GetSemantics() const { return getval<uint32_t>(4); }
};
struct IAtomicFlagClear : public Instruction {
  static constexpr SpvOp Opcode = SpvOpAtomicFlagClear;
  IAtomicFlagClear(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointer() const { return getval<uint32_t>(0); }
uint32_t GetMemory() const { return getval<uint32_t>(1); }
uint32_t GetSemantics() const { return getval<uint32_t>(2); }
};
struct IImageSparseRead : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSparseRead;
  IImageSparseRead(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetImageOperands() const { return getval<uint32_t>(4); }
};
struct ISizeOf : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSizeOf;
  ISizeOf(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
};
struct ITypePipeStorage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypePipeStorage;
  ITypePipeStorage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct IConstantPipeStorage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpConstantPipeStorage;
  IConstantPipeStorage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPacketSize() const { return getval<uint32_t>(2); }
uint32_t GetPacketAlignment() const { return getval<uint32_t>(3); }
uint32_t GetCapacity() const { return getval<uint32_t>(4); }
};
struct ICreatePipeFromPipeStorage : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCreatePipeFromPipeStorage;
  ICreatePipeFromPipeStorage(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPipeStorage() const { return getval<uint32_t>(2); }
};
struct IGetKernelLocalSizeForSubgroupCount : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetKernelLocalSizeForSubgroupCount;
  IGetKernelLocalSizeForSubgroupCount(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSubgroupCount() const { return getval<uint32_t>(2); }
uint32_t GetInvoke() const { return getval<uint32_t>(3); }
uint32_t GetParam() const { return getval<uint32_t>(4); }
uint32_t GetParamSize() const { return getval<uint32_t>(5); }
uint32_t GetParamAlign() const { return getval<uint32_t>(6); }
};
struct IGetKernelMaxNumSubgroups : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGetKernelMaxNumSubgroups;
  IGetKernelMaxNumSubgroups(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetInvoke() const { return getval<uint32_t>(2); }
uint32_t GetParam() const { return getval<uint32_t>(3); }
uint32_t GetParamSize() const { return getval<uint32_t>(4); }
uint32_t GetParamAlign() const { return getval<uint32_t>(5); }
};
struct ITypeNamedBarrier : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeNamedBarrier;
  ITypeNamedBarrier(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct INamedBarrierInitialize : public Instruction {
  static constexpr SpvOp Opcode = SpvOpNamedBarrierInitialize;
  INamedBarrierInitialize(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSubgroupCount() const { return getval<uint32_t>(2); }
};
struct IMemoryNamedBarrier : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMemoryNamedBarrier;
  IMemoryNamedBarrier(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetNamedBarrier() const { return getval<uint32_t>(0); }
uint32_t GetMemory() const { return getval<uint32_t>(1); }
uint32_t GetSemantics() const { return getval<uint32_t>(2); }
};
struct IModuleProcessed : public Instruction {
  static constexpr SpvOp Opcode = SpvOpModuleProcessed;
  IModuleProcessed(const spv_parsed_instruction_t *i) : Instruction(i) {}
  const char * GetProcess() const { return getstr(0); }
};
struct IExecutionModeId : public Instruction {
  static constexpr SpvOp Opcode = SpvOpExecutionModeId;
  IExecutionModeId(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetEntryPoint() const { return getval<uint32_t>(0); }
SpvExecutionMode GetMode() const { return getval<SpvExecutionMode>(1); }
};
struct IDecorateId : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDecorateId;
  IDecorateId(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTarget() const { return getval<uint32_t>(0); }
SpvDecoration GetDecoration() const { return getval<SpvDecoration>(1); }
};
struct IGroupNonUniformElect : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformElect;
  IGroupNonUniformElect(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
};
struct IGroupNonUniformAll : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformAll;
  IGroupNonUniformAll(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPredicate() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformAny : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformAny;
  IGroupNonUniformAny(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPredicate() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformAllEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformAllEqual;
  IGroupNonUniformAllEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformBroadcast : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBroadcast;
  IGroupNonUniformBroadcast(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetId() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformBroadcastFirst : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBroadcastFirst;
  IGroupNonUniformBroadcastFirst(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformBallot : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBallot;
  IGroupNonUniformBallot(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetPredicate() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformInverseBallot : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformInverseBallot;
  IGroupNonUniformInverseBallot(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformBallotBitExtract : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBallotBitExtract;
  IGroupNonUniformBallotBitExtract(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetIndex() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformBallotBitCount : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBallotBitCount;
  IGroupNonUniformBallotBitCount(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformBallotFindLSB : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBallotFindLSB;
  IGroupNonUniformBallotFindLSB(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformBallotFindMSB : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBallotFindMSB;
  IGroupNonUniformBallotFindMSB(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct IGroupNonUniformShuffle : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformShuffle;
  IGroupNonUniformShuffle(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetId() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformShuffleXor : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformShuffleXor;
  IGroupNonUniformShuffleXor(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetMask() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformShuffleUp : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformShuffleUp;
  IGroupNonUniformShuffleUp(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetDelta() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformShuffleDown : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformShuffleDown;
  IGroupNonUniformShuffleDown(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetDelta() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformIAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformIAdd;
  IGroupNonUniformIAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformFAdd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformFAdd;
  IGroupNonUniformFAdd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformIMul : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformIMul;
  IGroupNonUniformIMul(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformFMul : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformFMul;
  IGroupNonUniformFMul(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformSMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformSMin;
  IGroupNonUniformSMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformUMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformUMin;
  IGroupNonUniformUMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformFMin : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformFMin;
  IGroupNonUniformFMin(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformSMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformSMax;
  IGroupNonUniformSMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformUMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformUMax;
  IGroupNonUniformUMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformFMax : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformFMax;
  IGroupNonUniformFMax(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformBitwiseAnd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBitwiseAnd;
  IGroupNonUniformBitwiseAnd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformBitwiseOr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBitwiseOr;
  IGroupNonUniformBitwiseOr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformBitwiseXor : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformBitwiseXor;
  IGroupNonUniformBitwiseXor(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformLogicalAnd : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformLogicalAnd;
  IGroupNonUniformLogicalAnd(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformLogicalOr : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformLogicalOr;
  IGroupNonUniformLogicalOr(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformLogicalXor : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformLogicalXor;
  IGroupNonUniformLogicalXor(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetValue() const { return getval<uint32_t>(4); }
uint32_t GetClusterSize() const { return getval<uint32_t>(5); }
};
struct IGroupNonUniformQuadBroadcast : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformQuadBroadcast;
  IGroupNonUniformQuadBroadcast(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetIndex() const { return getval<uint32_t>(4); }
};
struct IGroupNonUniformQuadSwap : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformQuadSwap;
  IGroupNonUniformQuadSwap(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
uint32_t GetDirection() const { return getval<uint32_t>(4); }
};
struct ICopyLogical : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCopyLogical;
  ICopyLogical(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand() const { return getval<uint32_t>(2); }
};
struct IPtrEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpPtrEqual;
  IPtrEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IPtrNotEqual : public Instruction {
  static constexpr SpvOp Opcode = SpvOpPtrNotEqual;
  IPtrNotEqual(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct IPtrDiff : public Instruction {
  static constexpr SpvOp Opcode = SpvOpPtrDiff;
  IPtrDiff(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetOperand1() const { return getval<uint32_t>(2); }
uint32_t GetOperand2() const { return getval<uint32_t>(3); }
};
struct ISubgroupBallotKHR : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupBallotKHR;
  ISubgroupBallotKHR(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPredicate() const { return getval<uint32_t>(2); }
};
struct ISubgroupFirstInvocationKHR : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupFirstInvocationKHR;
  ISubgroupFirstInvocationKHR(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
};
struct ISubgroupAllKHR : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAllKHR;
  ISubgroupAllKHR(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPredicate() const { return getval<uint32_t>(2); }
};
struct ISubgroupAnyKHR : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAnyKHR;
  ISubgroupAnyKHR(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPredicate() const { return getval<uint32_t>(2); }
};
struct ISubgroupAllEqualKHR : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAllEqualKHR;
  ISubgroupAllEqualKHR(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPredicate() const { return getval<uint32_t>(2); }
};
struct ISubgroupReadInvocationKHR : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupReadInvocationKHR;
  ISubgroupReadInvocationKHR(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
uint32_t GetIndex() const { return getval<uint32_t>(3); }
};
struct IGroupIAddNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupIAddNonUniformAMD;
  IGroupIAddNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupFAddNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupFAddNonUniformAMD;
  IGroupFAddNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupFMinNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupFMinNonUniformAMD;
  IGroupFMinNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupUMinNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupUMinNonUniformAMD;
  IGroupUMinNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupSMinNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupSMinNonUniformAMD;
  IGroupSMinNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupFMaxNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupFMaxNonUniformAMD;
  IGroupFMaxNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupUMaxNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupUMaxNonUniformAMD;
  IGroupUMaxNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IGroupSMaxNonUniformAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupSMaxNonUniformAMD;
  IGroupSMaxNonUniformAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
SpvGroupOperation GetOperation() const { return getval<SpvGroupOperation>(3); }
uint32_t GetX() const { return getval<uint32_t>(4); }
};
struct IFragmentMaskFetchAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFragmentMaskFetchAMD;
  IFragmentMaskFetchAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
};
struct IFragmentFetchAMD : public Instruction {
  static constexpr SpvOp Opcode = SpvOpFragmentFetchAMD;
  IFragmentFetchAMD(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetFragmentIndex() const { return getval<uint32_t>(4); }
};
struct IImageSampleFootprintNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpImageSampleFootprintNV;
  IImageSampleFootprintNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSampledImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetGranularity() const { return getval<uint32_t>(4); }
uint32_t GetCoarse() const { return getval<uint32_t>(5); }
uint32_t GetImageOperands() const { return getval<uint32_t>(6); }
};
struct IGroupNonUniformPartitionNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpGroupNonUniformPartitionNV;
  IGroupNonUniformPartitionNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetValue() const { return getval<uint32_t>(2); }
};
struct IWritePackedPrimitiveIndices4x8NV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpWritePackedPrimitiveIndices4x8NV;
  IWritePackedPrimitiveIndices4x8NV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIndexOffset() const { return getval<uint32_t>(0); }
uint32_t GetPackedIndices() const { return getval<uint32_t>(1); }
};
struct IReportIntersectionNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpReportIntersectionNV;
  IReportIntersectionNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetHit() const { return getval<uint32_t>(2); }
uint32_t GetHitKind() const { return getval<uint32_t>(3); }
};
struct IIgnoreIntersectionNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpIgnoreIntersectionNV;
  IIgnoreIntersectionNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct ITerminateRayNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTerminateRayNV;
  ITerminateRayNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  
};
struct ITraceNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTraceNV;
  ITraceNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetAccel() const { return getval<uint32_t>(0); }
uint32_t GetRayFlags() const { return getval<uint32_t>(1); }
uint32_t GetCullMask() const { return getval<uint32_t>(2); }
uint32_t GetSBTOffset() const { return getval<uint32_t>(3); }
uint32_t GetSBTStride() const { return getval<uint32_t>(4); }
uint32_t GetMissIndex() const { return getval<uint32_t>(5); }
uint32_t GetRayOrigin() const { return getval<uint32_t>(6); }
uint32_t GetRayTmin() const { return getval<uint32_t>(7); }
uint32_t GetRayDirection() const { return getval<uint32_t>(8); }
uint32_t GetRayTmax() const { return getval<uint32_t>(9); }
uint32_t GetPayloadId() const { return getval<uint32_t>(10); }
};
struct ITypeAccelerationStructureNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAccelerationStructureNV;
  ITypeAccelerationStructureNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct IExecuteCallableNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpExecuteCallableNV;
  IExecuteCallableNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetSBTIndex() const { return getval<uint32_t>(0); }
uint32_t GetCallableDataId() const { return getval<uint32_t>(1); }
};
struct ITypeCooperativeMatrixNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeCooperativeMatrixNV;
  ITypeCooperativeMatrixNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetComponentType() const { return getval<uint32_t>(1); }
uint32_t GetExecution() const { return getval<uint32_t>(2); }
uint32_t GetRows() const { return getval<uint32_t>(3); }
uint32_t GetColumns() const { return getval<uint32_t>(4); }
};
struct ICooperativeMatrixLoadNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCooperativeMatrixLoadNV;
  ICooperativeMatrixLoadNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPointer() const { return getval<uint32_t>(2); }
uint32_t GetStride() const { return getval<uint32_t>(3); }
uint32_t GetColumnMajor() const { return getval<uint32_t>(4); }
uint32_t GetMemoryAccess() const { return getval<uint32_t>(5); }
};
struct ICooperativeMatrixStoreNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCooperativeMatrixStoreNV;
  ICooperativeMatrixStoreNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPointer() const { return getval<uint32_t>(0); }
uint32_t GetObject() const { return getval<uint32_t>(1); }
uint32_t GetStride() const { return getval<uint32_t>(2); }
uint32_t GetColumnMajor() const { return getval<uint32_t>(3); }
uint32_t GetMemoryAccess() const { return getval<uint32_t>(4); }
};
struct ICooperativeMatrixMulAddNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCooperativeMatrixMulAddNV;
  ICooperativeMatrixMulAddNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetA() const { return getval<uint32_t>(2); }
uint32_t GetB() const { return getval<uint32_t>(3); }
uint32_t GetC() const { return getval<uint32_t>(4); }
};
struct ICooperativeMatrixLengthNV : public Instruction {
  static constexpr SpvOp Opcode = SpvOpCooperativeMatrixLengthNV;
  ICooperativeMatrixLengthNV(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetType() const { return getval<uint32_t>(2); }
};
struct ISubgroupShuffleINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupShuffleINTEL;
  ISubgroupShuffleINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetData() const { return getval<uint32_t>(2); }
uint32_t GetInvocationId() const { return getval<uint32_t>(3); }
};
struct ISubgroupShuffleDownINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupShuffleDownINTEL;
  ISubgroupShuffleDownINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetCurrent() const { return getval<uint32_t>(2); }
uint32_t GetNext() const { return getval<uint32_t>(3); }
uint32_t GetDelta() const { return getval<uint32_t>(4); }
};
struct ISubgroupShuffleUpINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupShuffleUpINTEL;
  ISubgroupShuffleUpINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPrevious() const { return getval<uint32_t>(2); }
uint32_t GetCurrent() const { return getval<uint32_t>(3); }
uint32_t GetDelta() const { return getval<uint32_t>(4); }
};
struct ISubgroupShuffleXorINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupShuffleXorINTEL;
  ISubgroupShuffleXorINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetData() const { return getval<uint32_t>(2); }
uint32_t GetValue() const { return getval<uint32_t>(3); }
};
struct ISubgroupBlockReadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupBlockReadINTEL;
  ISubgroupBlockReadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPtr() const { return getval<uint32_t>(2); }
};
struct ISubgroupBlockWriteINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupBlockWriteINTEL;
  ISubgroupBlockWriteINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetPtr() const { return getval<uint32_t>(0); }
uint32_t GetData() const { return getval<uint32_t>(1); }
};
struct ISubgroupImageBlockReadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupImageBlockReadINTEL;
  ISubgroupImageBlockReadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
};
struct ISubgroupImageBlockWriteINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupImageBlockWriteINTEL;
  ISubgroupImageBlockWriteINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetImage() const { return getval<uint32_t>(0); }
uint32_t GetCoordinate() const { return getval<uint32_t>(1); }
uint32_t GetData() const { return getval<uint32_t>(2); }
};
struct ISubgroupImageMediaBlockReadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupImageMediaBlockReadINTEL;
  ISubgroupImageMediaBlockReadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImage() const { return getval<uint32_t>(2); }
uint32_t GetCoordinate() const { return getval<uint32_t>(3); }
uint32_t GetWidth() const { return getval<uint32_t>(4); }
uint32_t GetHeight() const { return getval<uint32_t>(5); }
};
struct ISubgroupImageMediaBlockWriteINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupImageMediaBlockWriteINTEL;
  ISubgroupImageMediaBlockWriteINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetImage() const { return getval<uint32_t>(0); }
uint32_t GetCoordinate() const { return getval<uint32_t>(1); }
uint32_t GetWidth() const { return getval<uint32_t>(2); }
uint32_t GetHeight() const { return getval<uint32_t>(3); }
uint32_t GetData() const { return getval<uint32_t>(4); }
};
struct IDecorateString : public Instruction {
  static constexpr SpvOp Opcode = SpvOpDecorateString;
  IDecorateString(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetTarget() const { return getval<uint32_t>(0); }
SpvDecoration GetDecoration() const { return getval<SpvDecoration>(1); }
};
struct IMemberDecorateString : public Instruction {
  static constexpr SpvOp Opcode = SpvOpMemberDecorateString;
  IMemberDecorateString(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetStructType() const { return getval<uint32_t>(0); }
uint32_t GetMember() const { return getval<uint32_t>(1); }
SpvDecoration GetDecoration() const { return getval<SpvDecoration>(2); }
};
struct IVmeImageINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpVmeImageINTEL;
  IVmeImageINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImageType() const { return getval<uint32_t>(2); }
uint32_t GetSampler() const { return getval<uint32_t>(3); }
};
struct ITypeVmeImageINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeVmeImageINTEL;
  ITypeVmeImageINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
uint32_t GetImageType() const { return getval<uint32_t>(1); }
};
struct ITypeAvcImePayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcImePayloadINTEL;
  ITypeAvcImePayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcRefPayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcRefPayloadINTEL;
  ITypeAvcRefPayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcSicPayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcSicPayloadINTEL;
  ITypeAvcSicPayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcMcePayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcMcePayloadINTEL;
  ITypeAvcMcePayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcMceResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcMceResultINTEL;
  ITypeAvcMceResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcImeResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcImeResultINTEL;
  ITypeAvcImeResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcImeResultSingleReferenceStreamoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcImeResultSingleReferenceStreamoutINTEL;
  ITypeAvcImeResultSingleReferenceStreamoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcImeResultDualReferenceStreamoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcImeResultDualReferenceStreamoutINTEL;
  ITypeAvcImeResultDualReferenceStreamoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcImeSingleReferenceStreaminINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcImeSingleReferenceStreaminINTEL;
  ITypeAvcImeSingleReferenceStreaminINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcImeDualReferenceStreaminINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcImeDualReferenceStreaminINTEL;
  ITypeAvcImeDualReferenceStreaminINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcRefResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcRefResultINTEL;
  ITypeAvcRefResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ITypeAvcSicResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpTypeAvcSicResultINTEL;
  ITypeAvcSicResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResult() const { return getval<uint32_t>(0); }
};
struct ISubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL;
  ISubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSliceType() const { return getval<uint32_t>(2); }
uint32_t GetQp() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL;
  ISubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetReferenceBasePenalty() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceGetDefaultInterShapePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL;
  ISubgroupAvcMceGetDefaultInterShapePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSliceType() const { return getval<uint32_t>(2); }
uint32_t GetQp() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceSetInterShapePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetInterShapePenaltyINTEL;
  ISubgroupAvcMceSetInterShapePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPackedShapePenalty() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL;
  ISubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSliceType() const { return getval<uint32_t>(2); }
uint32_t GetQp() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceSetInterDirectionPenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetInterDirectionPenaltyINTEL;
  ISubgroupAvcMceSetInterDirectionPenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetDirectionCost() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL;
  ISubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSliceType() const { return getval<uint32_t>(2); }
uint32_t GetQp() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL;
  ISubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSliceType() const { return getval<uint32_t>(2); }
uint32_t GetQp() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL;
  ISubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL;
  ISubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL;
  ISubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISubgroupAvcMceSetMotionVectorCostFunctionINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetMotionVectorCostFunctionINTEL;
  ISubgroupAvcMceSetMotionVectorCostFunctionINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPackedCostCenterDelta() const { return getval<uint32_t>(2); }
uint32_t GetPackedCostTable() const { return getval<uint32_t>(3); }
uint32_t GetCostPrecision() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL;
  ISubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSliceType() const { return getval<uint32_t>(2); }
uint32_t GetQp() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL;
  ISubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL;
  ISubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
};
struct ISubgroupAvcMceSetAcOnlyHaarINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetAcOnlyHaarINTEL;
  ISubgroupAvcMceSetAcOnlyHaarINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL;
  ISubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSourceFieldPolarity() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL;
  ISubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetReferenceFieldPolarity() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL;
  ISubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetForwardReferenceFieldPolarity() const { return getval<uint32_t>(2); }
uint32_t GetBackwardReferenceFieldPolarity() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcMceConvertToImePayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceConvertToImePayloadINTEL;
  ISubgroupAvcMceConvertToImePayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceConvertToImeResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceConvertToImeResultINTEL;
  ISubgroupAvcMceConvertToImeResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceConvertToRefPayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceConvertToRefPayloadINTEL;
  ISubgroupAvcMceConvertToRefPayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceConvertToRefResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceConvertToRefResultINTEL;
  ISubgroupAvcMceConvertToRefResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceConvertToSicPayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceConvertToSicPayloadINTEL;
  ISubgroupAvcMceConvertToSicPayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceConvertToSicResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceConvertToSicResultINTEL;
  ISubgroupAvcMceConvertToSicResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetMotionVectorsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetMotionVectorsINTEL;
  ISubgroupAvcMceGetMotionVectorsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterDistortionsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterDistortionsINTEL;
  ISubgroupAvcMceGetInterDistortionsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetBestInterDistortionsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetBestInterDistortionsINTEL;
  ISubgroupAvcMceGetBestInterDistortionsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterMajorShapeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterMajorShapeINTEL;
  ISubgroupAvcMceGetInterMajorShapeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterMinorShapeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterMinorShapeINTEL;
  ISubgroupAvcMceGetInterMinorShapeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterDirectionsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterDirectionsINTEL;
  ISubgroupAvcMceGetInterDirectionsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterMotionVectorCountINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterMotionVectorCountINTEL;
  ISubgroupAvcMceGetInterMotionVectorCountINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterReferenceIdsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterReferenceIdsINTEL;
  ISubgroupAvcMceGetInterReferenceIdsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL;
  ISubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPackedReferenceIds() const { return getval<uint32_t>(2); }
uint32_t GetPackedReferenceParameterFieldPolarities() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeInitializeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeInitializeINTEL;
  ISubgroupAvcImeInitializeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcCoord() const { return getval<uint32_t>(2); }
uint32_t GetPartitionMask() const { return getval<uint32_t>(3); }
uint32_t GetSADAdjustment() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeSetSingleReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeSetSingleReferenceINTEL;
  ISubgroupAvcImeSetSingleReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetRefOffset() const { return getval<uint32_t>(2); }
uint32_t GetSearchWindowConfig() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeSetDualReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeSetDualReferenceINTEL;
  ISubgroupAvcImeSetDualReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetFwdRefOffset() const { return getval<uint32_t>(2); }
uint32_t GetBwdRefOffset() const { return getval<uint32_t>(3); }
uint32_t GetSearchWindowConfig() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcImeRefWindowSizeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeRefWindowSizeINTEL;
  ISubgroupAvcImeRefWindowSizeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSearchWindowConfig() const { return getval<uint32_t>(2); }
uint32_t GetDualRef() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeAdjustRefOffsetINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeAdjustRefOffsetINTEL;
  ISubgroupAvcImeAdjustRefOffsetINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetRefOffset() const { return getval<uint32_t>(2); }
uint32_t GetSrcCoord() const { return getval<uint32_t>(3); }
uint32_t GetRefWindowSize() const { return getval<uint32_t>(4); }
uint32_t GetImageSize() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcImeConvertToMcePayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeConvertToMcePayloadINTEL;
  ISubgroupAvcImeConvertToMcePayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeSetMaxMotionVectorCountINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeSetMaxMotionVectorCountINTEL;
  ISubgroupAvcImeSetMaxMotionVectorCountINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetMaxMotionVectorCount() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeSetUnidirectionalMixDisableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeSetUnidirectionalMixDisableINTEL;
  ISubgroupAvcImeSetUnidirectionalMixDisableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeSetEarlySearchTerminationThresholdINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL;
  ISubgroupAvcImeSetEarlySearchTerminationThresholdINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetThreshold() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeSetWeightedSadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeSetWeightedSadINTEL;
  ISubgroupAvcImeSetWeightedSadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPackedSadWeights() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeEvaluateWithSingleReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithSingleReferenceINTEL;
  ISubgroupAvcImeEvaluateWithSingleReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetRefImage() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeEvaluateWithDualReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithDualReferenceINTEL;
  ISubgroupAvcImeEvaluateWithDualReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetFwdRefImage() const { return getval<uint32_t>(3); }
uint32_t GetBwdRefImage() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL;
  ISubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetRefImage() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
uint32_t GetStreaminComponents() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL;
  ISubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetFwdRefImage() const { return getval<uint32_t>(3); }
uint32_t GetBwdRefImage() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
uint32_t GetStreaminComponents() const { return getval<uint32_t>(6); }
};
struct ISubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL;
  ISubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetRefImage() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL;
  ISubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetFwdRefImage() const { return getval<uint32_t>(3); }
uint32_t GetBwdRefImage() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL;
  ISubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetRefImage() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
uint32_t GetStreaminComponents() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL;
  ISubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetFwdRefImage() const { return getval<uint32_t>(3); }
uint32_t GetBwdRefImage() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
uint32_t GetStreaminComponents() const { return getval<uint32_t>(6); }
};
struct ISubgroupAvcImeConvertToMceResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeConvertToMceResultINTEL;
  ISubgroupAvcImeConvertToMceResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeGetSingleReferenceStreaminINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetSingleReferenceStreaminINTEL;
  ISubgroupAvcImeGetSingleReferenceStreaminINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeGetDualReferenceStreaminINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetDualReferenceStreaminINTEL;
  ISubgroupAvcImeGetDualReferenceStreaminINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeStripSingleReferenceStreamoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeStripSingleReferenceStreamoutINTEL;
  ISubgroupAvcImeStripSingleReferenceStreamoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeStripDualReferenceStreamoutINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeStripDualReferenceStreamoutINTEL;
  ISubgroupAvcImeStripDualReferenceStreamoutINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL;
  ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
uint32_t GetMajorShape() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL;
  ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
uint32_t GetMajorShape() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL;
  ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
uint32_t GetMajorShape() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL;
  ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
uint32_t GetMajorShape() const { return getval<uint32_t>(3); }
uint32_t GetDirection() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL;
  ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
uint32_t GetMajorShape() const { return getval<uint32_t>(3); }
uint32_t GetDirection() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL;
  ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
uint32_t GetMajorShape() const { return getval<uint32_t>(3); }
uint32_t GetDirection() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcImeGetBorderReachedINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetBorderReachedINTEL;
  ISubgroupAvcImeGetBorderReachedINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetImageSelect() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcImeGetTruncatedSearchIndicationINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetTruncatedSearchIndicationINTEL;
  ISubgroupAvcImeGetTruncatedSearchIndicationINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL;
  ISubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL;
  ISubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL;
  ISubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcFmeInitializeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcFmeInitializeINTEL;
  ISubgroupAvcFmeInitializeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcCoord() const { return getval<uint32_t>(2); }
uint32_t GetMotionVectors() const { return getval<uint32_t>(3); }
uint32_t GetMajorShapes() const { return getval<uint32_t>(4); }
uint32_t GetMinorShapes() const { return getval<uint32_t>(5); }
uint32_t GetDirection() const { return getval<uint32_t>(6); }
uint32_t GetPixelResolution() const { return getval<uint32_t>(7); }
uint32_t GetSadAdjustment() const { return getval<uint32_t>(8); }
};
struct ISubgroupAvcBmeInitializeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcBmeInitializeINTEL;
  ISubgroupAvcBmeInitializeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcCoord() const { return getval<uint32_t>(2); }
uint32_t GetMotionVectors() const { return getval<uint32_t>(3); }
uint32_t GetMajorShapes() const { return getval<uint32_t>(4); }
uint32_t GetMinorShapes() const { return getval<uint32_t>(5); }
uint32_t GetDirection() const { return getval<uint32_t>(6); }
uint32_t GetPixelResolution() const { return getval<uint32_t>(7); }
uint32_t GetBidirectionalWeight() const { return getval<uint32_t>(8); }
uint32_t GetSadAdjustment() const { return getval<uint32_t>(9); }
};
struct ISubgroupAvcRefConvertToMcePayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefConvertToMcePayloadINTEL;
  ISubgroupAvcRefConvertToMcePayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcRefSetBidirectionalMixDisableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefSetBidirectionalMixDisableINTEL;
  ISubgroupAvcRefSetBidirectionalMixDisableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcRefSetBilinearFilterEnableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefSetBilinearFilterEnableINTEL;
  ISubgroupAvcRefSetBilinearFilterEnableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcRefEvaluateWithSingleReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefEvaluateWithSingleReferenceINTEL;
  ISubgroupAvcRefEvaluateWithSingleReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetRefImage() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcRefEvaluateWithDualReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefEvaluateWithDualReferenceINTEL;
  ISubgroupAvcRefEvaluateWithDualReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetFwdRefImage() const { return getval<uint32_t>(3); }
uint32_t GetBwdRefImage() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcRefEvaluateWithMultiReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefEvaluateWithMultiReferenceINTEL;
  ISubgroupAvcRefEvaluateWithMultiReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetPackedReferenceIds() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL;
  ISubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetPackedReferenceIds() const { return getval<uint32_t>(3); }
uint32_t GetPackedReferenceFieldPolarities() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcRefConvertToMceResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcRefConvertToMceResultINTEL;
  ISubgroupAvcRefConvertToMceResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicInitializeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicInitializeINTEL;
  ISubgroupAvcSicInitializeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcCoord() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicConfigureSkcINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicConfigureSkcINTEL;
  ISubgroupAvcSicConfigureSkcINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSkipBlockPartitionType() const { return getval<uint32_t>(2); }
uint32_t GetSkipMotionVectorMask() const { return getval<uint32_t>(3); }
uint32_t GetMotionVectors() const { return getval<uint32_t>(4); }
uint32_t GetBidirectionalWeight() const { return getval<uint32_t>(5); }
uint32_t GetSadAdjustment() const { return getval<uint32_t>(6); }
uint32_t GetPayload() const { return getval<uint32_t>(7); }
};
struct ISubgroupAvcSicConfigureIpeLumaINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicConfigureIpeLumaINTEL;
  ISubgroupAvcSicConfigureIpeLumaINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetLumaIntraPartitionMask() const { return getval<uint32_t>(2); }
uint32_t GetIntraNeighbourAvailabilty() const { return getval<uint32_t>(3); }
uint32_t GetLeftEdgeLumaPixels() const { return getval<uint32_t>(4); }
uint32_t GetUpperLeftCornerLumaPixel() const { return getval<uint32_t>(5); }
uint32_t GetUpperEdgeLumaPixels() const { return getval<uint32_t>(6); }
uint32_t GetUpperRightEdgeLumaPixels() const { return getval<uint32_t>(7); }
uint32_t GetSadAdjustment() const { return getval<uint32_t>(8); }
uint32_t GetPayload() const { return getval<uint32_t>(9); }
};
struct ISubgroupAvcSicConfigureIpeLumaChromaINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicConfigureIpeLumaChromaINTEL;
  ISubgroupAvcSicConfigureIpeLumaChromaINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetLumaIntraPartitionMask() const { return getval<uint32_t>(2); }
uint32_t GetIntraNeighbourAvailabilty() const { return getval<uint32_t>(3); }
uint32_t GetLeftEdgeLumaPixels() const { return getval<uint32_t>(4); }
uint32_t GetUpperLeftCornerLumaPixel() const { return getval<uint32_t>(5); }
uint32_t GetUpperEdgeLumaPixels() const { return getval<uint32_t>(6); }
uint32_t GetUpperRightEdgeLumaPixels() const { return getval<uint32_t>(7); }
uint32_t GetLeftEdgeChromaPixels() const { return getval<uint32_t>(8); }
uint32_t GetUpperLeftCornerChromaPixel() const { return getval<uint32_t>(9); }
uint32_t GetUpperEdgeChromaPixels() const { return getval<uint32_t>(10); }
uint32_t GetSadAdjustment() const { return getval<uint32_t>(11); }
uint32_t GetPayload() const { return getval<uint32_t>(12); }
};
struct ISubgroupAvcSicGetMotionVectorMaskINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetMotionVectorMaskINTEL;
  ISubgroupAvcSicGetMotionVectorMaskINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSkipBlockPartitionType() const { return getval<uint32_t>(2); }
uint32_t GetDirection() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcSicConvertToMcePayloadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicConvertToMcePayloadINTEL;
  ISubgroupAvcSicConvertToMcePayloadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicSetIntraLumaShapePenaltyINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL;
  ISubgroupAvcSicSetIntraLumaShapePenaltyINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPackedShapePenalty() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcSicSetIntraLumaModeCostFunctionINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL;
  ISubgroupAvcSicSetIntraLumaModeCostFunctionINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetLumaModePenalty() const { return getval<uint32_t>(2); }
uint32_t GetLumaPackedNeighborModes() const { return getval<uint32_t>(3); }
uint32_t GetLumaPackedNonDcPenalty() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcSicSetIntraChromaModeCostFunctionINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL;
  ISubgroupAvcSicSetIntraChromaModeCostFunctionINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetChromaModeBasePenalty() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcSicSetBilinearFilterEnableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicSetBilinearFilterEnableINTEL;
  ISubgroupAvcSicSetBilinearFilterEnableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicSetSkcForwardTransformEnableINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicSetSkcForwardTransformEnableINTEL;
  ISubgroupAvcSicSetSkcForwardTransformEnableINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPackedSadCoefficients() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcSicSetBlockBasedRawSkipSadINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL;
  ISubgroupAvcSicSetBlockBasedRawSkipSadINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetBlockBasedSkipType() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcSicEvaluateIpeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicEvaluateIpeINTEL;
  ISubgroupAvcSicEvaluateIpeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetPayload() const { return getval<uint32_t>(3); }
};
struct ISubgroupAvcSicEvaluateWithSingleReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicEvaluateWithSingleReferenceINTEL;
  ISubgroupAvcSicEvaluateWithSingleReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetRefImage() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcSicEvaluateWithDualReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicEvaluateWithDualReferenceINTEL;
  ISubgroupAvcSicEvaluateWithDualReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetFwdRefImage() const { return getval<uint32_t>(3); }
uint32_t GetBwdRefImage() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcSicEvaluateWithMultiReferenceINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicEvaluateWithMultiReferenceINTEL;
  ISubgroupAvcSicEvaluateWithMultiReferenceINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetPackedReferenceIds() const { return getval<uint32_t>(3); }
uint32_t GetPayload() const { return getval<uint32_t>(4); }
};
struct ISubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL;
  ISubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetSrcImage() const { return getval<uint32_t>(2); }
uint32_t GetPackedReferenceIds() const { return getval<uint32_t>(3); }
uint32_t GetPackedReferenceFieldPolarities() const { return getval<uint32_t>(4); }
uint32_t GetPayload() const { return getval<uint32_t>(5); }
};
struct ISubgroupAvcSicConvertToMceResultINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicConvertToMceResultINTEL;
  ISubgroupAvcSicConvertToMceResultINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetIpeLumaShapeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetIpeLumaShapeINTEL;
  ISubgroupAvcSicGetIpeLumaShapeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetBestIpeLumaDistortionINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetBestIpeLumaDistortionINTEL;
  ISubgroupAvcSicGetBestIpeLumaDistortionINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetBestIpeChromaDistortionINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetBestIpeChromaDistortionINTEL;
  ISubgroupAvcSicGetBestIpeChromaDistortionINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetPackedIpeLumaModesINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetPackedIpeLumaModesINTEL;
  ISubgroupAvcSicGetPackedIpeLumaModesINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetIpeChromaModeINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetIpeChromaModeINTEL;
  ISubgroupAvcSicGetIpeChromaModeINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL;
  ISubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL;
  ISubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};
struct ISubgroupAvcSicGetInterRawSadsINTEL : public Instruction {
  static constexpr SpvOp Opcode = SpvOpSubgroupAvcSicGetInterRawSadsINTEL;
  ISubgroupAvcSicGetInterRawSadsINTEL(const spv_parsed_instruction_t *i) : Instruction(i) {}
  uint32_t GetIdResultType() const { return getval<uint32_t>(0); }
uint32_t GetIdResult() const { return getval<uint32_t>(1); }
uint32_t GetPayload() const { return getval<uint32_t>(2); }
};

using IDecorateStringGOOGLE = IDecorateString;
using IMemberDecorateStringGOOGLE = IMemberDecorateString;

template<typename... Args>
struct Dispatch {
  virtual ~Dispatch() {}
  virtual spv_result_t do_default(const Instruction&, Args...) { return SPV_SUCCESS; }
  virtual spv_result_t do_missing(const Instruction&, Args...) { return SPV_UNSUPPORTED; }

  virtual spv_result_t do_Nop(const INop &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Undef(const IUndef &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SourceContinued(const ISourceContinued &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Source(const ISource &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SourceExtension(const ISourceExtension &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Name(const IName &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MemberName(const IMemberName &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_String(const IString &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Line(const ILine &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Extension(const IExtension &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ExtInstImport(const IExtInstImport &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ExtInst(const IExtInst &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MemoryModel(const IMemoryModel &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EntryPoint(const IEntryPoint &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ExecutionMode(const IExecutionMode &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Capability(const ICapability &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeVoid(const ITypeVoid &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeBool(const ITypeBool &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeInt(const ITypeInt &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeFloat(const ITypeFloat &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeVector(const ITypeVector &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeMatrix(const ITypeMatrix &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeImage(const ITypeImage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeSampler(const ITypeSampler &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeSampledImage(const ITypeSampledImage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeArray(const ITypeArray &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeRuntimeArray(const ITypeRuntimeArray &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeStruct(const ITypeStruct &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeOpaque(const ITypeOpaque &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypePointer(const ITypePointer &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeFunction(const ITypeFunction &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeEvent(const ITypeEvent &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeDeviceEvent(const ITypeDeviceEvent &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeReserveId(const ITypeReserveId &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeQueue(const ITypeQueue &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypePipe(const ITypePipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeForwardPointer(const ITypeForwardPointer &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConstantTrue(const IConstantTrue &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConstantFalse(const IConstantFalse &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Constant(const IConstant &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConstantComposite(const IConstantComposite &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConstantSampler(const IConstantSampler &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConstantNull(const IConstantNull &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SpecConstantTrue(const ISpecConstantTrue &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SpecConstantFalse(const ISpecConstantFalse &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SpecConstant(const ISpecConstant &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SpecConstantComposite(const ISpecConstantComposite &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SpecConstantOp(const ISpecConstantOp &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Function(const IFunction &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FunctionParameter(const IFunctionParameter &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FunctionEnd(const IFunctionEnd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FunctionCall(const IFunctionCall &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Variable(const IVariable &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageTexelPointer(const IImageTexelPointer &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Load(const ILoad &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Store(const IStore &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CopyMemory(const ICopyMemory &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CopyMemorySized(const ICopyMemorySized &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AccessChain(const IAccessChain &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_InBoundsAccessChain(const IInBoundsAccessChain &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_PtrAccessChain(const IPtrAccessChain &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ArrayLength(const IArrayLength &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GenericPtrMemSemantics(const IGenericPtrMemSemantics &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_InBoundsPtrAccessChain(const IInBoundsPtrAccessChain &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Decorate(const IDecorate &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MemberDecorate(const IMemberDecorate &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DecorationGroup(const IDecorationGroup &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupDecorate(const IGroupDecorate &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupMemberDecorate(const IGroupMemberDecorate &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_VectorExtractDynamic(const IVectorExtractDynamic &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_VectorInsertDynamic(const IVectorInsertDynamic &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_VectorShuffle(const IVectorShuffle &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CompositeConstruct(const ICompositeConstruct &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CompositeExtract(const ICompositeExtract &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CompositeInsert(const ICompositeInsert &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CopyObject(const ICopyObject &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Transpose(const ITranspose &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SampledImage(const ISampledImage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleImplicitLod(const IImageSampleImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleExplicitLod(const IImageSampleExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleDrefImplicitLod(const IImageSampleDrefImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleDrefExplicitLod(const IImageSampleDrefExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleProjImplicitLod(const IImageSampleProjImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleProjExplicitLod(const IImageSampleProjExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleProjDrefImplicitLod(const IImageSampleProjDrefImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleProjDrefExplicitLod(const IImageSampleProjDrefExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageFetch(const IImageFetch &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageGather(const IImageGather &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageDrefGather(const IImageDrefGather &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageRead(const IImageRead &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageWrite(const IImageWrite &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Image(const IImage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQueryFormat(const IImageQueryFormat &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQueryOrder(const IImageQueryOrder &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQuerySizeLod(const IImageQuerySizeLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQuerySize(const IImageQuerySize &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQueryLod(const IImageQueryLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQueryLevels(const IImageQueryLevels &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageQuerySamples(const IImageQuerySamples &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConvertFToU(const IConvertFToU &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConvertFToS(const IConvertFToS &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConvertSToF(const IConvertSToF &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConvertUToF(const IConvertUToF &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_UConvert(const IUConvert &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SConvert(const ISConvert &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FConvert(const IFConvert &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_QuantizeToF16(const IQuantizeToF16 &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConvertPtrToU(const IConvertPtrToU &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SatConvertSToU(const ISatConvertSToU &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SatConvertUToS(const ISatConvertUToS &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConvertUToPtr(const IConvertUToPtr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_PtrCastToGeneric(const IPtrCastToGeneric &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GenericCastToPtr(const IGenericCastToPtr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GenericCastToPtrExplicit(const IGenericCastToPtrExplicit &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Bitcast(const IBitcast &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SNegate(const ISNegate &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FNegate(const IFNegate &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IAdd(const IIAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FAdd(const IFAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ISub(const IISub &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FSub(const IFSub &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IMul(const IIMul &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FMul(const IFMul &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_UDiv(const IUDiv &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SDiv(const ISDiv &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FDiv(const IFDiv &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_UMod(const IUMod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SRem(const ISRem &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SMod(const ISMod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FRem(const IFRem &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FMod(const IFMod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_VectorTimesScalar(const IVectorTimesScalar &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MatrixTimesScalar(const IMatrixTimesScalar &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_VectorTimesMatrix(const IVectorTimesMatrix &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MatrixTimesVector(const IMatrixTimesVector &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MatrixTimesMatrix(const IMatrixTimesMatrix &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_OuterProduct(const IOuterProduct &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Dot(const IDot &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IAddCarry(const IIAddCarry &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ISubBorrow(const IISubBorrow &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_UMulExtended(const IUMulExtended &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SMulExtended(const ISMulExtended &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Any(const IAny &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_All(const IAll &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IsNan(const IIsNan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IsInf(const IIsInf &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IsFinite(const IIsFinite &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IsNormal(const IIsNormal &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SignBitSet(const ISignBitSet &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LessOrGreater(const ILessOrGreater &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Ordered(const IOrdered &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Unordered(const IUnordered &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LogicalEqual(const ILogicalEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LogicalNotEqual(const ILogicalNotEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LogicalOr(const ILogicalOr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LogicalAnd(const ILogicalAnd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LogicalNot(const ILogicalNot &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Select(const ISelect &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IEqual(const IIEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_INotEqual(const IINotEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_UGreaterThan(const IUGreaterThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SGreaterThan(const ISGreaterThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_UGreaterThanEqual(const IUGreaterThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SGreaterThanEqual(const ISGreaterThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ULessThan(const IULessThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SLessThan(const ISLessThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ULessThanEqual(const IULessThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SLessThanEqual(const ISLessThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FOrdEqual(const IFOrdEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FUnordEqual(const IFUnordEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FOrdNotEqual(const IFOrdNotEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FUnordNotEqual(const IFUnordNotEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FOrdLessThan(const IFOrdLessThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FUnordLessThan(const IFUnordLessThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FOrdGreaterThan(const IFOrdGreaterThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FUnordGreaterThan(const IFUnordGreaterThan &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FOrdLessThanEqual(const IFOrdLessThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FUnordLessThanEqual(const IFUnordLessThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FOrdGreaterThanEqual(const IFOrdGreaterThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FUnordGreaterThanEqual(const IFUnordGreaterThanEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ShiftRightLogical(const IShiftRightLogical &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ShiftRightArithmetic(const IShiftRightArithmetic &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ShiftLeftLogical(const IShiftLeftLogical &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitwiseOr(const IBitwiseOr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitwiseXor(const IBitwiseXor &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitwiseAnd(const IBitwiseAnd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Not(const INot &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitFieldInsert(const IBitFieldInsert &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitFieldSExtract(const IBitFieldSExtract &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitFieldUExtract(const IBitFieldUExtract &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitReverse(const IBitReverse &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BitCount(const IBitCount &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DPdx(const IDPdx &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DPdy(const IDPdy &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Fwidth(const IFwidth &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DPdxFine(const IDPdxFine &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DPdyFine(const IDPdyFine &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FwidthFine(const IFwidthFine &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DPdxCoarse(const IDPdxCoarse &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DPdyCoarse(const IDPdyCoarse &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FwidthCoarse(const IFwidthCoarse &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EmitVertex(const IEmitVertex &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EndPrimitive(const IEndPrimitive &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EmitStreamVertex(const IEmitStreamVertex &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EndStreamPrimitive(const IEndStreamPrimitive &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ControlBarrier(const IControlBarrier &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MemoryBarrier(const IMemoryBarrier &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicLoad(const IAtomicLoad &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicStore(const IAtomicStore &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicExchange(const IAtomicExchange &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicCompareExchange(const IAtomicCompareExchange &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicCompareExchangeWeak(const IAtomicCompareExchangeWeak &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicIIncrement(const IAtomicIIncrement &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicIDecrement(const IAtomicIDecrement &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicIAdd(const IAtomicIAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicISub(const IAtomicISub &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicSMin(const IAtomicSMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicUMin(const IAtomicUMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicSMax(const IAtomicSMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicUMax(const IAtomicUMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicAnd(const IAtomicAnd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicOr(const IAtomicOr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicXor(const IAtomicXor &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Phi(const IPhi &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LoopMerge(const ILoopMerge &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SelectionMerge(const ISelectionMerge &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Label(const ILabel &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Branch(const IBranch &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BranchConditional(const IBranchConditional &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Switch(const ISwitch &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Kill(const IKill &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Return(const IReturn &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReturnValue(const IReturnValue &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_Unreachable(const IUnreachable &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LifetimeStart(const ILifetimeStart &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_LifetimeStop(const ILifetimeStop &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupAsyncCopy(const IGroupAsyncCopy &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupWaitEvents(const IGroupWaitEvents &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupAll(const IGroupAll &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupAny(const IGroupAny &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupBroadcast(const IGroupBroadcast &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupIAdd(const IGroupIAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupFAdd(const IGroupFAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupFMin(const IGroupFMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupUMin(const IGroupUMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupSMin(const IGroupSMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupFMax(const IGroupFMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupUMax(const IGroupUMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupSMax(const IGroupSMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReadPipe(const IReadPipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_WritePipe(const IWritePipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReservedReadPipe(const IReservedReadPipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReservedWritePipe(const IReservedWritePipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReserveReadPipePackets(const IReserveReadPipePackets &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReserveWritePipePackets(const IReserveWritePipePackets &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CommitReadPipe(const ICommitReadPipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CommitWritePipe(const ICommitWritePipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IsValidReserveId(const IIsValidReserveId &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetNumPipePackets(const IGetNumPipePackets &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetMaxPipePackets(const IGetMaxPipePackets &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupReserveReadPipePackets(const IGroupReserveReadPipePackets &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupReserveWritePipePackets(const IGroupReserveWritePipePackets &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupCommitReadPipe(const IGroupCommitReadPipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupCommitWritePipe(const IGroupCommitWritePipe &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EnqueueMarker(const IEnqueueMarker &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_EnqueueKernel(const IEnqueueKernel &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetKernelNDrangeSubGroupCount(const IGetKernelNDrangeSubGroupCount &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetKernelNDrangeMaxSubGroupSize(const IGetKernelNDrangeMaxSubGroupSize &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetKernelWorkGroupSize(const IGetKernelWorkGroupSize &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetKernelPreferredWorkGroupSizeMultiple(const IGetKernelPreferredWorkGroupSizeMultiple &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_RetainEvent(const IRetainEvent &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReleaseEvent(const IReleaseEvent &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CreateUserEvent(const ICreateUserEvent &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IsValidEvent(const IIsValidEvent &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SetUserEventStatus(const ISetUserEventStatus &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CaptureEventProfilingInfo(const ICaptureEventProfilingInfo &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetDefaultQueue(const IGetDefaultQueue &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_BuildNDRange(const IBuildNDRange &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleImplicitLod(const IImageSparseSampleImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleExplicitLod(const IImageSparseSampleExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleDrefImplicitLod(const IImageSparseSampleDrefImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleDrefExplicitLod(const IImageSparseSampleDrefExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleProjImplicitLod(const IImageSparseSampleProjImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleProjExplicitLod(const IImageSparseSampleProjExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleProjDrefImplicitLod(const IImageSparseSampleProjDrefImplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseSampleProjDrefExplicitLod(const IImageSparseSampleProjDrefExplicitLod &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseFetch(const IImageSparseFetch &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseGather(const IImageSparseGather &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseDrefGather(const IImageSparseDrefGather &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseTexelsResident(const IImageSparseTexelsResident &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_NoLine(const INoLine &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicFlagTestAndSet(const IAtomicFlagTestAndSet &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_AtomicFlagClear(const IAtomicFlagClear &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSparseRead(const IImageSparseRead &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SizeOf(const ISizeOf &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypePipeStorage(const ITypePipeStorage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ConstantPipeStorage(const IConstantPipeStorage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CreatePipeFromPipeStorage(const ICreatePipeFromPipeStorage &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetKernelLocalSizeForSubgroupCount(const IGetKernelLocalSizeForSubgroupCount &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GetKernelMaxNumSubgroups(const IGetKernelMaxNumSubgroups &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeNamedBarrier(const ITypeNamedBarrier &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_NamedBarrierInitialize(const INamedBarrierInitialize &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MemoryNamedBarrier(const IMemoryNamedBarrier &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ModuleProcessed(const IModuleProcessed &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ExecutionModeId(const IExecutionModeId &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DecorateId(const IDecorateId &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformElect(const IGroupNonUniformElect &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformAll(const IGroupNonUniformAll &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformAny(const IGroupNonUniformAny &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformAllEqual(const IGroupNonUniformAllEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBroadcast(const IGroupNonUniformBroadcast &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBroadcastFirst(const IGroupNonUniformBroadcastFirst &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBallot(const IGroupNonUniformBallot &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformInverseBallot(const IGroupNonUniformInverseBallot &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBallotBitExtract(const IGroupNonUniformBallotBitExtract &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBallotBitCount(const IGroupNonUniformBallotBitCount &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBallotFindLSB(const IGroupNonUniformBallotFindLSB &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBallotFindMSB(const IGroupNonUniformBallotFindMSB &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformShuffle(const IGroupNonUniformShuffle &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformShuffleXor(const IGroupNonUniformShuffleXor &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformShuffleUp(const IGroupNonUniformShuffleUp &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformShuffleDown(const IGroupNonUniformShuffleDown &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformIAdd(const IGroupNonUniformIAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformFAdd(const IGroupNonUniformFAdd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformIMul(const IGroupNonUniformIMul &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformFMul(const IGroupNonUniformFMul &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformSMin(const IGroupNonUniformSMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformUMin(const IGroupNonUniformUMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformFMin(const IGroupNonUniformFMin &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformSMax(const IGroupNonUniformSMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformUMax(const IGroupNonUniformUMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformFMax(const IGroupNonUniformFMax &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBitwiseAnd(const IGroupNonUniformBitwiseAnd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBitwiseOr(const IGroupNonUniformBitwiseOr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformBitwiseXor(const IGroupNonUniformBitwiseXor &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformLogicalAnd(const IGroupNonUniformLogicalAnd &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformLogicalOr(const IGroupNonUniformLogicalOr &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformLogicalXor(const IGroupNonUniformLogicalXor &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformQuadBroadcast(const IGroupNonUniformQuadBroadcast &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformQuadSwap(const IGroupNonUniformQuadSwap &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CopyLogical(const ICopyLogical &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_PtrEqual(const IPtrEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_PtrNotEqual(const IPtrNotEqual &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_PtrDiff(const IPtrDiff &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupBallotKHR(const ISubgroupBallotKHR &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupFirstInvocationKHR(const ISubgroupFirstInvocationKHR &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAllKHR(const ISubgroupAllKHR &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAnyKHR(const ISubgroupAnyKHR &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAllEqualKHR(const ISubgroupAllEqualKHR &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupReadInvocationKHR(const ISubgroupReadInvocationKHR &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupIAddNonUniformAMD(const IGroupIAddNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupFAddNonUniformAMD(const IGroupFAddNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupFMinNonUniformAMD(const IGroupFMinNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupUMinNonUniformAMD(const IGroupUMinNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupSMinNonUniformAMD(const IGroupSMinNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupFMaxNonUniformAMD(const IGroupFMaxNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupUMaxNonUniformAMD(const IGroupUMaxNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupSMaxNonUniformAMD(const IGroupSMaxNonUniformAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FragmentMaskFetchAMD(const IFragmentMaskFetchAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_FragmentFetchAMD(const IFragmentFetchAMD &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ImageSampleFootprintNV(const IImageSampleFootprintNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_GroupNonUniformPartitionNV(const IGroupNonUniformPartitionNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_WritePackedPrimitiveIndices4x8NV(const IWritePackedPrimitiveIndices4x8NV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ReportIntersectionNV(const IReportIntersectionNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_IgnoreIntersectionNV(const IIgnoreIntersectionNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TerminateRayNV(const ITerminateRayNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TraceNV(const ITraceNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAccelerationStructureNV(const ITypeAccelerationStructureNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_ExecuteCallableNV(const IExecuteCallableNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeCooperativeMatrixNV(const ITypeCooperativeMatrixNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CooperativeMatrixLoadNV(const ICooperativeMatrixLoadNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CooperativeMatrixStoreNV(const ICooperativeMatrixStoreNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CooperativeMatrixMulAddNV(const ICooperativeMatrixMulAddNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_CooperativeMatrixLengthNV(const ICooperativeMatrixLengthNV &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupShuffleINTEL(const ISubgroupShuffleINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupShuffleDownINTEL(const ISubgroupShuffleDownINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupShuffleUpINTEL(const ISubgroupShuffleUpINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupShuffleXorINTEL(const ISubgroupShuffleXorINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupBlockReadINTEL(const ISubgroupBlockReadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupBlockWriteINTEL(const ISubgroupBlockWriteINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupImageBlockReadINTEL(const ISubgroupImageBlockReadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupImageBlockWriteINTEL(const ISubgroupImageBlockWriteINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupImageMediaBlockReadINTEL(const ISubgroupImageMediaBlockReadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupImageMediaBlockWriteINTEL(const ISubgroupImageMediaBlockWriteINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_DecorateString(const IDecorateString &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_MemberDecorateString(const IMemberDecorateString &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_VmeImageINTEL(const IVmeImageINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeVmeImageINTEL(const ITypeVmeImageINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcImePayloadINTEL(const ITypeAvcImePayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcRefPayloadINTEL(const ITypeAvcRefPayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcSicPayloadINTEL(const ITypeAvcSicPayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcMcePayloadINTEL(const ITypeAvcMcePayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcMceResultINTEL(const ITypeAvcMceResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcImeResultINTEL(const ITypeAvcImeResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcImeResultSingleReferenceStreamoutINTEL(const ITypeAvcImeResultSingleReferenceStreamoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcImeResultDualReferenceStreamoutINTEL(const ITypeAvcImeResultDualReferenceStreamoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcImeSingleReferenceStreaminINTEL(const ITypeAvcImeSingleReferenceStreaminINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcImeDualReferenceStreaminINTEL(const ITypeAvcImeDualReferenceStreaminINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcRefResultINTEL(const ITypeAvcRefResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_TypeAvcSicResultINTEL(const ITypeAvcSicResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL(const ISubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL(const ISubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultInterShapePenaltyINTEL(const ISubgroupAvcMceGetDefaultInterShapePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetInterShapePenaltyINTEL(const ISubgroupAvcMceSetInterShapePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL(const ISubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetInterDirectionPenaltyINTEL(const ISubgroupAvcMceSetInterDirectionPenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL(const ISubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL(const ISubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL(const ISubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL(const ISubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL(const ISubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetMotionVectorCostFunctionINTEL(const ISubgroupAvcMceSetMotionVectorCostFunctionINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL(const ISubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL(const ISubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL(const ISubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetAcOnlyHaarINTEL(const ISubgroupAvcMceSetAcOnlyHaarINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL(const ISubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL(const ISubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL(const ISubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceConvertToImePayloadINTEL(const ISubgroupAvcMceConvertToImePayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceConvertToImeResultINTEL(const ISubgroupAvcMceConvertToImeResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceConvertToRefPayloadINTEL(const ISubgroupAvcMceConvertToRefPayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceConvertToRefResultINTEL(const ISubgroupAvcMceConvertToRefResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceConvertToSicPayloadINTEL(const ISubgroupAvcMceConvertToSicPayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceConvertToSicResultINTEL(const ISubgroupAvcMceConvertToSicResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetMotionVectorsINTEL(const ISubgroupAvcMceGetMotionVectorsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterDistortionsINTEL(const ISubgroupAvcMceGetInterDistortionsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetBestInterDistortionsINTEL(const ISubgroupAvcMceGetBestInterDistortionsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterMajorShapeINTEL(const ISubgroupAvcMceGetInterMajorShapeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterMinorShapeINTEL(const ISubgroupAvcMceGetInterMinorShapeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterDirectionsINTEL(const ISubgroupAvcMceGetInterDirectionsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterMotionVectorCountINTEL(const ISubgroupAvcMceGetInterMotionVectorCountINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterReferenceIdsINTEL(const ISubgroupAvcMceGetInterReferenceIdsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL(const ISubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeInitializeINTEL(const ISubgroupAvcImeInitializeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeSetSingleReferenceINTEL(const ISubgroupAvcImeSetSingleReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeSetDualReferenceINTEL(const ISubgroupAvcImeSetDualReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeRefWindowSizeINTEL(const ISubgroupAvcImeRefWindowSizeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeAdjustRefOffsetINTEL(const ISubgroupAvcImeAdjustRefOffsetINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeConvertToMcePayloadINTEL(const ISubgroupAvcImeConvertToMcePayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeSetMaxMotionVectorCountINTEL(const ISubgroupAvcImeSetMaxMotionVectorCountINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeSetUnidirectionalMixDisableINTEL(const ISubgroupAvcImeSetUnidirectionalMixDisableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeSetEarlySearchTerminationThresholdINTEL(const ISubgroupAvcImeSetEarlySearchTerminationThresholdINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeSetWeightedSadINTEL(const ISubgroupAvcImeSetWeightedSadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithSingleReferenceINTEL(const ISubgroupAvcImeEvaluateWithSingleReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithDualReferenceINTEL(const ISubgroupAvcImeEvaluateWithDualReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL(const ISubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL(const ISubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL(const ISubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL(const ISubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL(const ISubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL(const ISubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeConvertToMceResultINTEL(const ISubgroupAvcImeConvertToMceResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetSingleReferenceStreaminINTEL(const ISubgroupAvcImeGetSingleReferenceStreaminINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetDualReferenceStreaminINTEL(const ISubgroupAvcImeGetDualReferenceStreaminINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeStripSingleReferenceStreamoutINTEL(const ISubgroupAvcImeStripSingleReferenceStreamoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeStripDualReferenceStreamoutINTEL(const ISubgroupAvcImeStripDualReferenceStreamoutINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL(const ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL(const ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL(const ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL(const ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL(const ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL(const ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetBorderReachedINTEL(const ISubgroupAvcImeGetBorderReachedINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetTruncatedSearchIndicationINTEL(const ISubgroupAvcImeGetTruncatedSearchIndicationINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL(const ISubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL(const ISubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL(const ISubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcFmeInitializeINTEL(const ISubgroupAvcFmeInitializeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcBmeInitializeINTEL(const ISubgroupAvcBmeInitializeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefConvertToMcePayloadINTEL(const ISubgroupAvcRefConvertToMcePayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefSetBidirectionalMixDisableINTEL(const ISubgroupAvcRefSetBidirectionalMixDisableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefSetBilinearFilterEnableINTEL(const ISubgroupAvcRefSetBilinearFilterEnableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefEvaluateWithSingleReferenceINTEL(const ISubgroupAvcRefEvaluateWithSingleReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefEvaluateWithDualReferenceINTEL(const ISubgroupAvcRefEvaluateWithDualReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefEvaluateWithMultiReferenceINTEL(const ISubgroupAvcRefEvaluateWithMultiReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL(const ISubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcRefConvertToMceResultINTEL(const ISubgroupAvcRefConvertToMceResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicInitializeINTEL(const ISubgroupAvcSicInitializeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicConfigureSkcINTEL(const ISubgroupAvcSicConfigureSkcINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicConfigureIpeLumaINTEL(const ISubgroupAvcSicConfigureIpeLumaINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicConfigureIpeLumaChromaINTEL(const ISubgroupAvcSicConfigureIpeLumaChromaINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetMotionVectorMaskINTEL(const ISubgroupAvcSicGetMotionVectorMaskINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicConvertToMcePayloadINTEL(const ISubgroupAvcSicConvertToMcePayloadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicSetIntraLumaShapePenaltyINTEL(const ISubgroupAvcSicSetIntraLumaShapePenaltyINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicSetIntraLumaModeCostFunctionINTEL(const ISubgroupAvcSicSetIntraLumaModeCostFunctionINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicSetIntraChromaModeCostFunctionINTEL(const ISubgroupAvcSicSetIntraChromaModeCostFunctionINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicSetBilinearFilterEnableINTEL(const ISubgroupAvcSicSetBilinearFilterEnableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicSetSkcForwardTransformEnableINTEL(const ISubgroupAvcSicSetSkcForwardTransformEnableINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicSetBlockBasedRawSkipSadINTEL(const ISubgroupAvcSicSetBlockBasedRawSkipSadINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicEvaluateIpeINTEL(const ISubgroupAvcSicEvaluateIpeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicEvaluateWithSingleReferenceINTEL(const ISubgroupAvcSicEvaluateWithSingleReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicEvaluateWithDualReferenceINTEL(const ISubgroupAvcSicEvaluateWithDualReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicEvaluateWithMultiReferenceINTEL(const ISubgroupAvcSicEvaluateWithMultiReferenceINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL(const ISubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicConvertToMceResultINTEL(const ISubgroupAvcSicConvertToMceResultINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetIpeLumaShapeINTEL(const ISubgroupAvcSicGetIpeLumaShapeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetBestIpeLumaDistortionINTEL(const ISubgroupAvcSicGetBestIpeLumaDistortionINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetBestIpeChromaDistortionINTEL(const ISubgroupAvcSicGetBestIpeChromaDistortionINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetPackedIpeLumaModesINTEL(const ISubgroupAvcSicGetPackedIpeLumaModesINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetIpeChromaModeINTEL(const ISubgroupAvcSicGetIpeChromaModeINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL(const ISubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL(const ISubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL &i, Args... args) { return do_default(i, args...); }
virtual spv_result_t do_SubgroupAvcSicGetInterRawSadsINTEL(const ISubgroupAvcSicGetInterRawSadsINTEL &i, Args... args) { return do_default(i, args...); }

  spv_result_t operator()(const Instruction *i, Args... args) {
    switch(i->Opcode()) {
      case SpvOpNop: return do_Nop(*i->Get<INop>(), args...);
case SpvOpUndef: return do_Undef(*i->Get<IUndef>(), args...);
case SpvOpSourceContinued: return do_SourceContinued(*i->Get<ISourceContinued>(), args...);
case SpvOpSource: return do_Source(*i->Get<ISource>(), args...);
case SpvOpSourceExtension: return do_SourceExtension(*i->Get<ISourceExtension>(), args...);
case SpvOpName: return do_Name(*i->Get<IName>(), args...);
case SpvOpMemberName: return do_MemberName(*i->Get<IMemberName>(), args...);
case SpvOpString: return do_String(*i->Get<IString>(), args...);
case SpvOpLine: return do_Line(*i->Get<ILine>(), args...);
case SpvOpExtension: return do_Extension(*i->Get<IExtension>(), args...);
case SpvOpExtInstImport: return do_ExtInstImport(*i->Get<IExtInstImport>(), args...);
case SpvOpExtInst: return do_ExtInst(*i->Get<IExtInst>(), args...);
case SpvOpMemoryModel: return do_MemoryModel(*i->Get<IMemoryModel>(), args...);
case SpvOpEntryPoint: return do_EntryPoint(*i->Get<IEntryPoint>(), args...);
case SpvOpExecutionMode: return do_ExecutionMode(*i->Get<IExecutionMode>(), args...);
case SpvOpCapability: return do_Capability(*i->Get<ICapability>(), args...);
case SpvOpTypeVoid: return do_TypeVoid(*i->Get<ITypeVoid>(), args...);
case SpvOpTypeBool: return do_TypeBool(*i->Get<ITypeBool>(), args...);
case SpvOpTypeInt: return do_TypeInt(*i->Get<ITypeInt>(), args...);
case SpvOpTypeFloat: return do_TypeFloat(*i->Get<ITypeFloat>(), args...);
case SpvOpTypeVector: return do_TypeVector(*i->Get<ITypeVector>(), args...);
case SpvOpTypeMatrix: return do_TypeMatrix(*i->Get<ITypeMatrix>(), args...);
case SpvOpTypeImage: return do_TypeImage(*i->Get<ITypeImage>(), args...);
case SpvOpTypeSampler: return do_TypeSampler(*i->Get<ITypeSampler>(), args...);
case SpvOpTypeSampledImage: return do_TypeSampledImage(*i->Get<ITypeSampledImage>(), args...);
case SpvOpTypeArray: return do_TypeArray(*i->Get<ITypeArray>(), args...);
case SpvOpTypeRuntimeArray: return do_TypeRuntimeArray(*i->Get<ITypeRuntimeArray>(), args...);
case SpvOpTypeStruct: return do_TypeStruct(*i->Get<ITypeStruct>(), args...);
case SpvOpTypeOpaque: return do_TypeOpaque(*i->Get<ITypeOpaque>(), args...);
case SpvOpTypePointer: return do_TypePointer(*i->Get<ITypePointer>(), args...);
case SpvOpTypeFunction: return do_TypeFunction(*i->Get<ITypeFunction>(), args...);
case SpvOpTypeEvent: return do_TypeEvent(*i->Get<ITypeEvent>(), args...);
case SpvOpTypeDeviceEvent: return do_TypeDeviceEvent(*i->Get<ITypeDeviceEvent>(), args...);
case SpvOpTypeReserveId: return do_TypeReserveId(*i->Get<ITypeReserveId>(), args...);
case SpvOpTypeQueue: return do_TypeQueue(*i->Get<ITypeQueue>(), args...);
case SpvOpTypePipe: return do_TypePipe(*i->Get<ITypePipe>(), args...);
case SpvOpTypeForwardPointer: return do_TypeForwardPointer(*i->Get<ITypeForwardPointer>(), args...);
case SpvOpConstantTrue: return do_ConstantTrue(*i->Get<IConstantTrue>(), args...);
case SpvOpConstantFalse: return do_ConstantFalse(*i->Get<IConstantFalse>(), args...);
case SpvOpConstant: return do_Constant(*i->Get<IConstant>(), args...);
case SpvOpConstantComposite: return do_ConstantComposite(*i->Get<IConstantComposite>(), args...);
case SpvOpConstantSampler: return do_ConstantSampler(*i->Get<IConstantSampler>(), args...);
case SpvOpConstantNull: return do_ConstantNull(*i->Get<IConstantNull>(), args...);
case SpvOpSpecConstantTrue: return do_SpecConstantTrue(*i->Get<ISpecConstantTrue>(), args...);
case SpvOpSpecConstantFalse: return do_SpecConstantFalse(*i->Get<ISpecConstantFalse>(), args...);
case SpvOpSpecConstant: return do_SpecConstant(*i->Get<ISpecConstant>(), args...);
case SpvOpSpecConstantComposite: return do_SpecConstantComposite(*i->Get<ISpecConstantComposite>(), args...);
case SpvOpSpecConstantOp: return do_SpecConstantOp(*i->Get<ISpecConstantOp>(), args...);
case SpvOpFunction: return do_Function(*i->Get<IFunction>(), args...);
case SpvOpFunctionParameter: return do_FunctionParameter(*i->Get<IFunctionParameter>(), args...);
case SpvOpFunctionEnd: return do_FunctionEnd(*i->Get<IFunctionEnd>(), args...);
case SpvOpFunctionCall: return do_FunctionCall(*i->Get<IFunctionCall>(), args...);
case SpvOpVariable: return do_Variable(*i->Get<IVariable>(), args...);
case SpvOpImageTexelPointer: return do_ImageTexelPointer(*i->Get<IImageTexelPointer>(), args...);
case SpvOpLoad: return do_Load(*i->Get<ILoad>(), args...);
case SpvOpStore: return do_Store(*i->Get<IStore>(), args...);
case SpvOpCopyMemory: return do_CopyMemory(*i->Get<ICopyMemory>(), args...);
case SpvOpCopyMemorySized: return do_CopyMemorySized(*i->Get<ICopyMemorySized>(), args...);
case SpvOpAccessChain: return do_AccessChain(*i->Get<IAccessChain>(), args...);
case SpvOpInBoundsAccessChain: return do_InBoundsAccessChain(*i->Get<IInBoundsAccessChain>(), args...);
case SpvOpPtrAccessChain: return do_PtrAccessChain(*i->Get<IPtrAccessChain>(), args...);
case SpvOpArrayLength: return do_ArrayLength(*i->Get<IArrayLength>(), args...);
case SpvOpGenericPtrMemSemantics: return do_GenericPtrMemSemantics(*i->Get<IGenericPtrMemSemantics>(), args...);
case SpvOpInBoundsPtrAccessChain: return do_InBoundsPtrAccessChain(*i->Get<IInBoundsPtrAccessChain>(), args...);
case SpvOpDecorate: return do_Decorate(*i->Get<IDecorate>(), args...);
case SpvOpMemberDecorate: return do_MemberDecorate(*i->Get<IMemberDecorate>(), args...);
case SpvOpDecorationGroup: return do_DecorationGroup(*i->Get<IDecorationGroup>(), args...);
case SpvOpGroupDecorate: return do_GroupDecorate(*i->Get<IGroupDecorate>(), args...);
case SpvOpGroupMemberDecorate: return do_GroupMemberDecorate(*i->Get<IGroupMemberDecorate>(), args...);
case SpvOpVectorExtractDynamic: return do_VectorExtractDynamic(*i->Get<IVectorExtractDynamic>(), args...);
case SpvOpVectorInsertDynamic: return do_VectorInsertDynamic(*i->Get<IVectorInsertDynamic>(), args...);
case SpvOpVectorShuffle: return do_VectorShuffle(*i->Get<IVectorShuffle>(), args...);
case SpvOpCompositeConstruct: return do_CompositeConstruct(*i->Get<ICompositeConstruct>(), args...);
case SpvOpCompositeExtract: return do_CompositeExtract(*i->Get<ICompositeExtract>(), args...);
case SpvOpCompositeInsert: return do_CompositeInsert(*i->Get<ICompositeInsert>(), args...);
case SpvOpCopyObject: return do_CopyObject(*i->Get<ICopyObject>(), args...);
case SpvOpTranspose: return do_Transpose(*i->Get<ITranspose>(), args...);
case SpvOpSampledImage: return do_SampledImage(*i->Get<ISampledImage>(), args...);
case SpvOpImageSampleImplicitLod: return do_ImageSampleImplicitLod(*i->Get<IImageSampleImplicitLod>(), args...);
case SpvOpImageSampleExplicitLod: return do_ImageSampleExplicitLod(*i->Get<IImageSampleExplicitLod>(), args...);
case SpvOpImageSampleDrefImplicitLod: return do_ImageSampleDrefImplicitLod(*i->Get<IImageSampleDrefImplicitLod>(), args...);
case SpvOpImageSampleDrefExplicitLod: return do_ImageSampleDrefExplicitLod(*i->Get<IImageSampleDrefExplicitLod>(), args...);
case SpvOpImageSampleProjImplicitLod: return do_ImageSampleProjImplicitLod(*i->Get<IImageSampleProjImplicitLod>(), args...);
case SpvOpImageSampleProjExplicitLod: return do_ImageSampleProjExplicitLod(*i->Get<IImageSampleProjExplicitLod>(), args...);
case SpvOpImageSampleProjDrefImplicitLod: return do_ImageSampleProjDrefImplicitLod(*i->Get<IImageSampleProjDrefImplicitLod>(), args...);
case SpvOpImageSampleProjDrefExplicitLod: return do_ImageSampleProjDrefExplicitLod(*i->Get<IImageSampleProjDrefExplicitLod>(), args...);
case SpvOpImageFetch: return do_ImageFetch(*i->Get<IImageFetch>(), args...);
case SpvOpImageGather: return do_ImageGather(*i->Get<IImageGather>(), args...);
case SpvOpImageDrefGather: return do_ImageDrefGather(*i->Get<IImageDrefGather>(), args...);
case SpvOpImageRead: return do_ImageRead(*i->Get<IImageRead>(), args...);
case SpvOpImageWrite: return do_ImageWrite(*i->Get<IImageWrite>(), args...);
case SpvOpImage: return do_Image(*i->Get<IImage>(), args...);
case SpvOpImageQueryFormat: return do_ImageQueryFormat(*i->Get<IImageQueryFormat>(), args...);
case SpvOpImageQueryOrder: return do_ImageQueryOrder(*i->Get<IImageQueryOrder>(), args...);
case SpvOpImageQuerySizeLod: return do_ImageQuerySizeLod(*i->Get<IImageQuerySizeLod>(), args...);
case SpvOpImageQuerySize: return do_ImageQuerySize(*i->Get<IImageQuerySize>(), args...);
case SpvOpImageQueryLod: return do_ImageQueryLod(*i->Get<IImageQueryLod>(), args...);
case SpvOpImageQueryLevels: return do_ImageQueryLevels(*i->Get<IImageQueryLevels>(), args...);
case SpvOpImageQuerySamples: return do_ImageQuerySamples(*i->Get<IImageQuerySamples>(), args...);
case SpvOpConvertFToU: return do_ConvertFToU(*i->Get<IConvertFToU>(), args...);
case SpvOpConvertFToS: return do_ConvertFToS(*i->Get<IConvertFToS>(), args...);
case SpvOpConvertSToF: return do_ConvertSToF(*i->Get<IConvertSToF>(), args...);
case SpvOpConvertUToF: return do_ConvertUToF(*i->Get<IConvertUToF>(), args...);
case SpvOpUConvert: return do_UConvert(*i->Get<IUConvert>(), args...);
case SpvOpSConvert: return do_SConvert(*i->Get<ISConvert>(), args...);
case SpvOpFConvert: return do_FConvert(*i->Get<IFConvert>(), args...);
case SpvOpQuantizeToF16: return do_QuantizeToF16(*i->Get<IQuantizeToF16>(), args...);
case SpvOpConvertPtrToU: return do_ConvertPtrToU(*i->Get<IConvertPtrToU>(), args...);
case SpvOpSatConvertSToU: return do_SatConvertSToU(*i->Get<ISatConvertSToU>(), args...);
case SpvOpSatConvertUToS: return do_SatConvertUToS(*i->Get<ISatConvertUToS>(), args...);
case SpvOpConvertUToPtr: return do_ConvertUToPtr(*i->Get<IConvertUToPtr>(), args...);
case SpvOpPtrCastToGeneric: return do_PtrCastToGeneric(*i->Get<IPtrCastToGeneric>(), args...);
case SpvOpGenericCastToPtr: return do_GenericCastToPtr(*i->Get<IGenericCastToPtr>(), args...);
case SpvOpGenericCastToPtrExplicit: return do_GenericCastToPtrExplicit(*i->Get<IGenericCastToPtrExplicit>(), args...);
case SpvOpBitcast: return do_Bitcast(*i->Get<IBitcast>(), args...);
case SpvOpSNegate: return do_SNegate(*i->Get<ISNegate>(), args...);
case SpvOpFNegate: return do_FNegate(*i->Get<IFNegate>(), args...);
case SpvOpIAdd: return do_IAdd(*i->Get<IIAdd>(), args...);
case SpvOpFAdd: return do_FAdd(*i->Get<IFAdd>(), args...);
case SpvOpISub: return do_ISub(*i->Get<IISub>(), args...);
case SpvOpFSub: return do_FSub(*i->Get<IFSub>(), args...);
case SpvOpIMul: return do_IMul(*i->Get<IIMul>(), args...);
case SpvOpFMul: return do_FMul(*i->Get<IFMul>(), args...);
case SpvOpUDiv: return do_UDiv(*i->Get<IUDiv>(), args...);
case SpvOpSDiv: return do_SDiv(*i->Get<ISDiv>(), args...);
case SpvOpFDiv: return do_FDiv(*i->Get<IFDiv>(), args...);
case SpvOpUMod: return do_UMod(*i->Get<IUMod>(), args...);
case SpvOpSRem: return do_SRem(*i->Get<ISRem>(), args...);
case SpvOpSMod: return do_SMod(*i->Get<ISMod>(), args...);
case SpvOpFRem: return do_FRem(*i->Get<IFRem>(), args...);
case SpvOpFMod: return do_FMod(*i->Get<IFMod>(), args...);
case SpvOpVectorTimesScalar: return do_VectorTimesScalar(*i->Get<IVectorTimesScalar>(), args...);
case SpvOpMatrixTimesScalar: return do_MatrixTimesScalar(*i->Get<IMatrixTimesScalar>(), args...);
case SpvOpVectorTimesMatrix: return do_VectorTimesMatrix(*i->Get<IVectorTimesMatrix>(), args...);
case SpvOpMatrixTimesVector: return do_MatrixTimesVector(*i->Get<IMatrixTimesVector>(), args...);
case SpvOpMatrixTimesMatrix: return do_MatrixTimesMatrix(*i->Get<IMatrixTimesMatrix>(), args...);
case SpvOpOuterProduct: return do_OuterProduct(*i->Get<IOuterProduct>(), args...);
case SpvOpDot: return do_Dot(*i->Get<IDot>(), args...);
case SpvOpIAddCarry: return do_IAddCarry(*i->Get<IIAddCarry>(), args...);
case SpvOpISubBorrow: return do_ISubBorrow(*i->Get<IISubBorrow>(), args...);
case SpvOpUMulExtended: return do_UMulExtended(*i->Get<IUMulExtended>(), args...);
case SpvOpSMulExtended: return do_SMulExtended(*i->Get<ISMulExtended>(), args...);
case SpvOpAny: return do_Any(*i->Get<IAny>(), args...);
case SpvOpAll: return do_All(*i->Get<IAll>(), args...);
case SpvOpIsNan: return do_IsNan(*i->Get<IIsNan>(), args...);
case SpvOpIsInf: return do_IsInf(*i->Get<IIsInf>(), args...);
case SpvOpIsFinite: return do_IsFinite(*i->Get<IIsFinite>(), args...);
case SpvOpIsNormal: return do_IsNormal(*i->Get<IIsNormal>(), args...);
case SpvOpSignBitSet: return do_SignBitSet(*i->Get<ISignBitSet>(), args...);
case SpvOpLessOrGreater: return do_LessOrGreater(*i->Get<ILessOrGreater>(), args...);
case SpvOpOrdered: return do_Ordered(*i->Get<IOrdered>(), args...);
case SpvOpUnordered: return do_Unordered(*i->Get<IUnordered>(), args...);
case SpvOpLogicalEqual: return do_LogicalEqual(*i->Get<ILogicalEqual>(), args...);
case SpvOpLogicalNotEqual: return do_LogicalNotEqual(*i->Get<ILogicalNotEqual>(), args...);
case SpvOpLogicalOr: return do_LogicalOr(*i->Get<ILogicalOr>(), args...);
case SpvOpLogicalAnd: return do_LogicalAnd(*i->Get<ILogicalAnd>(), args...);
case SpvOpLogicalNot: return do_LogicalNot(*i->Get<ILogicalNot>(), args...);
case SpvOpSelect: return do_Select(*i->Get<ISelect>(), args...);
case SpvOpIEqual: return do_IEqual(*i->Get<IIEqual>(), args...);
case SpvOpINotEqual: return do_INotEqual(*i->Get<IINotEqual>(), args...);
case SpvOpUGreaterThan: return do_UGreaterThan(*i->Get<IUGreaterThan>(), args...);
case SpvOpSGreaterThan: return do_SGreaterThan(*i->Get<ISGreaterThan>(), args...);
case SpvOpUGreaterThanEqual: return do_UGreaterThanEqual(*i->Get<IUGreaterThanEqual>(), args...);
case SpvOpSGreaterThanEqual: return do_SGreaterThanEqual(*i->Get<ISGreaterThanEqual>(), args...);
case SpvOpULessThan: return do_ULessThan(*i->Get<IULessThan>(), args...);
case SpvOpSLessThan: return do_SLessThan(*i->Get<ISLessThan>(), args...);
case SpvOpULessThanEqual: return do_ULessThanEqual(*i->Get<IULessThanEqual>(), args...);
case SpvOpSLessThanEqual: return do_SLessThanEqual(*i->Get<ISLessThanEqual>(), args...);
case SpvOpFOrdEqual: return do_FOrdEqual(*i->Get<IFOrdEqual>(), args...);
case SpvOpFUnordEqual: return do_FUnordEqual(*i->Get<IFUnordEqual>(), args...);
case SpvOpFOrdNotEqual: return do_FOrdNotEqual(*i->Get<IFOrdNotEqual>(), args...);
case SpvOpFUnordNotEqual: return do_FUnordNotEqual(*i->Get<IFUnordNotEqual>(), args...);
case SpvOpFOrdLessThan: return do_FOrdLessThan(*i->Get<IFOrdLessThan>(), args...);
case SpvOpFUnordLessThan: return do_FUnordLessThan(*i->Get<IFUnordLessThan>(), args...);
case SpvOpFOrdGreaterThan: return do_FOrdGreaterThan(*i->Get<IFOrdGreaterThan>(), args...);
case SpvOpFUnordGreaterThan: return do_FUnordGreaterThan(*i->Get<IFUnordGreaterThan>(), args...);
case SpvOpFOrdLessThanEqual: return do_FOrdLessThanEqual(*i->Get<IFOrdLessThanEqual>(), args...);
case SpvOpFUnordLessThanEqual: return do_FUnordLessThanEqual(*i->Get<IFUnordLessThanEqual>(), args...);
case SpvOpFOrdGreaterThanEqual: return do_FOrdGreaterThanEqual(*i->Get<IFOrdGreaterThanEqual>(), args...);
case SpvOpFUnordGreaterThanEqual: return do_FUnordGreaterThanEqual(*i->Get<IFUnordGreaterThanEqual>(), args...);
case SpvOpShiftRightLogical: return do_ShiftRightLogical(*i->Get<IShiftRightLogical>(), args...);
case SpvOpShiftRightArithmetic: return do_ShiftRightArithmetic(*i->Get<IShiftRightArithmetic>(), args...);
case SpvOpShiftLeftLogical: return do_ShiftLeftLogical(*i->Get<IShiftLeftLogical>(), args...);
case SpvOpBitwiseOr: return do_BitwiseOr(*i->Get<IBitwiseOr>(), args...);
case SpvOpBitwiseXor: return do_BitwiseXor(*i->Get<IBitwiseXor>(), args...);
case SpvOpBitwiseAnd: return do_BitwiseAnd(*i->Get<IBitwiseAnd>(), args...);
case SpvOpNot: return do_Not(*i->Get<INot>(), args...);
case SpvOpBitFieldInsert: return do_BitFieldInsert(*i->Get<IBitFieldInsert>(), args...);
case SpvOpBitFieldSExtract: return do_BitFieldSExtract(*i->Get<IBitFieldSExtract>(), args...);
case SpvOpBitFieldUExtract: return do_BitFieldUExtract(*i->Get<IBitFieldUExtract>(), args...);
case SpvOpBitReverse: return do_BitReverse(*i->Get<IBitReverse>(), args...);
case SpvOpBitCount: return do_BitCount(*i->Get<IBitCount>(), args...);
case SpvOpDPdx: return do_DPdx(*i->Get<IDPdx>(), args...);
case SpvOpDPdy: return do_DPdy(*i->Get<IDPdy>(), args...);
case SpvOpFwidth: return do_Fwidth(*i->Get<IFwidth>(), args...);
case SpvOpDPdxFine: return do_DPdxFine(*i->Get<IDPdxFine>(), args...);
case SpvOpDPdyFine: return do_DPdyFine(*i->Get<IDPdyFine>(), args...);
case SpvOpFwidthFine: return do_FwidthFine(*i->Get<IFwidthFine>(), args...);
case SpvOpDPdxCoarse: return do_DPdxCoarse(*i->Get<IDPdxCoarse>(), args...);
case SpvOpDPdyCoarse: return do_DPdyCoarse(*i->Get<IDPdyCoarse>(), args...);
case SpvOpFwidthCoarse: return do_FwidthCoarse(*i->Get<IFwidthCoarse>(), args...);
case SpvOpEmitVertex: return do_EmitVertex(*i->Get<IEmitVertex>(), args...);
case SpvOpEndPrimitive: return do_EndPrimitive(*i->Get<IEndPrimitive>(), args...);
case SpvOpEmitStreamVertex: return do_EmitStreamVertex(*i->Get<IEmitStreamVertex>(), args...);
case SpvOpEndStreamPrimitive: return do_EndStreamPrimitive(*i->Get<IEndStreamPrimitive>(), args...);
case SpvOpControlBarrier: return do_ControlBarrier(*i->Get<IControlBarrier>(), args...);
case SpvOpMemoryBarrier: return do_MemoryBarrier(*i->Get<IMemoryBarrier>(), args...);
case SpvOpAtomicLoad: return do_AtomicLoad(*i->Get<IAtomicLoad>(), args...);
case SpvOpAtomicStore: return do_AtomicStore(*i->Get<IAtomicStore>(), args...);
case SpvOpAtomicExchange: return do_AtomicExchange(*i->Get<IAtomicExchange>(), args...);
case SpvOpAtomicCompareExchange: return do_AtomicCompareExchange(*i->Get<IAtomicCompareExchange>(), args...);
case SpvOpAtomicCompareExchangeWeak: return do_AtomicCompareExchangeWeak(*i->Get<IAtomicCompareExchangeWeak>(), args...);
case SpvOpAtomicIIncrement: return do_AtomicIIncrement(*i->Get<IAtomicIIncrement>(), args...);
case SpvOpAtomicIDecrement: return do_AtomicIDecrement(*i->Get<IAtomicIDecrement>(), args...);
case SpvOpAtomicIAdd: return do_AtomicIAdd(*i->Get<IAtomicIAdd>(), args...);
case SpvOpAtomicISub: return do_AtomicISub(*i->Get<IAtomicISub>(), args...);
case SpvOpAtomicSMin: return do_AtomicSMin(*i->Get<IAtomicSMin>(), args...);
case SpvOpAtomicUMin: return do_AtomicUMin(*i->Get<IAtomicUMin>(), args...);
case SpvOpAtomicSMax: return do_AtomicSMax(*i->Get<IAtomicSMax>(), args...);
case SpvOpAtomicUMax: return do_AtomicUMax(*i->Get<IAtomicUMax>(), args...);
case SpvOpAtomicAnd: return do_AtomicAnd(*i->Get<IAtomicAnd>(), args...);
case SpvOpAtomicOr: return do_AtomicOr(*i->Get<IAtomicOr>(), args...);
case SpvOpAtomicXor: return do_AtomicXor(*i->Get<IAtomicXor>(), args...);
case SpvOpPhi: return do_Phi(*i->Get<IPhi>(), args...);
case SpvOpLoopMerge: return do_LoopMerge(*i->Get<ILoopMerge>(), args...);
case SpvOpSelectionMerge: return do_SelectionMerge(*i->Get<ISelectionMerge>(), args...);
case SpvOpLabel: return do_Label(*i->Get<ILabel>(), args...);
case SpvOpBranch: return do_Branch(*i->Get<IBranch>(), args...);
case SpvOpBranchConditional: return do_BranchConditional(*i->Get<IBranchConditional>(), args...);
case SpvOpSwitch: return do_Switch(*i->Get<ISwitch>(), args...);
case SpvOpKill: return do_Kill(*i->Get<IKill>(), args...);
case SpvOpReturn: return do_Return(*i->Get<IReturn>(), args...);
case SpvOpReturnValue: return do_ReturnValue(*i->Get<IReturnValue>(), args...);
case SpvOpUnreachable: return do_Unreachable(*i->Get<IUnreachable>(), args...);
case SpvOpLifetimeStart: return do_LifetimeStart(*i->Get<ILifetimeStart>(), args...);
case SpvOpLifetimeStop: return do_LifetimeStop(*i->Get<ILifetimeStop>(), args...);
case SpvOpGroupAsyncCopy: return do_GroupAsyncCopy(*i->Get<IGroupAsyncCopy>(), args...);
case SpvOpGroupWaitEvents: return do_GroupWaitEvents(*i->Get<IGroupWaitEvents>(), args...);
case SpvOpGroupAll: return do_GroupAll(*i->Get<IGroupAll>(), args...);
case SpvOpGroupAny: return do_GroupAny(*i->Get<IGroupAny>(), args...);
case SpvOpGroupBroadcast: return do_GroupBroadcast(*i->Get<IGroupBroadcast>(), args...);
case SpvOpGroupIAdd: return do_GroupIAdd(*i->Get<IGroupIAdd>(), args...);
case SpvOpGroupFAdd: return do_GroupFAdd(*i->Get<IGroupFAdd>(), args...);
case SpvOpGroupFMin: return do_GroupFMin(*i->Get<IGroupFMin>(), args...);
case SpvOpGroupUMin: return do_GroupUMin(*i->Get<IGroupUMin>(), args...);
case SpvOpGroupSMin: return do_GroupSMin(*i->Get<IGroupSMin>(), args...);
case SpvOpGroupFMax: return do_GroupFMax(*i->Get<IGroupFMax>(), args...);
case SpvOpGroupUMax: return do_GroupUMax(*i->Get<IGroupUMax>(), args...);
case SpvOpGroupSMax: return do_GroupSMax(*i->Get<IGroupSMax>(), args...);
case SpvOpReadPipe: return do_ReadPipe(*i->Get<IReadPipe>(), args...);
case SpvOpWritePipe: return do_WritePipe(*i->Get<IWritePipe>(), args...);
case SpvOpReservedReadPipe: return do_ReservedReadPipe(*i->Get<IReservedReadPipe>(), args...);
case SpvOpReservedWritePipe: return do_ReservedWritePipe(*i->Get<IReservedWritePipe>(), args...);
case SpvOpReserveReadPipePackets: return do_ReserveReadPipePackets(*i->Get<IReserveReadPipePackets>(), args...);
case SpvOpReserveWritePipePackets: return do_ReserveWritePipePackets(*i->Get<IReserveWritePipePackets>(), args...);
case SpvOpCommitReadPipe: return do_CommitReadPipe(*i->Get<ICommitReadPipe>(), args...);
case SpvOpCommitWritePipe: return do_CommitWritePipe(*i->Get<ICommitWritePipe>(), args...);
case SpvOpIsValidReserveId: return do_IsValidReserveId(*i->Get<IIsValidReserveId>(), args...);
case SpvOpGetNumPipePackets: return do_GetNumPipePackets(*i->Get<IGetNumPipePackets>(), args...);
case SpvOpGetMaxPipePackets: return do_GetMaxPipePackets(*i->Get<IGetMaxPipePackets>(), args...);
case SpvOpGroupReserveReadPipePackets: return do_GroupReserveReadPipePackets(*i->Get<IGroupReserveReadPipePackets>(), args...);
case SpvOpGroupReserveWritePipePackets: return do_GroupReserveWritePipePackets(*i->Get<IGroupReserveWritePipePackets>(), args...);
case SpvOpGroupCommitReadPipe: return do_GroupCommitReadPipe(*i->Get<IGroupCommitReadPipe>(), args...);
case SpvOpGroupCommitWritePipe: return do_GroupCommitWritePipe(*i->Get<IGroupCommitWritePipe>(), args...);
case SpvOpEnqueueMarker: return do_EnqueueMarker(*i->Get<IEnqueueMarker>(), args...);
case SpvOpEnqueueKernel: return do_EnqueueKernel(*i->Get<IEnqueueKernel>(), args...);
case SpvOpGetKernelNDrangeSubGroupCount: return do_GetKernelNDrangeSubGroupCount(*i->Get<IGetKernelNDrangeSubGroupCount>(), args...);
case SpvOpGetKernelNDrangeMaxSubGroupSize: return do_GetKernelNDrangeMaxSubGroupSize(*i->Get<IGetKernelNDrangeMaxSubGroupSize>(), args...);
case SpvOpGetKernelWorkGroupSize: return do_GetKernelWorkGroupSize(*i->Get<IGetKernelWorkGroupSize>(), args...);
case SpvOpGetKernelPreferredWorkGroupSizeMultiple: return do_GetKernelPreferredWorkGroupSizeMultiple(*i->Get<IGetKernelPreferredWorkGroupSizeMultiple>(), args...);
case SpvOpRetainEvent: return do_RetainEvent(*i->Get<IRetainEvent>(), args...);
case SpvOpReleaseEvent: return do_ReleaseEvent(*i->Get<IReleaseEvent>(), args...);
case SpvOpCreateUserEvent: return do_CreateUserEvent(*i->Get<ICreateUserEvent>(), args...);
case SpvOpIsValidEvent: return do_IsValidEvent(*i->Get<IIsValidEvent>(), args...);
case SpvOpSetUserEventStatus: return do_SetUserEventStatus(*i->Get<ISetUserEventStatus>(), args...);
case SpvOpCaptureEventProfilingInfo: return do_CaptureEventProfilingInfo(*i->Get<ICaptureEventProfilingInfo>(), args...);
case SpvOpGetDefaultQueue: return do_GetDefaultQueue(*i->Get<IGetDefaultQueue>(), args...);
case SpvOpBuildNDRange: return do_BuildNDRange(*i->Get<IBuildNDRange>(), args...);
case SpvOpImageSparseSampleImplicitLod: return do_ImageSparseSampleImplicitLod(*i->Get<IImageSparseSampleImplicitLod>(), args...);
case SpvOpImageSparseSampleExplicitLod: return do_ImageSparseSampleExplicitLod(*i->Get<IImageSparseSampleExplicitLod>(), args...);
case SpvOpImageSparseSampleDrefImplicitLod: return do_ImageSparseSampleDrefImplicitLod(*i->Get<IImageSparseSampleDrefImplicitLod>(), args...);
case SpvOpImageSparseSampleDrefExplicitLod: return do_ImageSparseSampleDrefExplicitLod(*i->Get<IImageSparseSampleDrefExplicitLod>(), args...);
case SpvOpImageSparseSampleProjImplicitLod: return do_ImageSparseSampleProjImplicitLod(*i->Get<IImageSparseSampleProjImplicitLod>(), args...);
case SpvOpImageSparseSampleProjExplicitLod: return do_ImageSparseSampleProjExplicitLod(*i->Get<IImageSparseSampleProjExplicitLod>(), args...);
case SpvOpImageSparseSampleProjDrefImplicitLod: return do_ImageSparseSampleProjDrefImplicitLod(*i->Get<IImageSparseSampleProjDrefImplicitLod>(), args...);
case SpvOpImageSparseSampleProjDrefExplicitLod: return do_ImageSparseSampleProjDrefExplicitLod(*i->Get<IImageSparseSampleProjDrefExplicitLod>(), args...);
case SpvOpImageSparseFetch: return do_ImageSparseFetch(*i->Get<IImageSparseFetch>(), args...);
case SpvOpImageSparseGather: return do_ImageSparseGather(*i->Get<IImageSparseGather>(), args...);
case SpvOpImageSparseDrefGather: return do_ImageSparseDrefGather(*i->Get<IImageSparseDrefGather>(), args...);
case SpvOpImageSparseTexelsResident: return do_ImageSparseTexelsResident(*i->Get<IImageSparseTexelsResident>(), args...);
case SpvOpNoLine: return do_NoLine(*i->Get<INoLine>(), args...);
case SpvOpAtomicFlagTestAndSet: return do_AtomicFlagTestAndSet(*i->Get<IAtomicFlagTestAndSet>(), args...);
case SpvOpAtomicFlagClear: return do_AtomicFlagClear(*i->Get<IAtomicFlagClear>(), args...);
case SpvOpImageSparseRead: return do_ImageSparseRead(*i->Get<IImageSparseRead>(), args...);
case SpvOpSizeOf: return do_SizeOf(*i->Get<ISizeOf>(), args...);
case SpvOpTypePipeStorage: return do_TypePipeStorage(*i->Get<ITypePipeStorage>(), args...);
case SpvOpConstantPipeStorage: return do_ConstantPipeStorage(*i->Get<IConstantPipeStorage>(), args...);
case SpvOpCreatePipeFromPipeStorage: return do_CreatePipeFromPipeStorage(*i->Get<ICreatePipeFromPipeStorage>(), args...);
case SpvOpGetKernelLocalSizeForSubgroupCount: return do_GetKernelLocalSizeForSubgroupCount(*i->Get<IGetKernelLocalSizeForSubgroupCount>(), args...);
case SpvOpGetKernelMaxNumSubgroups: return do_GetKernelMaxNumSubgroups(*i->Get<IGetKernelMaxNumSubgroups>(), args...);
case SpvOpTypeNamedBarrier: return do_TypeNamedBarrier(*i->Get<ITypeNamedBarrier>(), args...);
case SpvOpNamedBarrierInitialize: return do_NamedBarrierInitialize(*i->Get<INamedBarrierInitialize>(), args...);
case SpvOpMemoryNamedBarrier: return do_MemoryNamedBarrier(*i->Get<IMemoryNamedBarrier>(), args...);
case SpvOpModuleProcessed: return do_ModuleProcessed(*i->Get<IModuleProcessed>(), args...);
case SpvOpExecutionModeId: return do_ExecutionModeId(*i->Get<IExecutionModeId>(), args...);
case SpvOpDecorateId: return do_DecorateId(*i->Get<IDecorateId>(), args...);
case SpvOpGroupNonUniformElect: return do_GroupNonUniformElect(*i->Get<IGroupNonUniformElect>(), args...);
case SpvOpGroupNonUniformAll: return do_GroupNonUniformAll(*i->Get<IGroupNonUniformAll>(), args...);
case SpvOpGroupNonUniformAny: return do_GroupNonUniformAny(*i->Get<IGroupNonUniformAny>(), args...);
case SpvOpGroupNonUniformAllEqual: return do_GroupNonUniformAllEqual(*i->Get<IGroupNonUniformAllEqual>(), args...);
case SpvOpGroupNonUniformBroadcast: return do_GroupNonUniformBroadcast(*i->Get<IGroupNonUniformBroadcast>(), args...);
case SpvOpGroupNonUniformBroadcastFirst: return do_GroupNonUniformBroadcastFirst(*i->Get<IGroupNonUniformBroadcastFirst>(), args...);
case SpvOpGroupNonUniformBallot: return do_GroupNonUniformBallot(*i->Get<IGroupNonUniformBallot>(), args...);
case SpvOpGroupNonUniformInverseBallot: return do_GroupNonUniformInverseBallot(*i->Get<IGroupNonUniformInverseBallot>(), args...);
case SpvOpGroupNonUniformBallotBitExtract: return do_GroupNonUniformBallotBitExtract(*i->Get<IGroupNonUniformBallotBitExtract>(), args...);
case SpvOpGroupNonUniformBallotBitCount: return do_GroupNonUniformBallotBitCount(*i->Get<IGroupNonUniformBallotBitCount>(), args...);
case SpvOpGroupNonUniformBallotFindLSB: return do_GroupNonUniformBallotFindLSB(*i->Get<IGroupNonUniformBallotFindLSB>(), args...);
case SpvOpGroupNonUniformBallotFindMSB: return do_GroupNonUniformBallotFindMSB(*i->Get<IGroupNonUniformBallotFindMSB>(), args...);
case SpvOpGroupNonUniformShuffle: return do_GroupNonUniformShuffle(*i->Get<IGroupNonUniformShuffle>(), args...);
case SpvOpGroupNonUniformShuffleXor: return do_GroupNonUniformShuffleXor(*i->Get<IGroupNonUniformShuffleXor>(), args...);
case SpvOpGroupNonUniformShuffleUp: return do_GroupNonUniformShuffleUp(*i->Get<IGroupNonUniformShuffleUp>(), args...);
case SpvOpGroupNonUniformShuffleDown: return do_GroupNonUniformShuffleDown(*i->Get<IGroupNonUniformShuffleDown>(), args...);
case SpvOpGroupNonUniformIAdd: return do_GroupNonUniformIAdd(*i->Get<IGroupNonUniformIAdd>(), args...);
case SpvOpGroupNonUniformFAdd: return do_GroupNonUniformFAdd(*i->Get<IGroupNonUniformFAdd>(), args...);
case SpvOpGroupNonUniformIMul: return do_GroupNonUniformIMul(*i->Get<IGroupNonUniformIMul>(), args...);
case SpvOpGroupNonUniformFMul: return do_GroupNonUniformFMul(*i->Get<IGroupNonUniformFMul>(), args...);
case SpvOpGroupNonUniformSMin: return do_GroupNonUniformSMin(*i->Get<IGroupNonUniformSMin>(), args...);
case SpvOpGroupNonUniformUMin: return do_GroupNonUniformUMin(*i->Get<IGroupNonUniformUMin>(), args...);
case SpvOpGroupNonUniformFMin: return do_GroupNonUniformFMin(*i->Get<IGroupNonUniformFMin>(), args...);
case SpvOpGroupNonUniformSMax: return do_GroupNonUniformSMax(*i->Get<IGroupNonUniformSMax>(), args...);
case SpvOpGroupNonUniformUMax: return do_GroupNonUniformUMax(*i->Get<IGroupNonUniformUMax>(), args...);
case SpvOpGroupNonUniformFMax: return do_GroupNonUniformFMax(*i->Get<IGroupNonUniformFMax>(), args...);
case SpvOpGroupNonUniformBitwiseAnd: return do_GroupNonUniformBitwiseAnd(*i->Get<IGroupNonUniformBitwiseAnd>(), args...);
case SpvOpGroupNonUniformBitwiseOr: return do_GroupNonUniformBitwiseOr(*i->Get<IGroupNonUniformBitwiseOr>(), args...);
case SpvOpGroupNonUniformBitwiseXor: return do_GroupNonUniformBitwiseXor(*i->Get<IGroupNonUniformBitwiseXor>(), args...);
case SpvOpGroupNonUniformLogicalAnd: return do_GroupNonUniformLogicalAnd(*i->Get<IGroupNonUniformLogicalAnd>(), args...);
case SpvOpGroupNonUniformLogicalOr: return do_GroupNonUniformLogicalOr(*i->Get<IGroupNonUniformLogicalOr>(), args...);
case SpvOpGroupNonUniformLogicalXor: return do_GroupNonUniformLogicalXor(*i->Get<IGroupNonUniformLogicalXor>(), args...);
case SpvOpGroupNonUniformQuadBroadcast: return do_GroupNonUniformQuadBroadcast(*i->Get<IGroupNonUniformQuadBroadcast>(), args...);
case SpvOpGroupNonUniformQuadSwap: return do_GroupNonUniformQuadSwap(*i->Get<IGroupNonUniformQuadSwap>(), args...);
case SpvOpCopyLogical: return do_CopyLogical(*i->Get<ICopyLogical>(), args...);
case SpvOpPtrEqual: return do_PtrEqual(*i->Get<IPtrEqual>(), args...);
case SpvOpPtrNotEqual: return do_PtrNotEqual(*i->Get<IPtrNotEqual>(), args...);
case SpvOpPtrDiff: return do_PtrDiff(*i->Get<IPtrDiff>(), args...);
case SpvOpSubgroupBallotKHR: return do_SubgroupBallotKHR(*i->Get<ISubgroupBallotKHR>(), args...);
case SpvOpSubgroupFirstInvocationKHR: return do_SubgroupFirstInvocationKHR(*i->Get<ISubgroupFirstInvocationKHR>(), args...);
case SpvOpSubgroupAllKHR: return do_SubgroupAllKHR(*i->Get<ISubgroupAllKHR>(), args...);
case SpvOpSubgroupAnyKHR: return do_SubgroupAnyKHR(*i->Get<ISubgroupAnyKHR>(), args...);
case SpvOpSubgroupAllEqualKHR: return do_SubgroupAllEqualKHR(*i->Get<ISubgroupAllEqualKHR>(), args...);
case SpvOpSubgroupReadInvocationKHR: return do_SubgroupReadInvocationKHR(*i->Get<ISubgroupReadInvocationKHR>(), args...);
case SpvOpGroupIAddNonUniformAMD: return do_GroupIAddNonUniformAMD(*i->Get<IGroupIAddNonUniformAMD>(), args...);
case SpvOpGroupFAddNonUniformAMD: return do_GroupFAddNonUniformAMD(*i->Get<IGroupFAddNonUniformAMD>(), args...);
case SpvOpGroupFMinNonUniformAMD: return do_GroupFMinNonUniformAMD(*i->Get<IGroupFMinNonUniformAMD>(), args...);
case SpvOpGroupUMinNonUniformAMD: return do_GroupUMinNonUniformAMD(*i->Get<IGroupUMinNonUniformAMD>(), args...);
case SpvOpGroupSMinNonUniformAMD: return do_GroupSMinNonUniformAMD(*i->Get<IGroupSMinNonUniformAMD>(), args...);
case SpvOpGroupFMaxNonUniformAMD: return do_GroupFMaxNonUniformAMD(*i->Get<IGroupFMaxNonUniformAMD>(), args...);
case SpvOpGroupUMaxNonUniformAMD: return do_GroupUMaxNonUniformAMD(*i->Get<IGroupUMaxNonUniformAMD>(), args...);
case SpvOpGroupSMaxNonUniformAMD: return do_GroupSMaxNonUniformAMD(*i->Get<IGroupSMaxNonUniformAMD>(), args...);
case SpvOpFragmentMaskFetchAMD: return do_FragmentMaskFetchAMD(*i->Get<IFragmentMaskFetchAMD>(), args...);
case SpvOpFragmentFetchAMD: return do_FragmentFetchAMD(*i->Get<IFragmentFetchAMD>(), args...);
case SpvOpImageSampleFootprintNV: return do_ImageSampleFootprintNV(*i->Get<IImageSampleFootprintNV>(), args...);
case SpvOpGroupNonUniformPartitionNV: return do_GroupNonUniformPartitionNV(*i->Get<IGroupNonUniformPartitionNV>(), args...);
case SpvOpWritePackedPrimitiveIndices4x8NV: return do_WritePackedPrimitiveIndices4x8NV(*i->Get<IWritePackedPrimitiveIndices4x8NV>(), args...);
case SpvOpReportIntersectionNV: return do_ReportIntersectionNV(*i->Get<IReportIntersectionNV>(), args...);
case SpvOpIgnoreIntersectionNV: return do_IgnoreIntersectionNV(*i->Get<IIgnoreIntersectionNV>(), args...);
case SpvOpTerminateRayNV: return do_TerminateRayNV(*i->Get<ITerminateRayNV>(), args...);
case SpvOpTraceNV: return do_TraceNV(*i->Get<ITraceNV>(), args...);
case SpvOpTypeAccelerationStructureNV: return do_TypeAccelerationStructureNV(*i->Get<ITypeAccelerationStructureNV>(), args...);
case SpvOpExecuteCallableNV: return do_ExecuteCallableNV(*i->Get<IExecuteCallableNV>(), args...);
case SpvOpTypeCooperativeMatrixNV: return do_TypeCooperativeMatrixNV(*i->Get<ITypeCooperativeMatrixNV>(), args...);
case SpvOpCooperativeMatrixLoadNV: return do_CooperativeMatrixLoadNV(*i->Get<ICooperativeMatrixLoadNV>(), args...);
case SpvOpCooperativeMatrixStoreNV: return do_CooperativeMatrixStoreNV(*i->Get<ICooperativeMatrixStoreNV>(), args...);
case SpvOpCooperativeMatrixMulAddNV: return do_CooperativeMatrixMulAddNV(*i->Get<ICooperativeMatrixMulAddNV>(), args...);
case SpvOpCooperativeMatrixLengthNV: return do_CooperativeMatrixLengthNV(*i->Get<ICooperativeMatrixLengthNV>(), args...);
case SpvOpSubgroupShuffleINTEL: return do_SubgroupShuffleINTEL(*i->Get<ISubgroupShuffleINTEL>(), args...);
case SpvOpSubgroupShuffleDownINTEL: return do_SubgroupShuffleDownINTEL(*i->Get<ISubgroupShuffleDownINTEL>(), args...);
case SpvOpSubgroupShuffleUpINTEL: return do_SubgroupShuffleUpINTEL(*i->Get<ISubgroupShuffleUpINTEL>(), args...);
case SpvOpSubgroupShuffleXorINTEL: return do_SubgroupShuffleXorINTEL(*i->Get<ISubgroupShuffleXorINTEL>(), args...);
case SpvOpSubgroupBlockReadINTEL: return do_SubgroupBlockReadINTEL(*i->Get<ISubgroupBlockReadINTEL>(), args...);
case SpvOpSubgroupBlockWriteINTEL: return do_SubgroupBlockWriteINTEL(*i->Get<ISubgroupBlockWriteINTEL>(), args...);
case SpvOpSubgroupImageBlockReadINTEL: return do_SubgroupImageBlockReadINTEL(*i->Get<ISubgroupImageBlockReadINTEL>(), args...);
case SpvOpSubgroupImageBlockWriteINTEL: return do_SubgroupImageBlockWriteINTEL(*i->Get<ISubgroupImageBlockWriteINTEL>(), args...);
case SpvOpSubgroupImageMediaBlockReadINTEL: return do_SubgroupImageMediaBlockReadINTEL(*i->Get<ISubgroupImageMediaBlockReadINTEL>(), args...);
case SpvOpSubgroupImageMediaBlockWriteINTEL: return do_SubgroupImageMediaBlockWriteINTEL(*i->Get<ISubgroupImageMediaBlockWriteINTEL>(), args...);
case SpvOpDecorateString: return do_DecorateString(*i->Get<IDecorateString>(), args...);
case SpvOpMemberDecorateString: return do_MemberDecorateString(*i->Get<IMemberDecorateString>(), args...);
case SpvOpVmeImageINTEL: return do_VmeImageINTEL(*i->Get<IVmeImageINTEL>(), args...);
case SpvOpTypeVmeImageINTEL: return do_TypeVmeImageINTEL(*i->Get<ITypeVmeImageINTEL>(), args...);
case SpvOpTypeAvcImePayloadINTEL: return do_TypeAvcImePayloadINTEL(*i->Get<ITypeAvcImePayloadINTEL>(), args...);
case SpvOpTypeAvcRefPayloadINTEL: return do_TypeAvcRefPayloadINTEL(*i->Get<ITypeAvcRefPayloadINTEL>(), args...);
case SpvOpTypeAvcSicPayloadINTEL: return do_TypeAvcSicPayloadINTEL(*i->Get<ITypeAvcSicPayloadINTEL>(), args...);
case SpvOpTypeAvcMcePayloadINTEL: return do_TypeAvcMcePayloadINTEL(*i->Get<ITypeAvcMcePayloadINTEL>(), args...);
case SpvOpTypeAvcMceResultINTEL: return do_TypeAvcMceResultINTEL(*i->Get<ITypeAvcMceResultINTEL>(), args...);
case SpvOpTypeAvcImeResultINTEL: return do_TypeAvcImeResultINTEL(*i->Get<ITypeAvcImeResultINTEL>(), args...);
case SpvOpTypeAvcImeResultSingleReferenceStreamoutINTEL: return do_TypeAvcImeResultSingleReferenceStreamoutINTEL(*i->Get<ITypeAvcImeResultSingleReferenceStreamoutINTEL>(), args...);
case SpvOpTypeAvcImeResultDualReferenceStreamoutINTEL: return do_TypeAvcImeResultDualReferenceStreamoutINTEL(*i->Get<ITypeAvcImeResultDualReferenceStreamoutINTEL>(), args...);
case SpvOpTypeAvcImeSingleReferenceStreaminINTEL: return do_TypeAvcImeSingleReferenceStreaminINTEL(*i->Get<ITypeAvcImeSingleReferenceStreaminINTEL>(), args...);
case SpvOpTypeAvcImeDualReferenceStreaminINTEL: return do_TypeAvcImeDualReferenceStreaminINTEL(*i->Get<ITypeAvcImeDualReferenceStreaminINTEL>(), args...);
case SpvOpTypeAvcRefResultINTEL: return do_TypeAvcRefResultINTEL(*i->Get<ITypeAvcRefResultINTEL>(), args...);
case SpvOpTypeAvcSicResultINTEL: return do_TypeAvcSicResultINTEL(*i->Get<ITypeAvcSicResultINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL: return do_SubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL: return do_SubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL(*i->Get<ISubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL: return do_SubgroupAvcMceGetDefaultInterShapePenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultInterShapePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceSetInterShapePenaltyINTEL: return do_SubgroupAvcMceSetInterShapePenaltyINTEL(*i->Get<ISubgroupAvcMceSetInterShapePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL: return do_SubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceSetInterDirectionPenaltyINTEL: return do_SubgroupAvcMceSetInterDirectionPenaltyINTEL(*i->Get<ISubgroupAvcMceSetInterDirectionPenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL: return do_SubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL: return do_SubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL(*i->Get<ISubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL: return do_SubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL(*i->Get<ISubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL: return do_SubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL(*i->Get<ISubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL: return do_SubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL(*i->Get<ISubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL>(), args...);
case SpvOpSubgroupAvcMceSetMotionVectorCostFunctionINTEL: return do_SubgroupAvcMceSetMotionVectorCostFunctionINTEL(*i->Get<ISubgroupAvcMceSetMotionVectorCostFunctionINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL: return do_SubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL: return do_SubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL: return do_SubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL(*i->Get<ISubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcMceSetAcOnlyHaarINTEL: return do_SubgroupAvcMceSetAcOnlyHaarINTEL(*i->Get<ISubgroupAvcMceSetAcOnlyHaarINTEL>(), args...);
case SpvOpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL: return do_SubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL(*i->Get<ISubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL>(), args...);
case SpvOpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL: return do_SubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL(*i->Get<ISubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL>(), args...);
case SpvOpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL: return do_SubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL(*i->Get<ISubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL>(), args...);
case SpvOpSubgroupAvcMceConvertToImePayloadINTEL: return do_SubgroupAvcMceConvertToImePayloadINTEL(*i->Get<ISubgroupAvcMceConvertToImePayloadINTEL>(), args...);
case SpvOpSubgroupAvcMceConvertToImeResultINTEL: return do_SubgroupAvcMceConvertToImeResultINTEL(*i->Get<ISubgroupAvcMceConvertToImeResultINTEL>(), args...);
case SpvOpSubgroupAvcMceConvertToRefPayloadINTEL: return do_SubgroupAvcMceConvertToRefPayloadINTEL(*i->Get<ISubgroupAvcMceConvertToRefPayloadINTEL>(), args...);
case SpvOpSubgroupAvcMceConvertToRefResultINTEL: return do_SubgroupAvcMceConvertToRefResultINTEL(*i->Get<ISubgroupAvcMceConvertToRefResultINTEL>(), args...);
case SpvOpSubgroupAvcMceConvertToSicPayloadINTEL: return do_SubgroupAvcMceConvertToSicPayloadINTEL(*i->Get<ISubgroupAvcMceConvertToSicPayloadINTEL>(), args...);
case SpvOpSubgroupAvcMceConvertToSicResultINTEL: return do_SubgroupAvcMceConvertToSicResultINTEL(*i->Get<ISubgroupAvcMceConvertToSicResultINTEL>(), args...);
case SpvOpSubgroupAvcMceGetMotionVectorsINTEL: return do_SubgroupAvcMceGetMotionVectorsINTEL(*i->Get<ISubgroupAvcMceGetMotionVectorsINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterDistortionsINTEL: return do_SubgroupAvcMceGetInterDistortionsINTEL(*i->Get<ISubgroupAvcMceGetInterDistortionsINTEL>(), args...);
case SpvOpSubgroupAvcMceGetBestInterDistortionsINTEL: return do_SubgroupAvcMceGetBestInterDistortionsINTEL(*i->Get<ISubgroupAvcMceGetBestInterDistortionsINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterMajorShapeINTEL: return do_SubgroupAvcMceGetInterMajorShapeINTEL(*i->Get<ISubgroupAvcMceGetInterMajorShapeINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterMinorShapeINTEL: return do_SubgroupAvcMceGetInterMinorShapeINTEL(*i->Get<ISubgroupAvcMceGetInterMinorShapeINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterDirectionsINTEL: return do_SubgroupAvcMceGetInterDirectionsINTEL(*i->Get<ISubgroupAvcMceGetInterDirectionsINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterMotionVectorCountINTEL: return do_SubgroupAvcMceGetInterMotionVectorCountINTEL(*i->Get<ISubgroupAvcMceGetInterMotionVectorCountINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterReferenceIdsINTEL: return do_SubgroupAvcMceGetInterReferenceIdsINTEL(*i->Get<ISubgroupAvcMceGetInterReferenceIdsINTEL>(), args...);
case SpvOpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL: return do_SubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL(*i->Get<ISubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL>(), args...);
case SpvOpSubgroupAvcImeInitializeINTEL: return do_SubgroupAvcImeInitializeINTEL(*i->Get<ISubgroupAvcImeInitializeINTEL>(), args...);
case SpvOpSubgroupAvcImeSetSingleReferenceINTEL: return do_SubgroupAvcImeSetSingleReferenceINTEL(*i->Get<ISubgroupAvcImeSetSingleReferenceINTEL>(), args...);
case SpvOpSubgroupAvcImeSetDualReferenceINTEL: return do_SubgroupAvcImeSetDualReferenceINTEL(*i->Get<ISubgroupAvcImeSetDualReferenceINTEL>(), args...);
case SpvOpSubgroupAvcImeRefWindowSizeINTEL: return do_SubgroupAvcImeRefWindowSizeINTEL(*i->Get<ISubgroupAvcImeRefWindowSizeINTEL>(), args...);
case SpvOpSubgroupAvcImeAdjustRefOffsetINTEL: return do_SubgroupAvcImeAdjustRefOffsetINTEL(*i->Get<ISubgroupAvcImeAdjustRefOffsetINTEL>(), args...);
case SpvOpSubgroupAvcImeConvertToMcePayloadINTEL: return do_SubgroupAvcImeConvertToMcePayloadINTEL(*i->Get<ISubgroupAvcImeConvertToMcePayloadINTEL>(), args...);
case SpvOpSubgroupAvcImeSetMaxMotionVectorCountINTEL: return do_SubgroupAvcImeSetMaxMotionVectorCountINTEL(*i->Get<ISubgroupAvcImeSetMaxMotionVectorCountINTEL>(), args...);
case SpvOpSubgroupAvcImeSetUnidirectionalMixDisableINTEL: return do_SubgroupAvcImeSetUnidirectionalMixDisableINTEL(*i->Get<ISubgroupAvcImeSetUnidirectionalMixDisableINTEL>(), args...);
case SpvOpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL: return do_SubgroupAvcImeSetEarlySearchTerminationThresholdINTEL(*i->Get<ISubgroupAvcImeSetEarlySearchTerminationThresholdINTEL>(), args...);
case SpvOpSubgroupAvcImeSetWeightedSadINTEL: return do_SubgroupAvcImeSetWeightedSadINTEL(*i->Get<ISubgroupAvcImeSetWeightedSadINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceINTEL: return do_SubgroupAvcImeEvaluateWithSingleReferenceINTEL(*i->Get<ISubgroupAvcImeEvaluateWithSingleReferenceINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithDualReferenceINTEL: return do_SubgroupAvcImeEvaluateWithDualReferenceINTEL(*i->Get<ISubgroupAvcImeEvaluateWithDualReferenceINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL: return do_SubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL(*i->Get<ISubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL: return do_SubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL(*i->Get<ISubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL: return do_SubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL(*i->Get<ISubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL: return do_SubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL(*i->Get<ISubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL: return do_SubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL(*i->Get<ISubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL>(), args...);
case SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL: return do_SubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL(*i->Get<ISubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL>(), args...);
case SpvOpSubgroupAvcImeConvertToMceResultINTEL: return do_SubgroupAvcImeConvertToMceResultINTEL(*i->Get<ISubgroupAvcImeConvertToMceResultINTEL>(), args...);
case SpvOpSubgroupAvcImeGetSingleReferenceStreaminINTEL: return do_SubgroupAvcImeGetSingleReferenceStreaminINTEL(*i->Get<ISubgroupAvcImeGetSingleReferenceStreaminINTEL>(), args...);
case SpvOpSubgroupAvcImeGetDualReferenceStreaminINTEL: return do_SubgroupAvcImeGetDualReferenceStreaminINTEL(*i->Get<ISubgroupAvcImeGetDualReferenceStreaminINTEL>(), args...);
case SpvOpSubgroupAvcImeStripSingleReferenceStreamoutINTEL: return do_SubgroupAvcImeStripSingleReferenceStreamoutINTEL(*i->Get<ISubgroupAvcImeStripSingleReferenceStreamoutINTEL>(), args...);
case SpvOpSubgroupAvcImeStripDualReferenceStreamoutINTEL: return do_SubgroupAvcImeStripDualReferenceStreamoutINTEL(*i->Get<ISubgroupAvcImeStripDualReferenceStreamoutINTEL>(), args...);
case SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL: return do_SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL(*i->Get<ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL>(), args...);
case SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL: return do_SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL(*i->Get<ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL>(), args...);
case SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL: return do_SubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL(*i->Get<ISubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL>(), args...);
case SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL: return do_SubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL(*i->Get<ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL>(), args...);
case SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL: return do_SubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL(*i->Get<ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL>(), args...);
case SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL: return do_SubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL(*i->Get<ISubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL>(), args...);
case SpvOpSubgroupAvcImeGetBorderReachedINTEL: return do_SubgroupAvcImeGetBorderReachedINTEL(*i->Get<ISubgroupAvcImeGetBorderReachedINTEL>(), args...);
case SpvOpSubgroupAvcImeGetTruncatedSearchIndicationINTEL: return do_SubgroupAvcImeGetTruncatedSearchIndicationINTEL(*i->Get<ISubgroupAvcImeGetTruncatedSearchIndicationINTEL>(), args...);
case SpvOpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL: return do_SubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL(*i->Get<ISubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL>(), args...);
case SpvOpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL: return do_SubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL(*i->Get<ISubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL>(), args...);
case SpvOpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL: return do_SubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL(*i->Get<ISubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL>(), args...);
case SpvOpSubgroupAvcFmeInitializeINTEL: return do_SubgroupAvcFmeInitializeINTEL(*i->Get<ISubgroupAvcFmeInitializeINTEL>(), args...);
case SpvOpSubgroupAvcBmeInitializeINTEL: return do_SubgroupAvcBmeInitializeINTEL(*i->Get<ISubgroupAvcBmeInitializeINTEL>(), args...);
case SpvOpSubgroupAvcRefConvertToMcePayloadINTEL: return do_SubgroupAvcRefConvertToMcePayloadINTEL(*i->Get<ISubgroupAvcRefConvertToMcePayloadINTEL>(), args...);
case SpvOpSubgroupAvcRefSetBidirectionalMixDisableINTEL: return do_SubgroupAvcRefSetBidirectionalMixDisableINTEL(*i->Get<ISubgroupAvcRefSetBidirectionalMixDisableINTEL>(), args...);
case SpvOpSubgroupAvcRefSetBilinearFilterEnableINTEL: return do_SubgroupAvcRefSetBilinearFilterEnableINTEL(*i->Get<ISubgroupAvcRefSetBilinearFilterEnableINTEL>(), args...);
case SpvOpSubgroupAvcRefEvaluateWithSingleReferenceINTEL: return do_SubgroupAvcRefEvaluateWithSingleReferenceINTEL(*i->Get<ISubgroupAvcRefEvaluateWithSingleReferenceINTEL>(), args...);
case SpvOpSubgroupAvcRefEvaluateWithDualReferenceINTEL: return do_SubgroupAvcRefEvaluateWithDualReferenceINTEL(*i->Get<ISubgroupAvcRefEvaluateWithDualReferenceINTEL>(), args...);
case SpvOpSubgroupAvcRefEvaluateWithMultiReferenceINTEL: return do_SubgroupAvcRefEvaluateWithMultiReferenceINTEL(*i->Get<ISubgroupAvcRefEvaluateWithMultiReferenceINTEL>(), args...);
case SpvOpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL: return do_SubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL(*i->Get<ISubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL>(), args...);
case SpvOpSubgroupAvcRefConvertToMceResultINTEL: return do_SubgroupAvcRefConvertToMceResultINTEL(*i->Get<ISubgroupAvcRefConvertToMceResultINTEL>(), args...);
case SpvOpSubgroupAvcSicInitializeINTEL: return do_SubgroupAvcSicInitializeINTEL(*i->Get<ISubgroupAvcSicInitializeINTEL>(), args...);
case SpvOpSubgroupAvcSicConfigureSkcINTEL: return do_SubgroupAvcSicConfigureSkcINTEL(*i->Get<ISubgroupAvcSicConfigureSkcINTEL>(), args...);
case SpvOpSubgroupAvcSicConfigureIpeLumaINTEL: return do_SubgroupAvcSicConfigureIpeLumaINTEL(*i->Get<ISubgroupAvcSicConfigureIpeLumaINTEL>(), args...);
case SpvOpSubgroupAvcSicConfigureIpeLumaChromaINTEL: return do_SubgroupAvcSicConfigureIpeLumaChromaINTEL(*i->Get<ISubgroupAvcSicConfigureIpeLumaChromaINTEL>(), args...);
case SpvOpSubgroupAvcSicGetMotionVectorMaskINTEL: return do_SubgroupAvcSicGetMotionVectorMaskINTEL(*i->Get<ISubgroupAvcSicGetMotionVectorMaskINTEL>(), args...);
case SpvOpSubgroupAvcSicConvertToMcePayloadINTEL: return do_SubgroupAvcSicConvertToMcePayloadINTEL(*i->Get<ISubgroupAvcSicConvertToMcePayloadINTEL>(), args...);
case SpvOpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL: return do_SubgroupAvcSicSetIntraLumaShapePenaltyINTEL(*i->Get<ISubgroupAvcSicSetIntraLumaShapePenaltyINTEL>(), args...);
case SpvOpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL: return do_SubgroupAvcSicSetIntraLumaModeCostFunctionINTEL(*i->Get<ISubgroupAvcSicSetIntraLumaModeCostFunctionINTEL>(), args...);
case SpvOpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL: return do_SubgroupAvcSicSetIntraChromaModeCostFunctionINTEL(*i->Get<ISubgroupAvcSicSetIntraChromaModeCostFunctionINTEL>(), args...);
case SpvOpSubgroupAvcSicSetBilinearFilterEnableINTEL: return do_SubgroupAvcSicSetBilinearFilterEnableINTEL(*i->Get<ISubgroupAvcSicSetBilinearFilterEnableINTEL>(), args...);
case SpvOpSubgroupAvcSicSetSkcForwardTransformEnableINTEL: return do_SubgroupAvcSicSetSkcForwardTransformEnableINTEL(*i->Get<ISubgroupAvcSicSetSkcForwardTransformEnableINTEL>(), args...);
case SpvOpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL: return do_SubgroupAvcSicSetBlockBasedRawSkipSadINTEL(*i->Get<ISubgroupAvcSicSetBlockBasedRawSkipSadINTEL>(), args...);
case SpvOpSubgroupAvcSicEvaluateIpeINTEL: return do_SubgroupAvcSicEvaluateIpeINTEL(*i->Get<ISubgroupAvcSicEvaluateIpeINTEL>(), args...);
case SpvOpSubgroupAvcSicEvaluateWithSingleReferenceINTEL: return do_SubgroupAvcSicEvaluateWithSingleReferenceINTEL(*i->Get<ISubgroupAvcSicEvaluateWithSingleReferenceINTEL>(), args...);
case SpvOpSubgroupAvcSicEvaluateWithDualReferenceINTEL: return do_SubgroupAvcSicEvaluateWithDualReferenceINTEL(*i->Get<ISubgroupAvcSicEvaluateWithDualReferenceINTEL>(), args...);
case SpvOpSubgroupAvcSicEvaluateWithMultiReferenceINTEL: return do_SubgroupAvcSicEvaluateWithMultiReferenceINTEL(*i->Get<ISubgroupAvcSicEvaluateWithMultiReferenceINTEL>(), args...);
case SpvOpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL: return do_SubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL(*i->Get<ISubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL>(), args...);
case SpvOpSubgroupAvcSicConvertToMceResultINTEL: return do_SubgroupAvcSicConvertToMceResultINTEL(*i->Get<ISubgroupAvcSicConvertToMceResultINTEL>(), args...);
case SpvOpSubgroupAvcSicGetIpeLumaShapeINTEL: return do_SubgroupAvcSicGetIpeLumaShapeINTEL(*i->Get<ISubgroupAvcSicGetIpeLumaShapeINTEL>(), args...);
case SpvOpSubgroupAvcSicGetBestIpeLumaDistortionINTEL: return do_SubgroupAvcSicGetBestIpeLumaDistortionINTEL(*i->Get<ISubgroupAvcSicGetBestIpeLumaDistortionINTEL>(), args...);
case SpvOpSubgroupAvcSicGetBestIpeChromaDistortionINTEL: return do_SubgroupAvcSicGetBestIpeChromaDistortionINTEL(*i->Get<ISubgroupAvcSicGetBestIpeChromaDistortionINTEL>(), args...);
case SpvOpSubgroupAvcSicGetPackedIpeLumaModesINTEL: return do_SubgroupAvcSicGetPackedIpeLumaModesINTEL(*i->Get<ISubgroupAvcSicGetPackedIpeLumaModesINTEL>(), args...);
case SpvOpSubgroupAvcSicGetIpeChromaModeINTEL: return do_SubgroupAvcSicGetIpeChromaModeINTEL(*i->Get<ISubgroupAvcSicGetIpeChromaModeINTEL>(), args...);
case SpvOpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL: return do_SubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL(*i->Get<ISubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL>(), args...);
case SpvOpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL: return do_SubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL(*i->Get<ISubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL>(), args...);
case SpvOpSubgroupAvcSicGetInterRawSadsINTEL: return do_SubgroupAvcSicGetInterRawSadsINTEL(*i->Get<ISubgroupAvcSicGetInterRawSadsINTEL>(), args...);
      default:;
    }
    return do_missing(*i, args...);
  }
};

