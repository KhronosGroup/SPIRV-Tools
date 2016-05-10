// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#ifndef LIBSPIRV_VALIDATE_H_
#define LIBSPIRV_VALIDATE_H_

#include <algorithm>
#include <array>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "assembly_grammar.h"
#include "binary.h"
#include "diagnostic.h"
#include "instruction.h"
#include "spirv-tools/libspirv.h"
#include "spirv_definition.h"
#include "table.h"

#define MSG(msg)                                        \
  do {                                                  \
    libspirv::message(__FILE__, size_t(__LINE__), msg); \
  } while (0)

#define SHOW(exp)                                               \
  do {                                                          \
    libspirv::message(__FILE__, size_t(__LINE__), #exp, (exp)); \
  } while (0)

// Structures

// Info about a result ID.
typedef struct spv_id_info_t {
  // Id value.
  uint32_t id;
  // Type id, or 0 if no type.
  uint32_t type_id;
  // Opcode of the instruction defining the id.
  SpvOp opcode;
  // Binary words of the instruction defining the id.
  std::vector<uint32_t> words;
} spv_id_info_t;

namespace libspirv {

void message(std::string file, size_t line, std::string name);

template <typename T>
void message(std::string file, size_t line, std::string name, T val) {
  std::cout << file << ":" << line << ": " << name << " " << val << std::endl;
}

/// This enum represents the sections of a SPIRV module. See section 2.4
/// of the SPIRV spec for additional details of the order. The enumerant values
/// are in the same order as the vector returned by GetModuleOrder
enum ModuleLayoutSection {
  kLayoutCapabilities,          // < Section 2.4 #1
  kLayoutExtensions,            // < Section 2.4 #2
  kLayoutExtInstImport,         // < Section 2.4 #3
  kLayoutMemoryModel,           // < Section 2.4 #4
  kLayoutEntryPoint,            // < Section 2.4 #5
  kLayoutExecutionMode,         // < Section 2.4 #6
  kLayoutDebug1,                // < Section 2.4 #7 > 1
  kLayoutDebug2,                // < Section 2.4 #7 > 2
  kLayoutAnnotations,           // < Section 2.4 #8
  kLayoutTypes,                 // < Section 2.4 #9
  kLayoutFunctionDeclarations,  // < Section 2.4 #10
  kLayoutFunctionDefinitions    // < Section 2.4 #11
};

enum class FunctionDecl {
  kFunctionDeclUnknown,      // < Unknown function declaration
  kFunctionDeclDeclaration,  // < Function declaration
  kFunctionDeclDefinition    // < Function definition
};

class ValidationState_t;
class Function;

// This class represents a basic block in a SPIR-V module
class BasicBlock {
 public:
  /// Constructor for a BasicBlock
  ///
  /// @param[in] id The ID of the basic block
  /// @param[in] module A reference of the module of the basic block
  BasicBlock(uint32_t id);

  /// Returns the id of the BasicBlock
  uint32_t get_id() const { return id_; }

  /// Returns the predecessors of the BasicBlock
  const std::vector<BasicBlock*>& get_predecessors() const {
    return predecessors_;
  }

  /// Returns the predecessors of the BasicBlock
  std::vector<BasicBlock*>& get_predecessors() { return predecessors_; }

  /// Returns the successors of the BasicBlock
  const std::vector<BasicBlock*>& get_successors() const { return successors_; }

  /// Returns the successors of the BasicBlock
  std::vector<BasicBlock*>& get_successors() { return successors_; }

  /// Returns true if the  block should be reachable in the CFG
  bool is_reachable() const { return reachable_; }

  void set_reachability(bool reachability) { reachable_ = reachability; }

  /// Sets the immedate dominator of this basic block
  ///
  /// @param[in] dom_block The dominator block
  void SetImmediateDominator(BasicBlock* dom_block);

  /// Returns the immedate dominator of this basic block
  BasicBlock* GetImmediateDominator();

  /// Returns the immedate dominator of this basic block
  const BasicBlock* GetImmediateDominator() const;

  /// Adds @p next as a successor of this BasicBlock
  void RegisterBranchWithoutSuccessor(SpvOp branch_instruction);

  /// Adds @p next as a successor of this BasicBlock
  void RegisterSuccessor(BasicBlock& next, SpvOp branch_instruction);

  /// Adds @p next BasicBlocks as successors of this BasicBlock
  void RegisterSuccessor(std::vector<BasicBlock*> next, SpvOp branch_instruction);

  /// Returns true if the id of the BasicBlock matches
  bool operator==(const BasicBlock& other) const { return other.id_ == id_; }

  /// Returns true if the id of the BasicBlock matches
  bool operator==(const uint32_t& id) const { return id == id_; }

  /// @brief A BasicBlock dominator iterator class
  ///
  /// This iterator will iterate over the dominators of the block
  class DominatorIterator
      : public std::iterator<std::forward_iterator_tag, BasicBlock*> {
   public:
    /// @brief Constructs the end of dominator iterator
    ///
    /// This will create an iterator which will represent the element
    /// before the root node of the dominator tree
    DominatorIterator();

    /// @brief Constructs an iterator for the given block which points to
    ///        @p block
    ///
    /// @param block The block which is referenced by the iterator
    DominatorIterator(const BasicBlock* block);

    /// @brief Advances the iterator
    DominatorIterator& operator++();

    /// @brief Returns the current element
    const BasicBlock*& operator*();

    friend bool operator==(const DominatorIterator& lhs,
                           const DominatorIterator& rhs);

   private:
    const BasicBlock* current_;
  };

  /// Returns an iterator which points to the current block
  const DominatorIterator dom_begin() const;
  DominatorIterator dom_begin();

  /// Returns an iterator which points to one element past the first block
  const DominatorIterator dom_end() const;
  DominatorIterator dom_end();

 private:
  /// Id of the BasicBlock
  const uint32_t id_;

  /// Pointer to the immediate dominator of the BasicBlock
  BasicBlock* immediate_dominator_;

  /// The set of predecessors of the BasicBlock
  std::vector<BasicBlock*> predecessors_;

  /// The set of successors of the BasicBlock
  std::vector<BasicBlock*> successors_;

  /// The function which contains this block
  Function* function_;

  SpvOp branch_instruction_;

  bool reachable_;
};

/// @brief Returns true if the iterators point to the same element or if both
///        iterators point to the @p dom_end block
bool operator==(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs);

/// @brief Returns true if the iterators point to different elements and they
///        do not both point to the @p dom_end block
bool operator!=(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs);

/// @brief This class tracks the CFG constructs as defined in the SPIR-V spec
class CFConstruct {
  // Universal Limit of ResultID + 1
  static const uint32_t kInitialValue = 0x400000;

 public:
  CFConstruct(BasicBlock* header_block, BasicBlock* merge_block,
              BasicBlock* continue_block = nullptr)
      : header_block_(header_block),
        merge_block_(merge_block),
        continue_block_(continue_block) {}

  const BasicBlock* get_header() const { return header_block_; }
  const BasicBlock* get_merge() const { return merge_block_; }
  const BasicBlock* get_continue() const { return continue_block_; }

 private:
  BasicBlock* header_block_;    ///< The header block of a loop or selection
  BasicBlock* merge_block_;     ///< The merge block of a loop or selection
  BasicBlock* continue_block_;  ///< The continue block of a loop block
};

// This class manages all function declaration and definitions in a module. It
// handles the state and id information while parsing a function in the SPIR-V
// binary.
class Function {
 public:
  Function(uint32_t id, uint32_t result_type_id,
           SpvFunctionControlMask function_control, uint32_t function_type_id,
           ValidationState_t& module);

  /// Registers a function parameter in the current function
  /// @return Returns SPV_SUCCESS if the call was successful
  spv_result_t RegisterFunctionParameter(uint32_t id, uint32_t type_id);

  /// Sets the declaration type of the current function
  /// @return Returns SPV_SUCCESS if the call was successful
  spv_result_t RegisterSetFunctionDeclType(FunctionDecl type);

  // Registers a block in the current function. Subsequent block instructions
  // will target this block
  // @param id The ID of the label of the block
  /// @return Returns SPV_SUCCESS if the call was successful
  spv_result_t RegisterBlock(uint32_t id, bool is_definition = true);

  /// Registers a variable in the current block
  ///
  /// @param[in] type_id The type ID of the varaible
  /// @param[in] id      The ID of the varaible
  /// @param[in] storage The storage of the variable
  /// @param[in] init_id The initializer ID of the variable
  ///
  /// @return Returns SPV_SUCCESS if the call was successful
  spv_result_t RegisterBlockVariable(uint32_t type_id, uint32_t id,
                                     SpvStorageClass storage, uint32_t init_id);

  /// Registers a loop merge construct in the function
  ///
  /// @param[in] merge_id The merge block ID of the loop
  /// @param[in] continue_id The continue block ID of the loop
  ///
  /// @return Returns SPV_SUCCESS if the call was successful
  spv_result_t RegisterLoopMerge(uint32_t merge_id, uint32_t continue_id);

  /// Registers a selection merge construct in the function
  /// @return Returns SPV_SUCCESS if the call was successful
  spv_result_t RegisterSelectionMerge(uint32_t merge_id);

  /// Registers the end of the block
  void RegisterBlockEnd(SpvOp branch_instruction);

  /// Registers the end of the block
  void RegisterBlockEnd(uint32_t next_id, SpvOp branch_instruction);

  /// Registers the end of the block
  void RegisterBlockEnd(std::vector<uint32_t> next_list, SpvOp branch_instruction);

  /// Returns true if the \p merge_block_id is a merge block
  bool IsMergeBlock(uint32_t merge_block_id) const;

  /// Returns true if the \p id is the first block of this function
  bool IsFirstBlock(uint32_t id) const;

  /// Returns the first block of the current function
  const BasicBlock* get_first_block() const;

  /// Returns the first block of the current function
  BasicBlock* get_first_block();

  /// Returns a vector of all the blocks in the function
  const std::vector<BasicBlock*>& get_blocks() const;

  /// Returns a vector of all the blocks in the function
  std::vector<BasicBlock*>& get_blocks();

  /// Returns a list of all the cfg constructs in the function
  const std::list<CFConstruct>& get_constructs() const;

  /// Returns a list of all the cfg constructs in the function
  std::list<CFConstruct>& get_constructs();

  // Returns the number of blocks in the current function being parsed
  size_t get_block_count() const;

  /// Returns the id of the funciton
  uint32_t get_id() const { return id_; };

  // Returns the number of blocks in the current function being parsed
  size_t get_undefined_block_count() const;
  const std::unordered_set<uint32_t>& get_undefined_blocks() const {
    return undefined_blocks_;
  }

  /// Returns true if called after a label instruction but before a branch
  /// instruction
  bool in_block() const;

  /// Returns the block that is currently being parsed in the binary
  BasicBlock& get_current_block();

  /// Returns the block that is currently being parsed in the binary
  const BasicBlock& get_current_block() const;

  /// Prints a GraphViz digraph of the CFG of the current funciton
  void printDotGraph() const;

  /// Prints a directed graph of the CFG of the current funciton
  void printBlocks() const;

 private:
  /// Parent module
  ValidationState_t& module_;

  /// The result id of the OpLabel that defined this block
  uint32_t id_;

  /// The type of the function
  uint32_t function_type_id_;

  /// The type of the return value
  uint32_t result_type_id_;

  /// The control fo the funciton
  SpvFunctionControlMask function_control_;

  /// The type of declaration of each function
  FunctionDecl declaration_type_;

  /// The beginning of the block of functions
  std::unordered_map<uint32_t, BasicBlock> blocks_;

  /// A list of blocks in the order they appeared in the binary
  std::vector<BasicBlock*> ordered_blocks_;

  /// Blocks which are forward referenced by blocks but not defined
  std::unordered_set<uint32_t> undefined_blocks_;

  /// The block that is currently being parsed
  BasicBlock* current_block_;

  /// The constructs that are available in this function
  std::list<CFConstruct> cfg_constructs_;

  /// The variable IDs of the functions
  std::vector<uint32_t> variable_ids_;

  /// The function parameter ids of the functions
  std::vector<uint32_t> parameter_ids_;
};

class ValidationState_t {
 public:
  ValidationState_t(spv_diagnostic* diagnostic,
                    const spv_const_context context);

  // Forward declares the id in the module
  spv_result_t forwardDeclareId(uint32_t id);

  // Removes a forward declared ID if it has been defined
  spv_result_t removeIfForwardDeclared(uint32_t id);

  // Assigns a name to an ID
  void assignNameToId(uint32_t id, std::string name);

  // Returns a string representation of the ID in the format <id>[Name] where
  // the <id> is the numeric valid of the id and the Name is a name assigned by
  // the OpName instruction
  std::string getIdName(uint32_t id) const;

  std::string getIdOrName(uint32_t id) const;

  // Returns the number of ID which have been forward referenced but not defined
  size_t unresolvedForwardIdCount() const;

  // Returns a list of unresolved forward ids.
  std::vector<uint32_t> unresolvedForwardIds() const;

  // Returns true if the id has been defined
  bool isDefinedId(uint32_t id) const;

  // Increments the instruction count. Used for diagnostic
  int incrementInstructionCount();

  // Returns the current layout section which is being processed
  ModuleLayoutSection getLayoutSection() const;

  // Increments the module_layout_order_section_
  void progressToNextLayoutSectionOrder();

  // Determines if the op instruction is part of the current section
  bool isOpcodeInCurrentLayoutSection(SpvOp op);

  libspirv::DiagnosticStream diag(spv_result_t error_code) const;

  // Returns the function states
  std::list<Function>& get_functions();

  // Returns the function states
  Function& get_current_function();

  // Returns true if the called after a function instruction but before the
  // function end instruction
  bool in_function_body() const;

  // Returns true if called after a label instruction but before a branch
  // instruction
  bool in_block() const;

  // Keeps track of ID definitions and uses.
  class UseDefTracker {
   public:
    void AddDef(const spv_id_info_t& def) { defs_[def.id] = def; }

    void AddUse(uint32_t id) { uses_.insert(id); }

    // Finds id's def, if it exists.  If found, returns <true, def>.  Otherwise,
    // returns <false, something>.
    std::pair<bool, spv_id_info_t> FindDef(uint32_t id) const {
      if (defs_.count(id) == 0) {
        return std::make_pair(false, spv_id_info_t{});
      } else {
        // We are in a const function, so we cannot use defs.operator[]().
        // Luckily we know the key exists, so defs_.at() won't throw an
        // exception.
        return std::make_pair(true, defs_.at(id));
      }
    }

    // Returns uses of IDs lacking defs.
    std::unordered_set<uint32_t> FindUsesWithoutDefs() const {
      auto diff = uses_;
      for (const auto d : defs_) diff.erase(d.first);
      return diff;
    }

   private:
    std::unordered_set<uint32_t> uses_;
    std::unordered_map<uint32_t, spv_id_info_t> defs_;
  };

  UseDefTracker& usedefs() { return usedefs_; }
  const UseDefTracker& usedefs() const { return usedefs_; }

  std::vector<uint32_t>& entry_points() { return entry_points_; }
  const std::vector<uint32_t>& entry_points() const { return entry_points_; }

  // Registers the capability and its dependent capabilities
  void RegisterCapability(SpvCapability cap);

  // Registers the function in the module. Subsequent instructions will be
  // called against this function
  spv_result_t RegisterFunction(uint32_t id, uint32_t ret_type_id,
                                SpvFunctionControlMask function_control,
                                uint32_t function_type_id);

  // Register a function end instruction
  spv_result_t RegisterFunctionEnd();

  // Returns true if the capability is enabled in the module.
  bool hasCapability(SpvCapability cap) const;

  // Returns true if any of the capabilities are enabled.  Always true for
  // capabilities==0.
  bool HasAnyOf(spv_capability_mask_t capabilities) const;

  // Sets the addressing model of this module (logical/physical).
  void setAddressingModel(SpvAddressingModel am);

  // Returns the addressing model of this module, or Logical if uninitialized.
  SpvAddressingModel getAddressingModel() const;

  // Sets the memory model of this module.
  void setMemoryModel(SpvMemoryModel mm);

  // Returns the memory model of this module, or Simple if uninitialized.
  SpvMemoryModel getMemoryModel() const;

  AssemblyGrammar& grammar() { return grammar_; }

 private:
  spv_diagnostic* diagnostic_;
  // Tracks the number of instructions evaluated by the validator
  int instruction_counter_;

  // IDs which have been forward declared but have not been defined
  std::unordered_set<uint32_t> unresolved_forward_ids_;

  std::map<uint32_t, std::string> operand_names_;

  // The section of the code being processed
  ModuleLayoutSection current_layout_section_;

  std::list<Function> module_functions_;

  spv_capability_mask_t
      module_capabilities_;  // Module's declared capabilities.

  // Definitions and uses of all the IDs in the module.
  UseDefTracker usedefs_;

  // IDs that are entry points, ie, arguments to OpEntryPoint.
  std::vector<uint32_t> entry_points_;

  AssemblyGrammar grammar_;

  SpvAddressingModel addressing_model_;
  SpvMemoryModel memory_model_;

  // NOTE: See correspoding getter functions
  bool in_function_;
};

/// @brief Calculates dominator edges of a root basic block
///
/// This function calculates the dominator edges form a root BasicBlock. Uses
/// the dominator algorithm by Cooper et al.
///
/// @param[in] first_block the root or entry BasicBlock of a function
///
/// @return a set of dominator edges represented as a pair of blocks
std::vector<std::pair<BasicBlock*, BasicBlock*> > CalculateDominators(
    const BasicBlock& first_block);

/// @brief Performs the Control Flow Graph checks
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found. SPV_ERROR_INVALID_CFG otherwise
spv_result_t PerformCfgChecks(ValidationState_t& _);

// @brief Updates the immediate dominator for each of the block edges
//
// Updates the immediate dominator of the blocks for each of the edges
// provided by the @p dom_edges parameter
//
// @param[in,out] dom_edges The edges of the dominator tree
void UpdateImmediateDominators(
    std::vector<std::pair<BasicBlock*, BasicBlock*> >& dom_edges);

// @brief Prints all of the dominators of a BasicBlock
//
// @param[in] block The dominators of this block will be printed
void printDominatorList(BasicBlock& block);

}  // namespace libspirv

/// @brief Validate the ID usage of the instruction stream
///
/// @param[in] pInsts stream of instructions
/// @param[in] instCount number of instructions
/// @param[in] opcodeTable table of specified Opcodes
/// @param[in] operandTable table of specified operands
/// @param[in] usedefs use-def info from module parsing
/// @param[in,out] position current position in the stream
/// @param[out] pDiag contains diagnostic on failure
///
/// @return result code
spv_result_t spvValidateInstructionIDs(const spv_instruction_t* pInsts,
                                       const uint64_t instCount,
                                       const spv_opcode_table opcodeTable,
                                       const spv_operand_table operandTable,
                                       const spv_ext_inst_table extInstTable,
                                       const libspirv::ValidationState_t& state,
                                       spv_position position,
                                       spv_diagnostic* pDiag);

/// @brief Validate the ID's within a SPIR-V binary
///
/// @param[in] pInstructions array of instructions
/// @param[in] count number of elements in instruction array
/// @param[in] bound the binary header
/// @param[in] opcodeTable table of specified Opcodes
/// @param[in] operandTable table of specified operands
/// @param[in,out] position current word in the binary
/// @param[out] pDiagnostic contains diagnostic on failure
///
/// @return result code
spv_result_t spvValidateIDs(const spv_instruction_t* pInstructions,
                            const uint64_t count, const uint32_t bound,
                            const spv_opcode_table opcodeTable,
                            const spv_operand_table operandTable,
                            const spv_ext_inst_table extInstTable,
                            spv_position position, spv_diagnostic* pDiagnostic);

#define spvCheckReturn(expression) \
  if (spv_result_t error = (expression)) return error;

#endif  // LIBSPIRV_VALIDATE_H_
