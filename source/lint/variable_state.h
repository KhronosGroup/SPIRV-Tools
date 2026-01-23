#ifndef SOURCE_LINT_VARIABLE_STATE_H_
#define SOURCE_LINT_VARIABLE_STATE_H_

#include <cassert>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace spvtools {
namespace lint {
namespace uninitialized_variables {

enum class Initialized : uint8_t {
  // Known to be uninitialized in 1 or more paths
  No = 0,
  // Array-like has been stored to with dynamic indexing: initialization is not
  // known statically,
  // but assumed to be complete. This excludes false positive flags, but
  // produces false negatives.
  Unknown = 1,
  // Known to always be initialized
  Yes = 2,
};

std::ostream& operator<<(std::ostream& stream, const Initialized& i);

// An operand to OpAccessChain
struct AccessChainField {
  // If is_constant: literal field index, otherwise result_id of index
  uint32_t value;
  // Whether the field index is a statically known constant.
  bool is_constant;
};

class State;
struct StateComp;
struct StateCompUnknownLength;
struct StateLeaf {
  Initialized initialized;
};

using AnyState = std::variant<StateLeaf, StateComp, StateCompUnknownLength>;

struct StateComp {
  std::vector<State> components;
};

struct StateCompUnknownLength {
  // Lower bound on all components' initialization (Yes > Unknown > No)
  Initialized initialized;
};

// Represents the intialization state of a variable as a tree
// (single leaf for a scalar, actual tree for composite types).
// Supports specialized/limited lattice-like operations between comparable
// trees.
class State {
 public:
  static State NewLeaf(Initialized initialized) {
    return State(StateLeaf{initialized});
  };

  // Create a new composite type with specialization constant as length.
  // Placeholder, doesn't really do much except checking that a variable has
  // ever been stored to at all before loading from it.
  static State NewCompositeUnknownLength(Initialized lower_bound) {
    return State(StateCompUnknownLength{lower_bound});
  };

  template <typename... Args>
  static State NewComposite(Args&&... args) {
    StateComp comp{std::vector<State>(std::forward<Args>(args)...)};
    assert(comp.components.size() > 0);
    return State(std::move(comp));
  }

  static State NewComposite(std::initializer_list<State> list) {
    StateComp comp{std::vector<State>(list)};
    assert(comp.components.size() > 0);
    return State(std::move(comp));
  }

  // Minimum leaf value of the subtree
  Initialized MinState() const;
  // Maximum leaf value of the subtree
  Initialized MaxState() const;

  const Initialized* GetIfLeaf() const {
    if (const StateLeaf* leaf = std::get_if<StateLeaf>(&inner_)) {
      return &leaf->initialized;
    }
    return nullptr;
  }

  const std::vector<State>* GetIfComposite() const {
    if (const StateComp* comp = std::get_if<StateComp>(&inner_)) {
      return &comp->components;
    }
    return nullptr;
  }

  const Initialized* GetIfCompositeUnknownLength() const {
    if (const StateCompUnknownLength* comp =
            std::get_if<StateCompUnknownLength>(&inner_)) {
      assert(false && "TODO");
      return &comp->initialized;
    }
    return nullptr;
  }

  // Difference-like operation: returns the smallest State D such that
  // this->TryMax(D) == other.
  std::optional<State> TryDifference(const State& other) const;

 private:
  explicit State(AnyState inner) : inner_(std::move(inner)) {}

  std::optional<State> TryNodeWiseMinMax(const State& other, bool do_min) const;

 public:
  // Returns nullopt if States are uncomparable (contain composites of different
  // lengths)
  std::optional<bool> TryEquals(const State& other) const {
    return NodeWiseAll(other,
                       [](Initialized a, Initialized b) { return a == b; });
  }

  // Nodewise >= between States, Subset/Included operation
  // Returns nullopt if States are uncomparable (contain composites of different
  // lengths)
  std::optional<bool> TryLessOrEqual(const State& other) const {
    return NodeWiseAll(other,
                       [](Initialized a, Initialized b) { return a <= b; });
  }

  // Nodewise >= between States, Superset/Includes operation
  // Returns nullopt if States are uncomparable (contain composites of different
  // lengths)
  std::optional<bool> TryAllGreaterOrEqual(const State& other) const {
    return NodeWiseAll(other,
                       [](Initialized a, Initialized b) { return a >= b; });
  }

  // Nodewise maximum between States/Union operation.
  // Returns nullopt if States are uncomparable (contain composites of different
  // lengths)
  std::optional<State> TryUnion(const State& other) const {
    return TryNodeWiseMinMax(other, false);
  }

  // Nodewise minimum between States/Intersect operation.
  // Returns nullopt if States are uncomparable (contain composites of different
  // lengths)
  std::optional<State> TryIntersect(const State& other) const {
    return TryNodeWiseMinMax(other, true);
  }

 private:
  // Only helper for Equals, LessOrEqual, Max, Min
  template <typename F>
  std::optional<bool> NodeWiseAll(const State& other, F f) const {
    const auto get_if_single_state =
        [](const State& s) -> std::optional<Initialized> {
      if (const StateLeaf* leaf = std::get_if<StateLeaf>(&s.inner_)) {
        return leaf->initialized;
      }
      if (const StateCompUnknownLength* comp_unk =
              std::get_if<StateCompUnknownLength>(&s.inner_)) {
        return comp_unk->initialized;
      }
      return std::nullopt;
    };
    if (const std::optional<Initialized> sl = get_if_single_state(*this)) {
      if (const std::optional<Initialized> sr = get_if_single_state(other)) {
        return f(*sl, *sr);
      }
      const StateComp* cr = std::get_if<StateComp>(&other.inner_);
      assert(cr != nullptr);
      for (const State& component : cr->components) {
        const std::optional<bool> r = NodeWiseAll(component, f);
        if (!r.has_value()) {
          return std::nullopt;
        } else if (!r.value()) {
          return false;
        }
      }
      return true;
    }
    const StateComp* cl = std::get_if<StateComp>(&this->inner_);
    assert(cl != nullptr);
    StateComp copy = *cl;
    if (const std::optional<Initialized> sr = get_if_single_state(other)) {
      for (const State& component : cl->components) {
        const std::optional<bool> r = component.NodeWiseAll(other, f);
        if (!r.has_value()) {
          return std::nullopt;
        } else if (!r.value()) {
          return false;
        }
      }
      return true;
    }
    const StateComp* cr = std::get_if<StateComp>(&other.inner_);
    assert(cr != nullptr);
    if (cl->components.size() != cr->components.size()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < cl->components.size(); ++i) {
      const std::optional<bool> r =
          cl->components.at(i).NodeWiseAll(cr->components.at(i), f);
      if (!r.has_value()) {
        return std::nullopt;
      } else if (!r.value()) {
        return false;
      }
    }
    return true;
  };
  State() = delete;
  AnyState inner_;
};

std::ostream& operator<<(std::ostream& stream, const State&);

// States of all variables at a point in time with simple copy-on-write semantics.
// Supports some specialized/limited lattice-like operations.
class VarStateMap {
  using StateMap = std::unordered_map<uint32_t, State>;

 public:
  VarStateMap() : states_(std::make_shared<StateMap>()) {}
  explicit VarStateMap(const std::shared_ptr<StateMap>& states) : states_(states) {}

  VarStateMap Clone() const {
    return VarStateMap(std::make_shared<StateMap>(*states_));
  }

  std::optional<State> Get(uint32_t var_id) const {
    const auto found = states_->find(var_id);
    if (found == states_->cend()) {
      return std::nullopt;
    }
    return found->second;
  };

  // Nodewise operations over all variables. 
  // States for variables contained in both operands *must* be comparable.

  VarStateMap Union(uint32_t var_id, const State& state) const;
  VarStateMap Union(const VarStateMap& other) const;
  VarStateMap Intersect(const VarStateMap& other) const;
  VarStateMap Difference(const VarStateMap& other) const;
  bool Equals(const VarStateMap& other) const;
  bool GreaterOrEqual(const VarStateMap& other) const;

  VarStateMap ClampAll(Initialized to_max) const;
  size_t Size() const { return states_->size(); }

  friend std::ostream& operator<<(std::ostream&, const VarStateMap&);
 private:
  std::shared_ptr<std::unordered_map<uint32_t, State>> states_;
};


}  // namespace uninitialized_variables
}  // namespace lint
}  // namespace spvtools

#endif  // SOURCE_LINT_VARIABLE_STATE_H_
