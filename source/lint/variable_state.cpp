#include "source/lint/variable_state.h"

#include <iostream>
#include <ostream>
#include <variant>
#include <vector>

namespace spvtools {
namespace lint {
namespace uninitialized_variables {

Initialized init_state_difference(const Initialized& a, const Initialized& b) {
  if (b == Initialized::No) {
    return a;
  }
  if (a == b || b == Initialized::Yes) {
    return Initialized::No;
  }
  assert(b == Initialized::Unknown);
  if (a == Initialized::Yes) {
    return Initialized::Yes;
  }
  return Initialized::Unknown;
}

std::optional<State> State::TryDifference(const State& other) const {
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
  // Zero
  if (other.MaxState() == Initialized::No) {
    return *this;
  }
  // Top
  if (this->MaxState() <= other.MinState()) {
    return State::NewLeaf(Initialized::No);
  }
  if (const std::optional<Initialized> sr = get_if_single_state(other)) {
    if (const std::optional<Initialized> sl = get_if_single_state(*this)) {
      return State::NewLeaf(init_state_difference(*sl, *sr));
    }
    // Tree - Leaf
    const StateComp* cl = std::get_if<StateComp>(&inner_);
    assert(cl != nullptr);
    std::vector<State> copy;
    for (const State& component : cl->components) {
      const std::optional<State> r = component.TryDifference(other);
      if (!r.has_value()) {
        return std::nullopt;
      }
      copy.push_back(*r);
    }
    return State(StateComp{copy});
  }
  const StateComp* cr = std::get_if<StateComp>(&other.inner_);
  assert(cr != nullptr);
  if (const std::optional<Initialized> sl = get_if_single_state(*this)) {
    // Leaf - Tree
    std::vector<State> result_components;
    for (size_t i = 0; i < cr->components.size(); ++i) {
      const std::optional<State> r = this->TryDifference(cr->components.at(i));
      if (!r.has_value()) {
        return std::nullopt;
      }
      result_components.push_back(*r);
    }
    return State(StateComp{result_components});
  }
  const StateComp* cl = std::get_if<StateComp>(&inner_);
  assert(cr != nullptr);
  if (cl->components.size() != cr->components.size()) {
    return std::nullopt;
  };

  std::vector<State> result_components;
  for (size_t i = 0; i < cl->components.size(); ++i) {
    const std::optional<State> r =
        cl->components.at(i).TryDifference(cr->components.at(i));
    if (!r.has_value()) {
      return std::nullopt;
    }
    result_components.push_back(*r);
  }
  return State(StateComp{result_components});
}

Initialized State::MinState() const {
  Initialized min = Initialized::Yes;
  if (const StateLeaf* leaf = std::get_if<StateLeaf>(&inner_)) {
    min = leaf->initialized;
  } else if (const StateCompUnknownLength* comp_unk =
                 std::get_if<StateCompUnknownLength>(&inner_)) {
    min = comp_unk->initialized;
  } else if (const StateComp* comp = std::get_if<StateComp>(&inner_)) {
    for (const State& component : comp->components) {
      min = std::min(min, component.MinState());
      if (min == Initialized::No) {
        break;
      }
    }
  } else {
    assert(false && "unreachable");
  }
  return min;
}

Initialized State::MaxState() const {
  Initialized max = Initialized::No;
  if (const StateLeaf* leaf = std::get_if<StateLeaf>(&inner_)) {
    max = leaf->initialized;
  } else if (const StateCompUnknownLength* comp_unk =
                 std::get_if<StateCompUnknownLength>(&inner_)) {
    max = comp_unk->initialized;
  } else if (const StateComp* comp = std::get_if<StateComp>(&inner_)) {
    for (const State& component : comp->components) {
      max = std::max(max, component.MaxState());
      if (max == Initialized::Yes) {
        break;
      }
    }
  } else {
    assert(false && "unreachable");
  }
  return max;
}

std::ostream& operator<< (std::ostream& stream, const Initialized& i) {
  switch(i) {
    case Initialized::No: 
      stream << "N";
      break;
    case Initialized::Unknown: 
      stream << "U";
      break;
    case Initialized::Yes: 
      stream << "Y";
      break;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const State& s) {
  if (const Initialized* leaf = s.GetIfLeaf()) {
    stream << *leaf;
  } else if (const Initialized* comp_unk = s.GetIfCompositeUnknownLength()) {
    stream << "UnkLen [ " << *comp_unk << " ]";
  } else if (const std::vector<State>* comp = s.GetIfComposite()) {
    stream << "{";
    for (size_t i = 0; i < comp->size(); ++i) {
      stream << comp->at(i);
      if (i < comp->size() - 1) {
        stream << ", ";
      }
    }
    stream << "}";
  }
  return stream;
}

std::optional<State> State::TryNodeWiseMinMax(const State& other,
                                              bool do_min) const {
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
      if ((do_min && *sl <= *sr) || (!do_min && *sl >= *sr)) {
        return *this;
      }
      return other;
    }
    const StateComp* cr = std::get_if<StateComp>(&other.inner_);
    assert(cr != nullptr);
    if ((do_min && *sl <= other.MinState()) ||
        (!do_min && *sl >= other.MaxState())) {
      return *this;
    }
    if ((do_min && *sl >= other.MaxState()) ||
        (!do_min && *sl <= other.MinState())) {
      return other;
    }
    std::vector<State> copy;
    for (const State& component : cr->components) {
      const std::optional<State> r = this->TryNodeWiseMinMax(component, do_min);
      if (!r.has_value()) {
        return std::nullopt;
      }
      copy.push_back(*r);
    }
    return State(StateComp{copy});
  }
  const StateComp* cl = std::get_if<StateComp>(&this->inner_);
  assert(cl != nullptr);
  if (const std::optional<Initialized> sr = get_if_single_state(other)) {
    return other.TryNodeWiseMinMax(*this, do_min);
  }
  const StateComp* cr = std::get_if<StateComp>(&other.inner_);
  assert(cr != nullptr);
  if (cl->components.size() != cr->components.size()) {
    return std::nullopt;
  };

  if ((do_min && MaxState() <= other.MinState()) ||
      (!do_min && other.MaxState() <= MinState())) {
    return *this;
  }
  if ((do_min && other.MaxState() <= MinState()) ||
      (!do_min && MaxState() <= other.MinState())) {
    return other;
  }
  std::vector<State> copy;
  for (size_t i = 0; i < cl->components.size(); ++i) {
    const std::optional<State> r =
        cl->components.at(i).TryNodeWiseMinMax(cr->components.at(i), do_min);
    if (!r.has_value()) {
      return std::nullopt;
    }
    copy.push_back(*r);
  }
  return State(StateComp{copy});
}

VarStateMap VarStateMap::Union(const VarStateMap& other) const {
  if (states_ == other.states_) {
    return *this;
  }

  if (GreaterOrEqual(other)) {
    return *this;
  }
  if (other.GreaterOrEqual(*this)) {
    return other;
  }

  VarStateMap copy = Clone();

  for (auto& b_it : *other.states_) {
    const uint32_t var_id = b_it.first;
    const auto& found_a = states_->find(var_id);
    if (found_a == states_->cend()) {
      copy.states_->insert_or_assign(var_id, b_it.second);
    } else {
      copy.states_->insert_or_assign(
          b_it.first, b_it.second.TryUnion(found_a->second).value());
    }
  }
  return VarStateMap(copy);
}

VarStateMap VarStateMap::Intersect(const VarStateMap& other) const {
  if (states_ == other.states_) {
    return *this;
  }

  VarStateMap result;

  for (auto& a_it : *states_) {
    const uint32_t var_id = a_it.first;
    const State& a_state = a_it.second;
    const auto& found_b = other.states_->find(var_id);

    if (found_b != other.states_->end()) {
      result.states_->insert_or_assign(var_id,
                                       a_state.TryIntersect(found_b->second).value());
    }
  }
  return result;
}

VarStateMap VarStateMap::Difference(const VarStateMap& other) const {
  if (states_ == other.states_ || other.GreaterOrEqual(*this)) {
    return VarStateMap();
  }
  VarStateMap result = Clone();
  for (auto& a_it : *states_) {
    const uint32_t var_id = a_it.first;
    const State& a_state = a_it.second;
    const auto& found_b = other.states_->find(var_id);

    if (found_b != other.states_->end()) {
      result.states_->insert_or_assign(
          a_it.first, a_state.TryDifference(found_b->second).value());
    }
  }
  return result;
}

VarStateMap VarStateMap::Union(uint32_t var_id, const State& state) const {
  const auto& found = states_->find(var_id);
  if (found != states_->cend() &&
      found->second.TryAllGreaterOrEqual(state).value()) {
    return *this;
  }
  VarStateMap result = Clone();
  if (found == states_->cend()) {
    result.states_->insert({var_id, state});
  } else {
    result.states_->insert_or_assign(var_id,
                                     found->second.TryUnion(state).value());
  }
  return result;
}

VarStateMap VarStateMap::ClampAll(Initialized to_max) const {
  std::shared_ptr<StateMap> result_map = std::make_shared<StateMap>();
  for (const auto& it : *states_) {
    result_map->insert(
        {it.first, it.second.TryIntersect(State::NewLeaf(to_max)).value()});
  }
  return VarStateMap(result_map);
}

bool VarStateMap::Equals(const VarStateMap& other) const {
  if (states_ == other.states_) {
    return true;
  }
  for (const auto& it : *states_) {
    const auto& found = other.states_->find(it.first);
    if (found == other.states_->cend() ||
        !found->second.TryEquals(it.second).value()) {
      return false;
    }
  }
  for (const auto& it : *other.states_) {
    const auto& found = states_->find(it.first);
    if (found == states_->cend()) {
      return false;
    }
  }
  return true;
}

bool VarStateMap::GreaterOrEqual(const VarStateMap& other) const {
  if (states_ == other.states_) {
    return true;
  }
  for (const auto& it : *other.states_) {
    const auto& found = states_->find(it.first);
    if (found == states_->cend() ||
        !found->second.TryAllGreaterOrEqual(it.second).value()) {
      return false;
    }
  }
  return true;
}

std::ostream& operator<<(std::ostream& stream, const VarStateMap& vs) {
  for (const auto& it : *vs.states_) {
    stream << it.first << ": " << it.second << "\n";
  }
  return stream;
}

}  // namespace uninitialized_variables
}  // namespace lint
}  // namespace spvtools
