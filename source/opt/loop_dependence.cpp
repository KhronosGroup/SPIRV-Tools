// Copyright (c) 2018 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "opt/loop_dependence.h"

#include <string>
#include <utility>
#include <vector>

#include "opt/instruction.h"
#include "opt/scalar_analysis.h"
#include "opt/scalar_analysis_nodes.h"

namespace spvtools {
namespace opt {

bool LoopDependenceAnalysis::GetDependence(const ir::Instruction* source,
                                           const ir::Instruction* destination,
                                           DistanceVector* distance_vector) {
  // Start off by finding and marking all the loops in |loops_| that are
  // irrelevant to the dependence analysis.
  MarkUnsusedDistanceEntriesAsIrrelevant(source, destination, distance_vector);

  ir::Instruction* source_access_chain = GetOperandDefinition(source, 0);
  ir::Instruction* destination_access_chain =
      GetOperandDefinition(destination, 0);

  // If the access chains aren't collecting from the same structure there is no
  // dependence.
  ir::Instruction* source_array = GetOperandDefinition(source_access_chain, 0);
  ir::Instruction* destination_array =
      GetOperandDefinition(destination_access_chain, 0);
  if (source_array != destination_array) {
    PrintDebug("Proved independence through different arrays.");
    return true;
  }

  // To handle multiple subscripts we must get every operand in the access
  // chains past the first.
  std::vector<ir::Instruction*> source_subscripts = GetSubscripts(source);
  std::vector<ir::Instruction*> destination_subscripts =
      GetSubscripts(destination);

  auto sets_of_subscripts =
      PartitionSubscripts(source_subscripts, destination_subscripts);

  auto first_coupled = std::partition(
      std::begin(sets_of_subscripts), std::end(sets_of_subscripts),
      [](const std::set<std::pair<ir::Instruction*, ir::Instruction*>>& set) {
        return set.size() == 1;
      });

  // Go through each subscript testing for independence.
  // If any subscript results in independence, we prove independence between the
  // load and store.
  // If we can't prove independence we store what information we can gather in
  // a DistanceVector.
  for (auto it = std::begin(sets_of_subscripts); it < first_coupled; ++it) {
    auto source_subscript = std::get<0>(*(*it).begin());
    auto destination_subscript = std::get<1>(*(*it).begin());

    SENode* source_node = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.AnalyzeInstruction(source_subscript));
    SENode* destination_node = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.AnalyzeInstruction(destination_subscript));

    // auto subscript = GetSubscriptForInstruction(source_subscript);
    auto subscript_pair = std::make_pair(source_node, destination_node);

    const ir::Loop* loop = GetLoopForSubscriptPair(&subscript_pair);
    if (loop) {
      if (!IsSupportedLoop(loop)) {
        PrintDebug(
            "GetDependence found an unsupported loop form. Assuming <=> for "
            "loop.");
        DistanceEntry* distance_entry =
            GetDistanceEntryForSubscriptPair(&subscript_pair, distance_vector);
        if (distance_entry) {
          distance_entry->direction = DistanceEntry::Directions::ALL;
        }
        continue;
      }
    }

    // If either node is simplified to a CanNotCompute we can't perform any
    // analysis so must assume <=> dependence and return.
    if (source_node->GetType() == SENode::CanNotCompute ||
        destination_node->GetType() == SENode::CanNotCompute) {
      // Record the <=> dependence if we can get a DistanceEntry
      PrintDebug(
          "GetDependence found source_node || destination_node as "
          "CanNotCompute. Abandoning evaluation for this subscript.");
      DistanceEntry* distance_entry =
          GetDistanceEntryForSubscriptPair(&subscript_pair, distance_vector);
      if (distance_entry) {
        distance_entry->direction = DistanceEntry::Directions::ALL;
      }
      continue;
    }

    // We have no induction variables so can apply a ZIV test.
    if (IsZIV(subscript_pair)) {
      PrintDebug("Found a ZIV subscript pair");
      if (ZIVTest(source_node, destination_node)) {
        PrintDebug("Proved independence with ZIVTest.");
        return true;
      }
    }

    // We have only one induction variable so should attempt an SIV test.
    if (IsSIV(subscript_pair)) {
      PrintDebug("Found a SIV subscript pair.");
      if (SIVTest(&subscript_pair, distance_vector)) {
        PrintDebug("Proved independence with SIVTest.");
        return true;
      }
    }

    // We have multiple induction variables so should attempt an MIV test.
    if (IsMIV(subscript_pair)) {
      // Currently not handled
      PrintDebug(
          "Found a MIV subscript pair. MIV is currently unhandled.\n"
          "Assuming dependence in all directions for this subscript pair.");
      continue;
    }
  }

  // We were unable to prove independence so must gather all of the direction
  // information we found.
  PrintDebug(
      "Couldn't prove independence.\n"
      "All possible direction information has been collected in the input "
      "DistanceVector.");

  return false;
}

bool LoopDependenceAnalysis::ZIVTest(SENode* source, SENode* destination) {
  PrintDebug("Performing ZIVTest");
  // If source == destination, dependence with direction = and distance 0.
  if (source == destination) {
    PrintDebug("ZIVTest found EQ dependence.");
    return false;
  } else {
    PrintDebug("ZIVTest found independence.");
    // Otherwise we prove independence.
    return true;
  }
}

bool LoopDependenceAnalysis::SIVTest(
    std::pair<SENode*, SENode*>* subscript_pair,
    DistanceVector* distance_vector) {
  DistanceEntry* distance_entry =
      GetDistanceEntryForSubscriptPair(subscript_pair, distance_vector);
  if (!distance_entry) {
    PrintDebug(
        "SIVTest could not find a DistanceEntry for subscript_pair. Exiting");
  }

  SENode* source_node = std::get<0>(*subscript_pair);
  SENode* destination_node = std::get<1>(*subscript_pair);

  int64_t source_induction_count = CountInductionVariables(source_node);
  int64_t destination_induction_count =
      CountInductionVariables(destination_node);

  // If the source node has no induction variables we can apply a
  // WeakZeroSrcTest.
  if (source_induction_count == 0) {
    PrintDebug("Found source has no induction variable.");
    if (WeakZeroSourceSIVTest(
            source_node, destination_node->AsSERecurrentNode(),
            destination_node->AsSERecurrentNode()->GetCoefficient(),
            distance_entry)) {
      PrintDebug("Proved independence with WeakZeroSourceSIVTest.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    }
  }

  // If the destination has no induction variables we can apply a
  // WeakZeroDestTest.
  if (destination_induction_count == 0) {
    PrintDebug("Found destination has no induction variable.");
    if (WeakZeroDestinationSIVTest(
            source_node->AsSERecurrentNode(), destination_node,
            source_node->AsSERecurrentNode()->GetCoefficient(),
            distance_entry)) {
      PrintDebug("Proved independence with WeakZeroDestinationSIVTest.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    }
  }

  // We now need to collect the SERecurrentExpr nodes from source and
  // destination. We do not handle cases where source or destination have
  // multiple SERecurrentExpr nodes.
  std::vector<SERecurrentNode*> source_recurrent_nodes =
      source_node->CollectRecurrentNodes();
  std::vector<SERecurrentNode*> destination_recurrent_nodes =
      destination_node->CollectRecurrentNodes();

  if (source_recurrent_nodes.size() == 1 &&
      destination_recurrent_nodes.size() == 1) {
    PrintDebug("Found source and destination have 1 induction variable.");
    SERecurrentNode* source_recurrent_expr = *source_recurrent_nodes.begin();
    SERecurrentNode* destination_recurrent_expr =
        *destination_recurrent_nodes.begin();

    // If the coefficients are identical we can apply a StrongSIVTest.
    if (source_recurrent_expr->GetCoefficient() ==
        destination_recurrent_expr->GetCoefficient()) {
      PrintDebug("Found source and destination share coefficient.");
      if (StrongSIVTest(source_node, destination_node,
                        source_recurrent_expr->GetCoefficient(),
                        distance_entry)) {
        PrintDebug("Proved independence with StrongSIVTest");
        distance_entry->dependence_information =
            DistanceEntry::DependenceInformation::DIRECTION;
        distance_entry->direction = DistanceEntry::Directions::NONE;
        return true;
      }
    }

    // If the coefficients are of equal magnitude and opposite sign we can
    // apply a WeakCrossingSIVTest.
    if (source_recurrent_expr->GetCoefficient() ==
        scalar_evolution_.CreateNegation(
            destination_recurrent_expr->GetCoefficient())) {
      PrintDebug("Found source coefficient = -destination coefficient.");
      if (WeakCrossingSIVTest(source_node, destination_node,
                              source_recurrent_expr->GetCoefficient(),
                              distance_entry)) {
        PrintDebug("Proved independence with WeakCrossingSIVTest");
        distance_entry->dependence_information =
            DistanceEntry::DependenceInformation::DIRECTION;
        distance_entry->direction = DistanceEntry::Directions::NONE;
        return true;
      }
    }
  }

  return false;
}

bool LoopDependenceAnalysis::StrongSIVTest(SENode* source, SENode* destination,
                                           SENode* coefficient,
                                           DistanceEntry* distance_entry) {
  PrintDebug("Performing StrongSIVTest.");
  // If both source and destination are SERecurrentNodes we can perform tests
  // based on distance.
  // If either source or destination contain value unknown nodes or if one or
  // both are not SERecurrentNodes we must attempt a symbolic test.
  std::vector<SEValueUnknown*> source_value_unknown_nodes =
      source->CollectValueUnknownNodes();
  std::vector<SEValueUnknown*> destination_value_unknown_nodes =
      destination->CollectValueUnknownNodes();
  if (source_value_unknown_nodes.size() > 0 ||
      destination_value_unknown_nodes.size() > 0) {
    PrintDebug(
        "StrongSIVTest found symbolics. Will attempt SymbolicStrongSIVTest.");
    return SymbolicStrongSIVTest(source, destination, coefficient,
                                 distance_entry);
  }

  if (!source->AsSERecurrentNode() || !destination->AsSERecurrentNode()) {
    PrintDebug(
        "StrongSIVTest could not simplify source and destination to "
        "SERecurrentNodes so will exit.");
    distance_entry->direction = DistanceEntry::Directions::ALL;
    return false;
  }

  // Build an SENode for distance.
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const ir::Loop* subscript_loop = GetLoopForSubscriptPair(&subscript_pair);
  SENode* source_constant_term =
      GetConstantTerm(subscript_loop, source->AsSERecurrentNode());
  SENode* destination_constant_term =
      GetConstantTerm(subscript_loop, destination->AsSERecurrentNode());
  SENode* constant_term_delta =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateSubtraction(
          destination_constant_term, source_constant_term));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  // We must check the offset delta and coefficient are constants.
  int64_t distance = 0;
  SEConstantNode* delta_constant = constant_term_delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    PrintDebug(
        "StrongSIVTest found delta value and coefficient value as constants "
        "with values:\n"
        "\tdelta value: " +
        ToString(delta_value) +
        "\n\tcoefficient value: " + ToString(coefficient_value) + "\n");
    // Check if the distance is not integral to try to prove independence.
    if (delta_value % coefficient_value != 0) {
      PrintDebug(
          "StrongSIVTest proved independence through distance not being an "
          "integer.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / coefficient_value;
      PrintDebug("StrongSIV test found distance as " + ToString(distance));
    }
  } else {
    // If we can't fold delta and coefficient to single values we can't produce
    // distance.
    // As a result we can't perform the rest of the pass and must assume
    // dependence in all directions.
    PrintDebug("StrongSIVTest could not produce a distance. Must exit.");
    distance_entry->distance = DistanceEntry::Directions::ALL;
    return false;
  }

  // Next we gather the upper and lower bounds as constants if possible. If
  // distance > upper_bound - lower_bound we prove independence.
  SENode* lower_bound = GetLowerBound(subscript_loop);
  SENode* upper_bound = GetUpperBound(subscript_loop);
  if (lower_bound && upper_bound) {
    PrintDebug("StrongSIVTest found bounds.");
    SENode* bounds = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.CreateSubtraction(upper_bound, lower_bound));

    if (bounds->GetType() == SENode::SENodeType::Constant) {
      int64_t bounds_value = bounds->AsSEConstantNode()->FoldToSingleValue();
      PrintDebug(
          "StrongSIVTest found upper_bound - lower_bound as a constant with "
          "value " +
          ToString(bounds_value));

      // If the absolute value of the distance is > upper bound - lower bound
      // then we prove independence.
      if (llabs(distance) > llabs(bounds_value)) {
        PrintDebug(
            "StrongSIVTest proved independence through distance escaping the "
            "loop bounds.");
        distance_entry->dependence_information =
            DistanceEntry::DependenceInformation::DISTANCE;
        distance_entry->direction = DistanceEntry::Directions::NONE;
        distance_entry->distance = distance;
        return true;
      }
    }
  } else {
    PrintDebug("StrongSIVTest was unable to gather lower and upper bounds.");
  }

  // Otherwise we can get a direction as follows
  //             { < if distance > 0
  // direction = { = if distance == 0
  //             { > if distance < 0
  PrintDebug(
      "StrongSIVTest could not prove independence. Gathering direction "
      "information.");
  if (distance > 0) {
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DISTANCE;
    distance_entry->direction = DistanceEntry::Directions::LT;
    distance_entry->distance = distance;
    return false;
  }
  if (distance == 0) {
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DISTANCE;
    distance_entry->direction = DistanceEntry::Directions::EQ;
    distance_entry->distance = 0;
    return false;
  }
  if (distance < 0) {
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DISTANCE;
    distance_entry->direction = DistanceEntry::Directions::GT;
    distance_entry->distance = distance;
    return false;
  }

  // We were unable to prove independence or discern any additional information
  // Must assume <=> direction.
  PrintDebug(
      "StrongSIVTest was unable to determine any dependence information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::SymbolicStrongSIVTest(
    SENode* source, SENode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing SymbolicStrongSIVTest.");
  SENode* source_destination_delta = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(source, destination));
  // By cancelling out the induction variables by subtracting the source and
  // destination we can produce an expression of symbolics and constants. This
  // expression can be compared to the loop bounds to find if the offset is
  // outwith the bounds.
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const ir::Loop* subscript_loop = GetLoopForSubscriptPair(&subscript_pair);
  if (IsProvablyOutsideOfLoopBounds(subscript_loop, source_destination_delta,
                                    coefficient)) {
    PrintDebug(
        "SymbolicStrongSIVTest proved independence through loop bounds.");
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DIRECTION;
    distance_entry->direction = DistanceEntry::Directions::NONE;
    return true;
  }
  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "SymbolicStrongSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::WeakZeroSourceSIVTest(
    SENode* source, SERecurrentNode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing WeakZeroSourceSIVTest.");
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const ir::Loop* subscript_loop = GetLoopForSubscriptPair(&subscript_pair);
  // Build an SENode for distance.
  SENode* destination_constant_term =
      GetConstantTerm(subscript_loop, destination);
  SENode* delta = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(source, destination_constant_term));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  int64_t distance = 0;
  SEConstantNode* delta_constant = delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    PrintDebug(
        "WeakZeroSourceSIVTest folding delta and coefficient to constants.");
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    // Check if the distance is not integral.
    if (delta_value % coefficient_value != 0) {
      PrintDebug(
          "WeakZeroSourceSIVTest proved independence through distance not "
          "being an integer.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / coefficient_value;
      PrintDebug(
          "WeakZeroSourceSIVTest calculated distance with the following "
          "values\n"
          "\tdelta value: " +
          ToString(delta_value) +
          "\n\tcoefficient value: " + ToString(coefficient_value) +
          "\n\tdistance: " + ToString(distance) + "\n");
    }
  } else {
    PrintDebug(
        "WeakZeroSourceSIVTest was unable to fold delta and coefficient to "
        "constants.");
  }

  // If we can prove the distance is outside the bounds we prove independence.
  SEConstantNode* lower_bound =
      GetLowerBound(subscript_loop)->AsSEConstantNode();
  SEConstantNode* upper_bound =
      GetUpperBound(subscript_loop)->AsSEConstantNode();
  if (lower_bound && upper_bound) {
    PrintDebug("WeakZeroSourceSIVTest found bounds as SEConstantNodes.");
    int64_t lower_bound_value = lower_bound->FoldToSingleValue();
    int64_t upper_bound_value = upper_bound->FoldToSingleValue();
    if (!IsWithinBounds(llabs(distance), lower_bound_value,
                        upper_bound_value)) {
      PrintDebug(
          "WeakZeroSourceSIVTest proved independence through distance escaping "
          "the loop bounds.");
      PrintDebug(
          "Bound values were as follow\n"
          "\tlower bound value: " +
          ToString(lower_bound_value) +
          "\n\tupper bound value: " + ToString(upper_bound_value) +
          "\n\tdistance value: " + ToString(distance) + "\n");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DISTANCE;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      distance_entry->distance = distance;
      return true;
    }
  } else {
    PrintDebug(
        "WeakZeroSourceSIVTest was unable to find lower and upper bound as "
        "SEConstantNodes.");
  }

  // Now we want to see if we can detect to peel the first or last iterations.

  // We get the FirstTripValue as GetFirstTripInductionNode() +
  // GetConstantTerm(destination)
  SENode* first_trip_SENode =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
          GetFirstTripInductionNode(subscript_loop),
          GetConstantTerm(subscript_loop, destination)));

  // If source == FirstTripValue, peel_first.
  if (first_trip_SENode) {
    PrintDebug("WeakZeroSourceSIVTest built first_trip_SENode.");
    if (first_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroSourceSIVTest has found first_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(first_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (source == first_trip_SENode) {
      // We have found that peeling the first iteration will break dependency.
      PrintDebug(
          "WeakZeroSourceSIVTest has found peeling first iteration will break "
          "dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_first = true;
      return false;
    }
  } else {
    PrintDebug("WeakZeroSourceSIVTest was unable to build first_trip_SENode");
  }

  // We get the LastTripValue as GetFinalTripInductionNode(coefficient) +
  // GetConstantTerm(destination)
  SENode* final_trip_SENode =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
          GetFinalTripInductionNode(subscript_loop, coefficient),
          GetConstantTerm(subscript_loop, destination)));

  // If source == LastTripValue, peel_last.
  if (final_trip_SENode) {
    PrintDebug("WeakZeroSourceSIVTest built final_trip_SENode.");
    if (first_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroSourceSIVTest has found final_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(final_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (source == final_trip_SENode) {
      // We have found that peeling the last iteration will break dependency.
      PrintDebug(
          "WeakZeroSourceSIVTest has found peeling final iteration will break "
          "dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_last = true;
      return false;
    }
  } else {
    PrintDebug("WeakZeroSourceSIVTest was unable to build final_trip_SENode");
  }

  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "WeakZeroSourceSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::WeakZeroDestinationSIVTest(
    SERecurrentNode* source, SENode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing WeakZeroDestinationSIVTest.");
  // Build an SENode for distance.
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const ir::Loop* subscript_loop = GetLoopForSubscriptPair(&subscript_pair);
  SENode* source_constant_term = GetConstantTerm(subscript_loop, source);
  SENode* delta = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(destination, source_constant_term));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  int64_t distance = 0;
  SEConstantNode* delta_constant = delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    PrintDebug(
        "WeakZeroDestinationSIVTest folding delta and coefficient to "
        "constants.");
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    // Check if the distance is not integral.
    if (delta_value % coefficient_value != 0) {
      PrintDebug(
          "WeakZeroDestinationSIVTest proved independence through distance not "
          "being an integer.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / coefficient_value;
      PrintDebug(
          "WeakZeroDestinationSIVTest calculated distance with the following "
          "values\n"
          "\tdelta value: " +
          ToString(delta_value) +
          "\n\tcoefficient value: " + ToString(coefficient_value) +
          "\n\tdistance: " + ToString(distance) + "\n");
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to fold delta and coefficient "
        "to constants.");
  }

  // If we can prove the distance is outside the bounds we prove independence.
  SEConstantNode* lower_bound =
      GetLowerBound(subscript_loop)->AsSEConstantNode();
  SEConstantNode* upper_bound =
      GetUpperBound(subscript_loop)->AsSEConstantNode();
  if (lower_bound && upper_bound) {
    PrintDebug("WeakZeroDestinationSIVTest found bounds as SEConstantNodes.");
    int64_t lower_bound_value = lower_bound->FoldToSingleValue();
    int64_t upper_bound_value = upper_bound->FoldToSingleValue();
    if (!IsWithinBounds(llabs(distance), lower_bound_value,
                        upper_bound_value)) {
      PrintDebug(
          "WeakZeroDestinationSIVTest proved independence through distance "
          "escaping the loop bounds.");
      PrintDebug(
          "Bound values were as follows\n"
          "\tlower bound value: " +
          ToString(lower_bound_value) +
          "\n\tupper bound value: " + ToString(upper_bound_value) +
          "\n\tdistance value: " + ToString(distance));
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DISTANCE;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      distance_entry->distance = distance;
      return true;
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to find lower and upper bound "
        "as SEConstantNodes.");
  }

  // Now we want to see if we can detect to peel the first or last iterations.

  // We get the FirstTripValue as GetFirstTripInductionNode() +
  // GetConstantTerm(source)
  SENode* first_trip_SENode = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateAddNode(GetFirstTripInductionNode(subscript_loop),
                                      GetConstantTerm(subscript_loop, source)));

  // If destination == FirstTripValue, peel_first.
  if (first_trip_SENode) {
    PrintDebug("WeakZeroDestinationSIVTest built first_trip_SENode.");
    if (first_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroDestinationSIVTest has found first_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(first_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (destination == first_trip_SENode) {
      // We have found that peeling the first iteration will break dependency.
      PrintDebug(
          "WeakZeroDestinationSIVTest has found peeling first iteration will "
          "break dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_first = true;
      return false;
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to build first_trip_SENode");
  }

  // We get the LastTripValue as GetFinalTripInductionNode(coefficient) +
  // GetConstantTerm(source)
  SENode* final_trip_SENode =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
          GetFinalTripInductionNode(subscript_loop, coefficient),
          GetConstantTerm(subscript_loop, source)));

  // If destination == LastTripValue, peel_last.
  if (final_trip_SENode) {
    PrintDebug("WeakZeroDestinationSIVTest built final_trip_SENode.");
    if (final_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroDestinationSIVTest has found final_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(final_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (destination == final_trip_SENode) {
      // We have found that peeling the last iteration will break dependency.
      PrintDebug(
          "WeakZeroDestinationSIVTest has found peeling final iteration will "
          "break dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_last = true;
      return false;
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to build final_trip_SENode");
  }

  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "WeakZeroDestinationSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::WeakCrossingSIVTest(
    SENode* source, SENode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing WeakCrossingSIVTest.");
  // We currently can't handle symbolic WeakCrossingSIVTests. If either source
  // or destination are not SERecurrentNodes we must exit.
  if (!source->AsSERecurrentNode() || !destination->AsSERecurrentNode()) {
    PrintDebug(
        "WeakCrossingSIVTest found source or destination != SERecurrentNode. "
        "Exiting");
    distance_entry->direction = DistanceEntry::Directions::ALL;
    return false;
  }

  // Build an SENode for distance.
  SENode* offset_delta =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateSubtraction(
          destination->AsSERecurrentNode()->GetOffset(),
          source->AsSERecurrentNode()->GetOffset()));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  int64_t distance = 0;
  SEConstantNode* delta_constant = offset_delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    PrintDebug(
        "WeakCrossingSIVTest folding offset_delta and coefficient to "
        "constants.");
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    // Check if the distance is not integral or if it has a non-integral part
    // equal to 1/2.
    if (delta_value % (2 * coefficient_value) != 0 &&
        static_cast<float>(delta_value % (2 * coefficient_value)) /
                static_cast<float>(2 * coefficient_value) !=
            0.5) {
      PrintDebug(
          "WeakCrossingSIVTest proved independence through distance escaping "
          "the loop bounds.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / (2 * coefficient_value);
    }

    if (distance == 0) {
      PrintDebug("WeakCrossingSIVTest found EQ dependence.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DISTANCE;
      distance_entry->direction = DistanceEntry::Directions::EQ;
      distance_entry->distance = 0;
      return false;
    }
  } else {
    PrintDebug(
        "WeakCrossingSIVTest was unable to fold offset_delta and coefficient "
        "to constants.");
  }

  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "WeakCrossingSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

using PartitionedSubscripts =
    std::vector<std::set<std::pair<ir::Instruction*, ir::Instruction*>>>;
PartitionedSubscripts LoopDependenceAnalysis::PartitionSubscripts(
    const std::vector<ir::Instruction*>& source_subscripts,
    const std::vector<ir::Instruction*>& destination_subscripts) {
  PartitionedSubscripts partitions{};

  auto num_subscripts = source_subscripts.size();

  // Create initial partitions
  for (size_t i = 0; i < num_subscripts; ++i) {
    partitions.push_back({{source_subscripts[i], destination_subscripts[i]}});
  }

  for (auto loop : loops_) {
    int64_t k = -1;

    for (size_t j = 0; j < partitions.size(); ++j) {
      auto& current_partition = partitions[j];

      auto it = std::find_if(
          current_partition.begin(), current_partition.end(),
          [loop,
           this](const std::pair<ir::Instruction*, ir::Instruction*>& elem)
              -> bool {
            auto rec_0 =
                scalar_evolution_.AnalyzeInstruction(std::get<0>(elem))
                    ->CollectRecurrentNodes();
            auto rec_1 =
                scalar_evolution_.AnalyzeInstruction(std::get<1>(elem))
                    ->CollectRecurrentNodes();

            rec_0.insert(rec_0.end(), rec_1.begin(), rec_1.end());

            auto loops_in_pair = CollectLoops(rec_0);
            auto end_it = loops_in_pair.end();

            return std::find(loops_in_pair.begin(), end_it, loop) != end_it;
          });

      auto has_loop = it != current_partition.end();

      if (has_loop) {
        if (k == -1) {
          k = j;
        } else {
          // Add partitions[j] to partitions[k] and discard partitions[j]
          partitions[static_cast<size_t>(k)].insert(current_partition.begin(),
                                                    current_partition.end());
          current_partition.clear();
        }
      }
    }
  }

  // Remove empty (discarded) partitions
  partitions.erase(
      std::remove_if(
          partitions.begin(), partitions.end(),
          [](const std::set<std::pair<ir::Instruction*, ir::Instruction*>>&
                 partition) { return partition.empty(); }),
      partitions.end());

  return partitions;
}

}  // namespace opt
}  // namespace spvtools
