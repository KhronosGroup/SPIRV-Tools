#include "source/lint/variable_state.h"

#include "gtest/gtest.h"

namespace spvtools {
namespace lint {
namespace uninitialized_variables {

State leafY() { return State::NewLeaf(Initialized::Yes); }
State leafU() { return State::NewLeaf(Initialized::Unknown); }
State leafN() { return State::NewLeaf(Initialized::No); }

// Unwrapping equality, throws on uncomparable operands
bool operator==(const State& lhs, const State& rhs) {
  const std::optional<bool> r = (lhs).TryEquals(rhs);
  EXPECT_TRUE(r.has_value());
  return r.value();
}

bool operator!=(const State& lhs, const State& rhs) { return !(lhs == rhs); }

TEST(VarState, BasicOperations) {
  EXPECT_EQ(leafY(), leafY());
  EXPECT_EQ(leafN(), leafN());
  EXPECT_NE(leafY(), leafN());
  EXPECT_NE(leafY(), leafU());

  State comp =
      State::NewComposite({State::NewComposite({leafY(), leafN()}), leafN()});
  EXPECT_EQ(comp.MaxState(), Initialized::Yes);
  EXPECT_EQ(comp.MinState(), Initialized::No);
  EXPECT_EQ(
      leafN().TryUnion(comp),
      State::NewComposite({State::NewComposite({leafY(), leafN()}), leafN()}));

  EXPECT_EQ(leafN().TryUnion(leafY()), leafY());
  EXPECT_EQ(leafN().TryIntersect(leafY()), leafN());
  EXPECT_EQ(leafU().TryIntersect(leafY()), leafU());

  EXPECT_EQ(leafU().TryDifference(leafY()), leafN());
  EXPECT_EQ(leafY().TryDifference(leafY()), leafN());
  EXPECT_EQ(leafY().TryDifference(leafU()), leafY());
  EXPECT_EQ(leafU().TryDifference(leafN()), leafU());
  EXPECT_EQ(leafN().TryDifference(leafN()), leafN());

  const State comp_n =
      State::NewComposite({leafN(), leafN(), leafN(), leafN()});
  const State comp_y =
      State::NewComposite({leafY(), leafY(), leafY(), leafY()});
  const State comp_u =
      State::NewComposite({leafU(), leafU(), leafU(), leafU()});

  EXPECT_TRUE(comp_y.TryEquals(comp_y).value());
  EXPECT_TRUE(comp_n.TryEquals(comp_n).value());
  EXPECT_FALSE(comp_y.TryEquals(comp_u).value());

  EXPECT_EQ(comp_n.TryUnion(comp_n), comp_n);
  EXPECT_EQ(comp_y.TryUnion(comp_n), comp_y);
  EXPECT_EQ(comp_y.TryIntersect(comp_u), comp_u);

  EXPECT_EQ(comp_y.TryDifference(comp_u), comp_y);
  EXPECT_EQ(comp_y.TryDifference(comp_n), comp_y);
  EXPECT_EQ(comp_n.TryDifference(comp_y), comp_n);

  EXPECT_TRUE(comp_y.TryAllGreaterOrEqual(comp_n).value());
  EXPECT_TRUE(comp_y.TryAllGreaterOrEqual(comp_u).value());
  EXPECT_TRUE(comp_u.TryAllGreaterOrEqual(comp_n).value());
  EXPECT_FALSE(comp_n.TryAllGreaterOrEqual(comp_u).value());
  EXPECT_FALSE(comp_n.TryAllGreaterOrEqual(comp_y).value());

  EXPECT_FALSE(comp_n.TryAllGreaterOrEqual(leafY()).value());
  EXPECT_TRUE(comp_y.TryAllGreaterOrEqual(leafN()).value());
  EXPECT_TRUE(comp_y.TryAllGreaterOrEqual(leafY()).value());
  EXPECT_TRUE(leafY().TryAllGreaterOrEqual(comp_u).value());

  const State comp_a =
      State::NewComposite({leafN(), leafY(), leafY(), leafN()});
  const State comp_b =
      State::NewComposite({leafY(), leafN(), leafU(), leafN()});

  EXPECT_EQ(comp_a.TryUnion(leafN()), comp_a);
  EXPECT_EQ(leafN().TryUnion(comp_a), comp_a);

  const State comp_max_ab =
      State::NewComposite({leafY(), leafY(), leafY(), leafN()});
  const State comp_min_ab =
      State::NewComposite({leafN(), leafN(), leafU(), leafN()});
  const State comp_min_aU =
      State::NewComposite({leafN(), leafU(), leafU(), leafN()});
  const State comp_max_aU =
      State::NewComposite({leafU(), leafY(), leafY(), leafU()});
  const State comp_min_bU =
      State::NewComposite({leafU(), leafN(), leafU(), leafN()});
  const State comp_max_bU =
      State::NewComposite({leafY(), leafU(), leafU(), leafU()});
  const State comp_a_minus_b =
      State::NewComposite({leafN(), leafY(), leafY(), leafN()});
  const State comp_b_minus_a =
      State::NewComposite({leafY(), leafN(), leafN(), leafN()});

  EXPECT_EQ(comp_a.TryUnion(comp_b), comp_max_ab);
  EXPECT_EQ(comp_b.TryUnion(comp_a), comp_max_ab);
  EXPECT_EQ(comp_b.TryIntersect(comp_a), comp_min_ab);

  EXPECT_EQ(comp_a.TryIntersect(leafU()), comp_min_aU);
  EXPECT_EQ(leafU().TryIntersect(comp_a), comp_min_aU);
  EXPECT_EQ(leafU().TryUnion(comp_a), comp_max_aU);
  EXPECT_EQ(comp_a.TryUnion(leafU()), comp_max_aU);
  EXPECT_EQ(comp_b.TryUnion(leafU()), comp_max_bU);
  EXPECT_EQ(leafU().TryUnion(comp_b), comp_max_bU);

  EXPECT_EQ(leafY().TryUnion(comp_a), leafY());
  EXPECT_EQ(leafY().TryIntersect(comp_a), comp_a);
  EXPECT_EQ(leafN().TryDifference(comp_a), leafN());
  EXPECT_EQ(comp_a.TryDifference(comp_a), leafN());
  EXPECT_EQ(comp_a.TryDifference(leafY()), comp_n);
  EXPECT_EQ(comp_b.TryDifference(leafN()), comp_b);
  EXPECT_EQ(comp_a.TryDifference(comp_b), comp_a_minus_b);
  EXPECT_EQ(comp_b.TryDifference(comp_a), comp_b_minus_a);

  EXPECT_EQ(comp_b.TryDifference(leafN()).value().TryUnion(comp_a),
            comp_max_ab);
  EXPECT_EQ(comp_a.TryDifference(comp_y.TryDifference(leafN()).value())
                .value()
                .TryUnion(comp_b),
            comp_b);

  EXPECT_FALSE(comp_a.TryAllGreaterOrEqual(comp_b).value());
  EXPECT_FALSE(comp_a.TryAllGreaterOrEqual(leafY()).value());
  EXPECT_FALSE(comp_b.TryAllGreaterOrEqual(comp_a).value());

  const State tree_c = State::NewComposite({leafN(), comp_a, comp_b, leafY()});
  const State tree_d = State::NewComposite({leafY(), comp_b, comp_a, leafY()});

  EXPECT_FALSE(tree_c.TryAllGreaterOrEqual(leafY()).value());
  EXPECT_FALSE(leafN().TryAllGreaterOrEqual(tree_d).value());
  EXPECT_FALSE(tree_c.TryAllGreaterOrEqual(tree_d).value());
  EXPECT_FALSE(tree_d.TryAllGreaterOrEqual(tree_c).value());
  EXPECT_EQ(tree_c, tree_c);

  const State tree_max_cd =
      State::NewComposite({leafY(), comp_max_ab, comp_max_ab, leafY()});
  const State tree_min_cf =
      State::NewComposite({leafN(), comp_min_ab, comp_min_ab, leafY()});
  const State tree_min_cU =
      State::NewComposite({leafN(), comp_min_aU, comp_min_bU, leafU()});
  const State tree_max_bd =
      State::NewComposite({leafY(), comp_b, comp_max_aU, leafY()});

  const State tree_max_ac =
      State::NewComposite({leafN(), leafY(), leafY(), leafY()});
  const State tree_min_ac =
      State::NewComposite({leafN(), comp_a, comp_b, leafN()});

  EXPECT_EQ(tree_c.TryUnion(tree_d), tree_max_cd);
  EXPECT_EQ(tree_d.TryUnion(tree_c), tree_max_cd);
  EXPECT_EQ(tree_d.TryUnion(comp_b), tree_max_bd);
  EXPECT_EQ(comp_b.TryUnion(tree_d), tree_max_bd);
  EXPECT_EQ(tree_c.TryIntersect(tree_d), tree_min_cf);
  EXPECT_EQ(tree_d.TryIntersect(tree_c), tree_min_cf);
  EXPECT_EQ(tree_c.TryIntersect(leafU()), tree_min_cU);
  EXPECT_EQ(leafU().TryIntersect(tree_c), tree_min_cU);

  EXPECT_EQ(tree_c.TryUnion(comp_a), tree_max_ac);
  EXPECT_EQ(comp_a.TryUnion(tree_c), tree_max_ac);
  EXPECT_EQ(tree_c.TryIntersect(comp_a), tree_min_ac);
  EXPECT_EQ(comp_a.TryIntersect(tree_c), tree_min_ac);

  const State tree_c_minus_d =
      State::NewComposite({leafN(), comp_a_minus_b, comp_b_minus_a, leafN()});
  const State tree_d_minus_c =
      State::NewComposite({leafY(), comp_b_minus_a, comp_a_minus_b, leafN()});

  EXPECT_EQ(tree_c.TryDifference(tree_d), tree_c_minus_d);
  EXPECT_EQ(tree_d.TryDifference(tree_c), tree_d_minus_c);

  const State tree_e = State::NewComposite({tree_c, comp_b, tree_d, leafY()});
  const State tree_f = State::NewComposite({comp_a, leafU(), comp_b, tree_e});
  const State tree_max_ef =
      State::NewComposite({tree_max_ac, comp_max_bU, tree_max_bd, leafY()});

  EXPECT_EQ(tree_e.TryUnion(tree_f), tree_max_ef);
  EXPECT_EQ(tree_f.TryUnion(tree_e), tree_max_ef);

  EXPECT_FALSE(tree_e.TryAllGreaterOrEqual(tree_f).value());
  EXPECT_FALSE(tree_f.TryAllGreaterOrEqual(tree_e).value());

  EXPECT_FALSE(comp_a.TryAllGreaterOrEqual(tree_c).value());
  EXPECT_FALSE(tree_c.TryAllGreaterOrEqual(comp_a).value());
  EXPECT_TRUE(
      comp_a.TryUnion(State::NewComposite({leafN(), leafU(), comp_b, leafY()}))
          .value()
          .TryAllGreaterOrEqual(tree_c)
          .value());
}

TEST(VarState, VarStatesMap) {
  VarStateMap empty;
  VarStateMap empty2;
  const auto make_a = []() {
    VarStateMap e;
    return e.Union(1, leafN())
        .Union(2, leafY())
        .Union(3, State::NewComposite({leafN(), leafN(), leafN()}));
  };
  // Identical, but not sharing a CoW backing map
  VarStateMap a1 = make_a();
  VarStateMap a2 = make_a();

  EXPECT_TRUE(a1.Get(1).has_value());
  EXPECT_TRUE(a1.Get(2).has_value());
  EXPECT_TRUE(a1.Get(3).has_value());
  EXPECT_EQ(a1.Get(1), leafN());
  EXPECT_EQ(a1.Get(2), leafY());
  EXPECT_EQ(a1.Get(3), State::NewComposite({leafN(), leafN(), leafN()}));

  EXPECT_TRUE(empty.Equals(empty2));
  EXPECT_TRUE(a1.Equals(a2));
  EXPECT_TRUE(a2.Equals(a1));

  VarStateMap dont_change_operands = empty.Union(a1);
  EXPECT_TRUE(empty.Equals(empty2));
  EXPECT_TRUE(a1.Equals(a2));
  EXPECT_TRUE(dont_change_operands.Equals(a2));

  EXPECT_TRUE(empty.Difference(a1).Equals(empty));
  EXPECT_TRUE(empty.Difference(a1).Equals(empty2));
  EXPECT_TRUE(a1.Difference(empty).Equals(a2));
  EXPECT_TRUE(a2.Difference(a1).Equals(empty));

  VarStateMap b_expected =
      empty.Union(1, leafU())
          .Union(2, leafY())
          .Union(3, State::NewComposite({leafN(), leafY(), leafN()}));
  // Check associativity
  VarStateMap b1 =
      a1.Union(1, leafU())
          .Union(3, State::NewComposite({leafN(), leafY(), leafN()}));
  VarStateMap b2 = a1.Union(empty.Union(
      a1.Union(1, leafU())
          .Union(3, State::NewComposite({leafN(), leafY(), leafN()}))));
  EXPECT_TRUE(b1.Equals(b_expected));
  EXPECT_TRUE(b2.Equals(b_expected));
  EXPECT_TRUE(b1.Equals(b2));

  EXPECT_TRUE(a1.Equals(a2));
  EXPECT_FALSE(a1.Equals(b_expected));
  EXPECT_TRUE(b1.GreaterOrEqual(a2));
  EXPECT_FALSE(a2.GreaterOrEqual(b1));

  EXPECT_TRUE(b1.Intersect(b2).Equals(b_expected));
  EXPECT_TRUE(b1.Intersect(empty).Equals(empty2));
  VarStateMap c =
      VarStateMap()
          .Union(3, State::NewComposite({leafU(), leafY(), leafN()}))
          .Union(111, leafY());
  VarStateMap inter_bc =
      VarStateMap().Union(3, State::NewComposite({leafN(), leafY(), leafN()}));
  EXPECT_TRUE(b1.Intersect(c).Equals(inter_bc));
  EXPECT_TRUE(c.Intersect(b1).Equals(inter_bc));

  VarStateMap b_minus_c =
      VarStateMap().Union(1, leafU()).Union(2, leafY()).Union(3, leafN());
  EXPECT_TRUE(b1.Difference(c).Equals(b_minus_c)) << b1.Difference(c) << "\n"
                                                  << b_minus_c;
  EXPECT_TRUE(b1.Difference(c).Union(c).GreaterOrEqual(b1));
  EXPECT_TRUE(b1.Difference(c).Union(c).Intersect(b2).Equals(b1));
}

}  // namespace uninitialized_variables
}  // namespace lint
}  // namespace spvtools
