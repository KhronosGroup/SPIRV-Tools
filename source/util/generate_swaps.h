// Copyright (c) 2016 Google Inc.
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

#ifndef SOURCE_UTIL_GENERATE_SWAPS_H_
#define SOURCE_UTIL_GENERATE_SWAPS_H_

#include <memory>
#include <utility>
#include <vector>
#include <utility>
#include <map>

namespace spvtools {

    std::vector<std::pair<uint32_t,uint32_t>> generate_swaps(std::vector<uint32_t>& v, std::vector<uint32_t>& perm_v){
        
        size_t vec_size = v.size();
        size_t i;
        using std::pair;
        std::vector<pair<uint32_t,uint32_t>> res;
        
        // the parent vector keeps track of the swaps that were already done
        // For example if the original vector was [0,1,2,3,4,5]
        // and the permuted vector was [0,3,5,1,2,4]
        // we compare the values at each position
        // and generate the swaps (1,3),(3,1),(2,5),(2,4),(4,5)
        // however the swaps (3,1) and (4,5) are redundant
        // and they can be avoided by setting parent[3]=1 at swap (1,3)
        // and parent[4]=2,parent[5]=2, at swaps (2,4),(2,5)
        // and avoiding swapping when two elements have the same parent

        std::map<int,int>parent;
        for(int vec_elem:v){
            parent[vec_elem] = vec_elem;
        }

        for(i=0;i<vec_size;i++){

            if(v[i] != perm_v[i] && parent[v[i]] != parent[perm_v[i]]){
                parent[v[i]] = parent[perm_v[i]] = std::min(v[i],perm_v[i]);
                res.push_back(std::make_pair(v[i],perm_v[i]));
            }
        }

        return res;
    }

}  // namespace spvtools

#endif  // SOURCE_UTIL_GENERATE_SWAPS_H_


