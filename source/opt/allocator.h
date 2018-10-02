// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_OPT_ALLOCATOR_H_
#define SOURCE_OPT_ALLOCATOR_H_

#include <cassert>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace spvtools {

// A batch allocator.
//
// This allocator wraps around malloc/free and conducts memory allocation on
// a |kPageSize|-byte footprint. This groups multiple small allocation into a
// larger one so that we can improve performance.
//
// This allocator adopts an ever-increasing approach; new memory are kept
// allocated even if previously requested memory are deallocated. It's only
// at the time of the destruction of this allocator that all the memory gets
// released.
template <std::size_t kPageSize = 4096>
class AllocatorImpl {
 public:
  AllocatorImpl() : cur_page_next_(nullptr), cur_page_end_(nullptr) {
    GrabNewStandardPage();  // Start with a new page
  }

  // Forbid copy constructor/assignment
  AllocatorImpl(const AllocatorImpl&) = delete;
  AllocatorImpl& operator=(const AllocatorImpl&) = delete;

  // Forbid move constructor/assignment
  AllocatorImpl(AllocatorImpl&&) = delete;
  AllocatorImpl& operator=(AllocatorImpl&&) = delete;

  ~AllocatorImpl() { Clear(); }

  void* Allocate(std::size_t size, std::size_t align) {
    assert(size > 0);
    assert(IsPowerOfTwo(align) && "alignment must be a power of two");

    // Try to see if we can allocate from the current normal page
    char* aligned_next = (char*)AlignUp(cur_page_next_, align);
    if (aligned_next + size <= cur_page_end_) {
      cur_page_next_ = aligned_next + size;
      return aligned_next;
    }

    // If we cannot allocate from the current page, it means the requested
    // size is too large: we need to either do dedicated allocation or
    // start a new page.

    // Do dedicated allocation if the size is bigger than what a page provides
    std::size_t padded_size = size + align - 1;
    if (padded_size > kPageSize) {
      return AlignUp(GrabNewOutlierPage(padded_size), align);
    }

    // Grab a new page for the allocation
    GrabNewStandardPage();
    aligned_next = (char*)AlignUp(cur_page_next_, align);
    cur_page_next_ = aligned_next + size;
    return aligned_next;
  }

  void Deallocate(const void* /*ptr*/, std::size_t /*size*/) {}

 private:
  // Returns true if the given number is a power of two.
  static bool IsPowerOfTwo(std::size_t number) {
    return number != 0 && (number & (number - 1)) == 0;
  }

  // Aligns the given pointer |ptr| according to the given alignment |align|
  // with rounding up.
  static void* AlignUp(void* ptr, std::size_t align) {
    assert(IsPowerOfTwo(align) && "alignment must be a power of two");
    assert((uintptr_t)ptr + align - 1 >= (uintptr_t)ptr);

    return (void*)(((uintptr_t)ptr + align - 1) & ~(uintptr_t)(align - 1));
  }

  /// Deallocates all memory allocated by this allocator.
  void Clear() {
    for (void* page : pages_) free(page);
    for (void* outlier : outliers_) free(outlier);

    cur_page_next_ = nullptr;
    cur_page_end_ = nullptr;
  }

  // Grab a new standard page from the system.
  void GrabNewStandardPage() {
    char* page = static_cast<char*>(std::malloc(kPageSize));
    pages_.push_back(page);
    cur_page_next_ = page;
    cur_page_end_ = page + kPageSize;
  }

  // Grab an outlier page of the given |size| from the system.
  void* GrabNewOutlierPage(std::size_t size) {
    void* page = std::malloc(size);
    outliers_.push_back(page);
    return page;
  }

  std::vector<void*> outliers_;  // Allocated outlier pages
  std::vector<void*> pages_;     // Allocated normal pages
  char* cur_page_next_;          // The next available space in the current page
  char* cur_page_end_;           // The end of the curent page
};

typedef AllocatorImpl<> Allocator;

// Sets the allocator to the given |allocator|. Use nullptr to mean std::new.
void SetCustomAllocator(Allocator* allocator);

void* CustomAllocate(std::size_t size);
void CustomDeallocate(void* ptr, std::size_t size);

// A RAII class for maintaining custom allocator.
class AllocatorRAII {
 public:
  AllocatorRAII() : allocator_() { SetCustomAllocator(&allocator_); }
  ~AllocatorRAII() { SetCustomAllocator(nullptr); }

 private:
  Allocator allocator_;
};

// A custom allocator implementing the allocator interface required by
// STL containers
template <typename T>
struct StlAllocator {
  // Type definitions
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef size_t difference_type;

  // Rebind allocator to type U
  template <typename U>
  struct rebind {
    typedef StlAllocator<U> other;
  };

  StlAllocator() {}

  template <class U>
  StlAllocator(const StlAllocator<U>&) {}

  // Allocates without initialization |n| elements of type T
  T* allocate(std::size_t n) {
    return static_cast<T*>(CustomAllocate(sizeof(T) * n));
  }

  // Deallocates without destruction the storage pointed by |p| of
  // the given |size|
  void deallocate(T* p, std::size_t size) { CustomDeallocate(p, size); }

  // Initializes the object allocated at |p| with the given value |val|
  void construct(pointer p, const_reference val) { new (p) T(val); }

  // Constructs an object of type |U| a the location given by |p|,
  // passing through all other arguments to the constructor.
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    ::new ((void*)p) U(std::forward<Args>(args)...);
  }

  // Deconstructs the object at |p|. It does not free the memory.
  void destroy(pointer p) { ((T*)p)->~T(); }

  // Deconstructs the object at |p|. It does not free the memory.
  template <typename U>
  void destroy(U* p) {
    p->~U();
  }

  // Returns the theoretically maximal possible number of T stored in
  // this allocator.
  size_type max_size() const {
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }
};

// STL uses the equality operator to determine if memory allocated by one
// allocator can be deallocated with another.
template <class T, class U>
bool operator==(const StlAllocator<T>&, const StlAllocator<U>&) {
  return false;  // For safety
}

template <class T, class U>
bool operator!=(const StlAllocator<T>& x, const StlAllocator<U>& y) {
  return !(x == y);
}

template <class Key, class T, class Compare = std::less<Key>>
using CAMap = std::map<Key, T, Compare, StlAllocator<std::pair<const Key, T>>>;

template <class Key, class T, class Hash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>>
using CAUnorderedMap =
    std::unordered_map<Key, T, Hash, KeyEqual,
                       StlAllocator<std::pair<const Key, T>>>;

template <class Key, class T, class Compare = std::less<Key>>
using CAMultiMap =
    std::multimap<Key, T, Compare, StlAllocator<std::pair<const Key, T>>>;

template <class Key, class Hash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>>
using CAUnorderedSet =
    std::unordered_set<Key, Hash, KeyEqual, StlAllocator<Key>>;

template <class Key, class Compare = std::less<Key>>
using CASet = std::set<Key, Compare, StlAllocator<Key>>;

}  // namespace spvtools

#endif  // SOURCE_OPT_ALLOCATOR_H_
