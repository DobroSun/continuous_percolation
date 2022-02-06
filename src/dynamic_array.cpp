#pragma once

#define GROW_FUNCTION(x) (2*(x) + 8)


template<class T>
struct dynamic_array {
  T*   data     = NULL;
  uint size     = 0;
  uint capacity = 0;

  T& operator[](uint64 index) {
    assert(index < size);
    return data[index];
  }

  const T& operator[](uint64 index) const {
    assert(index < size);
    return data[index];
  }
};

template<class T>
void array_reserve(dynamic_array<T>* a, size_t new_capacity) {
  assert(new_capacity > a->capacity && "dynamic_array<T>;:reserve new_capacity is <= then array one, won't do anything!");

  a->data     = (T*) realloc(a->data, sizeof(T) * new_capacity);
  a->capacity = new_capacity;
}

template<class T>
void array_resize(dynamic_array<T>* a, size_t new_size) {
  array_reserve(a, new_size);
  a->size = new_size;
  assert(a->size == a->capacity);
}


template<class T>
T* array_add(dynamic_array<T>* a) {
  if(a->size == a->capacity) {
    array_reserve(a, GROW_FUNCTION(a->capacity));
  }
  return &a->data[a->size++];
}

template<class T>
T* array_add(dynamic_array<T>* a, T v) {
  return &(*array_add(a) = v);
}

template<class T, class F>
T* array_find_by_predicate(dynamic_array<T>* a, F predicate) {
  for(T& p : *a) {
    if(predicate(p)) {
      return &p;
    }
  }
  return NULL;
}

template<class T>
bool array_contains(dynamic_array<T>* a, T v) {
  return array_find(a, v) != NULL;
}

template<class T>
void array_clear(dynamic_array<T>* a) {
  a->size = 0;
}

#if 0
template<class T>
void array_copy(dynamic_array<T>* a, const dynamic_array<T>* b) {
  if(a->capacity < b->capacity) {
    dealloc(a->allocator, a->data);
    a->data     = (T*) alloc(a->allocator, sizeof(T)*b->capacity);
    a->capacity = b->capacity;
  }
  memcpy(a->data, b->data, sizeof(T)*b->size);
  a->size = b->size;
}

template<class T>
dynamic_array<T> array_copy(const dynamic_array<T>* b) {
  dynamic_array<T> a;
  a.data     = (T*) alloc(a.allocator, sizeof(T)*b->capacity);
  a.size     = b->size;
  a.capacity = b->capacity;
  memcpy(a.data, b->data, sizeof(T)*b->size);
  return a;
}
#endif

template<class T>
void array_free(dynamic_array<T>* a) {
  if (a->data) free(a->data);
  a->data = NULL;
  a->size = 0;
  a->capacity = 0;
}

#undef GROW_FUNCTION

