#include <vector>

#define __um  __attribute__((address_space(3)))
#define __register  __attribute__((address_space(4)))

template<typename T>
struct SL_TENSOR_UM {
    T &operator()(int...);
    T &operator+(SL_TENSOR_UM<T> rhs);
    T &operator+=(SL_TENSOR_UM<T> rhs);
    T &operator=(SL_TENSOR_UM<T> rhs);
} __um;

template<typename T>
struct SL_TENSOR {
    T &operator()(int...);
    T &operator+(SL_TENSOR<T> rhs);
    T &operator+=(SL_TENSOR<T> rhs);
    T &operator=(SL_TENSOR<T> rhs);
} __register;

template<typename T> SL_TENSOR_UM<T> to_tensor(T *ptr, std::vector<int> shape);
