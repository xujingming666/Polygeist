// #include <vector>

#define __um  __attribute__((address_space(3)))
#define __register  __attribute__((address_space(4)))

// Annotation to indicate an argument as output
#define __output __attribute__((annotate("output")))

template<typename T>
struct SL_TENSOR_UM {
    SL_TENSOR_UM<T> &operator()(int...);
    SL_TENSOR_UM<T> &operator+(SL_TENSOR_UM<T> rhs);
    SL_TENSOR_UM<T> &operator+=(SL_TENSOR_UM<T> rhs);
    SL_TENSOR_UM<T> &operator=(SL_TENSOR_UM<T> rhs);
} __um;

template<typename T>
struct SL_TENSOR {
    SL_TENSOR<T> &operator()(int...);
    SL_TENSOR<T> &operator+(SL_TENSOR<T> rhs);
    SL_TENSOR<T> &operator+=(SL_TENSOR<T> rhs);
    SL_TENSOR<T> &operator=(SL_TENSOR<T> rhs);
} __register;

template<typename T>
struct SL_VECTOR {
    SL_VECTOR(T...);
    SL_VECTOR<T> &operator()(int...);
    SL_VECTOR<T> &operator+(SL_VECTOR<T> rhs);
    SL_VECTOR<T> &operator+=(SL_VECTOR<T> rhs);
    SL_VECTOR<T> &operator=(SL_VECTOR<T> rhs);
} __register;

template<typename T> SL_TENSOR_UM<T> to_tensor(T *ptr, SL_VECTOR<int> shape);

template<typename T> SL_TENSOR_UM<T> mac_add(SL_TENSOR_UM<T> lhs, SL_TENSOR_UM<T> rhs);


