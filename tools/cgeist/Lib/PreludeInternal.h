// #include <vector>

#define __ddr  __attribute__((address_space(1)))
#define __um  __attribute__((address_space(0)))
#define __register  __attribute__((address_space(4)))

// Annotation to indicate an argument as output
#define __output __attribute__((annotate("output")))

template<typename T>
struct SL_TENSOR_UM {
    SL_TENSOR_UM<T> &operator()(int...);
    SL_TENSOR_UM<T> &operator+(SL_TENSOR_UM<T> &rhs);
    SL_TENSOR_UM<T> &operator+=(SL_TENSOR_UM<T> &rhs);
    SL_TENSOR_UM<T> &operator=(SL_TENSOR_UM<T> &rhs);
} __um;

template<typename T>
struct SL_TENSOR {
    SL_TENSOR<T> &operator()(int...);
    SL_TENSOR<T> &operator+(SL_TENSOR<T> &rhs);
    SL_TENSOR<T> &operator+=(SL_TENSOR<T> &rhs);
    SL_TENSOR<T> &operator=(SL_TENSOR<T> &rhs);
} __register;

template<typename T>
struct SL_VECTOR {
    SL_VECTOR(T...);
    SL_VECTOR<T> &operator()(int...);
    SL_VECTOR<T> &operator+(SL_VECTOR<T> &rhs);
    SL_VECTOR<T> &operator+=(SL_VECTOR<T> &rhs);
    SL_VECTOR<T> &operator=(SL_VECTOR<T> &rhs);
} __register;

template<typename T> SL_TENSOR<T> &to_tensor(__ddr T *ptr, SL_VECTOR<int> shape);

template<typename T> SL_TENSOR_UM<T> &mac_load(SL_TENSOR<T> &src, SL_VECTOR<int> size, SL_VECTOR<int> offset);
template<typename T> void mac_store(SL_TENSOR<T> &dst, SL_TENSOR_UM<T> &src, SL_VECTOR<int>  offset);
template<typename T> SL_TENSOR_UM<T> &mac_fill(T lhs, SL_VECTOR<int>  offset);

template<typename T> SL_TENSOR_UM<T> &mac_add(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_sub(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_mul(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_div(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_div_unsigned(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_min_signed(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_max_signed(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_min_unsigned(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_max_unsigned(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_powf(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);

template<typename T> SL_TENSOR_UM<T> &mac_cos(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_exp(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_log(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_abs(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_ceil(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_floor(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_negf(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_reciprocal(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_round(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_sqrt(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_rsqrt(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_square(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_tanh(SL_TENSOR_UM<T> &lhs);
template<typename T> SL_TENSOR_UM<T> &mac_erf(SL_TENSOR_UM<T> &lhs);

template<typename T> SL_TENSOR_UM<T> &mac_matmul(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs);
template<typename T> SL_TENSOR_UM<T> &mac_conv2d(SL_TENSOR_UM<T> &lhs, SL_TENSOR_UM<T> &rhs,
        SL_VECTOR<int> stride, SL_VECTOR<int> dilation, SL_VECTOR<int> padding = {0, 0, 0, 0});
template<typename T> SL_TENSOR_UM<T> &mac_broadcast(SL_TENSOR_UM<T> &input,
        SL_VECTOR<int> shape, SL_VECTOR<int> axis);

template<typename T> SL_TENSOR_UM<T> &mac_min(SL_TENSOR_UM<T> &input, const int axis);
