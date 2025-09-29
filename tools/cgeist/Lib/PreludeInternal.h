// #include <vector>

#define __um  __attribute__((address_space(3)))
#define __register  __attribute__((address_space(4)))

// Annotation to indicate an argument as output
#define __output __attribute__((annotate("output")))

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

template<typename T>
struct SL_VECTOR {
    SL_VECTOR(T...);
    T &operator()(int...);
    T &operator+(SL_VECTOR<T> rhs);
    T &operator+=(SL_VECTOR<T> rhs);
    T &operator=(SL_VECTOR<T> rhs);
} __register;

template<typename T>
struct SL_VECTOR1 {
    SL_VECTOR1(int, int);
    T &operator()(int...);
    T &operator+(SL_VECTOR1<T> rhs);
    T &operator+=(SL_VECTOR1<T> rhs);
    T &operator=(SL_VECTOR1<T> rhs);
} __register;

template<typename T> SL_TENSOR_UM<T> to_tensor(T *ptr, SL_VECTOR<int> shape);

template<typename T> SL_TENSOR_UM<T> mac_add(SL_TENSOR_UM<T> lhs, SL_TENSOR_UM<T> rhs);


