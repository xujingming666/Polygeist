// #include <vector>
// #include <initializer_list>

void kernel_deriche(float *lhs, float *rhs, __output float *out, __output float *out1, int length, int j, int h) {
    // auto lhs_tensor = to_tensor(lhs, {length});
    SL_VECTOR<int> temp = {length};
    // SL_VECTOR<int> temp1({length, i});

    auto lhs_tensor = to_tensor(lhs, temp);
    auto rhs_tensor = to_tensor(rhs, temp);
    auto out_tensor = to_tensor(out, temp);
    auto out1_tensor = to_tensor(out1, temp);

    for (int i =0;i<length; i++) {
        if (i > j) {
            out1_tensor = mac_add(out_tensor, lhs_tensor);
            out1_tensor = mac_add(out1_tensor, rhs_tensor);
        }
        for(int k =0; k < i; k++) {
            if (k > j) {
                out1_tensor = mac_add(out1_tensor, lhs_tensor);
                out1_tensor = mac_add(out1_tensor, lhs_tensor);
            } else {
                out_tensor += mac_add(out_tensor, rhs_tensor);
                out_tensor += mac_add(out_tensor, rhs_tensor);
            }
        }
        if (i > h) {
            out1_tensor = mac_add(out_tensor, lhs_tensor);
            out1_tensor = mac_add(out1_tensor, rhs_tensor);
        }
    }

    for (int i = 0 ; i < h; i++) {
        if (i > j) {
            break;
        }
        out_tensor += mac_add(out_tensor, rhs_tensor);
    }

    // SL_VECTOR1<int> temp2({1, 1});
    // int arr[] = {1, 2, 3}; 

    // std::initializer_list<int> init = {length, i};
    // std::vector<int> init = {i, length};
    // int k = i*length;
}

