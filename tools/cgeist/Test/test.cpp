// #include <vector>
// #include <initializer_list>

void kernel_deriche(__ddr float *lhs, __ddr float *rhs, __ddr __output float *out, __ddr __output float *out1, int length, int j, int h) {
    // auto lhs_tensor = to_tensor(lhs, {length});
    SL_VECTOR<int> temp = {length, length};
    // SL_VECTOR<int> temp1({length, i});

    auto lhs_tensor = to_tensor(lhs, temp);
    auto rhs_tensor = to_tensor(rhs, temp);
    auto out_tensor = to_tensor(out, temp);
    auto out1_tensor = to_tensor(out1, temp);

    auto lhs_um = mac_load(lhs_tensor, {16, 16}, {0, 0});
    auto rhs_um = mac_load(rhs_tensor, {16, 16}, {0, 0});
    
    auto out_um = mac_fill(0.0f, {16, 16});
    auto out1_um = mac_fill(0.1f, {16, 16});

    for (int i =0;i<length; i++) {
        if (i > j) {
            out_um = mac_add(out_um, lhs_um);
            out1_um = mac_add(out1_um, rhs_um);
        }
        for(int k =0; k < i; k++) {
            if (k > j) {
                out1_um = mac_add(out1_um, lhs_um);
                out1_um = mac_add(out1_um, lhs_um);
            } else {
                out_um += mac_add(out_um, rhs_um);
                out_um += mac_add(out_um, rhs_um);
            }
        }
        if (i > h) {
            out1_um = mac_add(out_um, lhs_um);
            out1_um = mac_add(out1_um, rhs_um);
        }
    }

    for (int i = 0 ; i < h; i++) {
        if (i > j) {
            break;
        }
        out_um += mac_add(out_um, rhs_um);
    }

    mac_store(out_tensor, out_um, {0, 0});
    mac_store(out1_tensor, out1_um, {0, 0});
    // SL_VECTOR1<int> temp2({1, 1});
    // int arr[] = {1, 2, 3}; 

    // std::initializer_list<int> init = {length, i};
    // std::vector<int> init = {i, length};
    // int k = i*length;
}

