// #include <vector>
// #include <initializer_list>

int arr[] = {3,4,6};

void kernel_deriche(__ddr float *lhs, __ddr float *rhs, __ddr __output float *out, __ddr __output float *out1, int length, int j, int h) {
    // auto lhs_tensor = to_tensor(lhs, {length});
    arr[0] = length;
    arr[2] = length;
    SL_VECTOR<int> temp = {arr[0], arr[2]};
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
            out_um = mac_sub(out_um, lhs_um);
            out_um = mac_mul(out_um, lhs_um);
            out_um = mac_div(out_um, lhs_um);
            // out_um = mac_div_unsigned(out_um, lhs_um);
            out_um = mac_min_signed(out_um, lhs_um);
            // out_um = mac_min_unsigned(out_um, lhs_um);
            out_um = mac_max_signed(out_um, lhs_um);
            // out_um = mac_max_unsigned(out_um, lhs_um);
            out_um = mac_powf(out_um, lhs_um);
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

    out_um = mac_exp(out_um);
    out_um = mac_log(out_um);
    out_um = mac_abs(out_um);
    out_um = mac_ceil(out_um);
    out_um = mac_floor(out_um);
    out_um = mac_negf(out_um);
    out_um = mac_reciprocal(out_um);
    out_um = mac_round(out_um);
    out_um = mac_sqrt(out_um);
    out_um = mac_rsqrt(out_um);
    out_um = mac_square(out_um);
    out_um = mac_tanh(out_um);
    out_um = mac_erf(out_um);

    mac_store(out_tensor, out_um, {0, 0});
    mac_store(out1_tensor, out1_um, {0, 0});
    // SL_VECTOR1<int> temp2({1, 1});
    // int arr[] = {1, 2, 3}; 

    // std::initializer_list<int> init = {length, i};
    // std::vector<int> init = {i, length};
    // int k = i*length;
}

