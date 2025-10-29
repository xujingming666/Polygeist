// #include <vector>
// #include <initializer_list>

int arr[] = {3,4,6};

void kernel_deriche(__ddr float *lhs, __ddr float *rhs, __ddr __output float *out, __ddr __output float *out1, int length, int j, int h) {
    // auto lhs_tensor = to_tensor(lhs, {length});
    arr[0] = length;
    arr[2] = length;
    SL_VECTOR<int> temp = {arr[0], arr[2], arr[2], arr[2]};
    // SL_VECTOR<int> temp1({length, i});
    int vlen = 1024;
    auto lhs_tensor = to_tensor(lhs, temp);
    auto rhs_tensor = to_tensor(rhs, temp);
    auto out_tensor = to_tensor(out, temp);
    auto out1_tensor = to_tensor(out1, temp);

    auto lhs_um = mac_load(lhs_tensor, {vlen, vlen, vlen, vlen}, {0, 0, 0, 0});
    auto rhs_um = mac_load(rhs_tensor, {vlen, vlen, vlen, vlen}, {0, 0, 0, 0});
    auto weight_um = mac_load(rhs_tensor, {vlen, vlen, 3, 3}, {0, 0, 0, 0});
    auto out_um = mac_load(out_tensor, {vlen, vlen, vlen, vlen}, {0, 0, 0, 0});
    auto out1_um = mac_load(out1_tensor, {vlen, vlen, vlen, vlen}, {0, 0, 0, 0});

    for (int i =0, j = 32;i<length; i += 32) {
        if (i > j) {
            out_um += mac_add(out_um, lhs_um);
            out_um += mac_sub(out_um, lhs_um);
            out_um += mac_mul(out_um, lhs_um);
            out_um += mac_div(out_um, lhs_um);
            // out_um = mac_div_unsigned(out_um, lhs_um);
            out_um = mac_min_signed(out_um, lhs_um);
            // out_um = mac_min_unsigned(out_um, lhs_um);
            out_um = mac_max_signed(out_um, lhs_um);
            // out_um = mac_max_unsigned(out_um, lhs_um);
            auto out_um2 = mac_powf(out_um, lhs_um);
            out1_um = mac_add(out1_um, out_um2);
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
            out1_um += mac_add(out1_um, lhs_um);
            out1_um += mac_conv2d(out1_um, weight_um, {1, 1}, {1, 1});
            out1_um += mac_sqrt(out1_um);
            out1_um += mac_add(out1_um, rhs_um);
        }
        j+=16;
    }
    mac_store(out_tensor, out_um, {0, 0});
    mac_store(out1_tensor, out1_um, {0, 0});
}

