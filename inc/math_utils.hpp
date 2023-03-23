#include <vector>

template<bool a_transpose>
std::vector<double> dot_product(const std::vector<std::vector<double>>& a, const std::vector<double>& b) {
    size_t result_size = a_transpose ? a[0].size() : a.size();

    std::vector<double> result(result_size, 1);

    for (size_t i = 0; i < result_size; i++) {
        double sum = 0;

        for (size_t j = 0; j < b.size(); j++) {
            sum += a[a_transpose ? j : i][a_transpose ? i : j] * b[j];
        }

        result[i] = sum;
    }

    return result;
}