#ifndef LEXOCRAFT_SHORT_PRINT_HPP
#define LEXOCRAFT_SHORT_PRINT_HPP

#include <iomanip>
#include <iostream>

#include <Eigen/Eigen>
#include <string>

namespace lc {
    /*
     Only prints the first N rows or columns of the eigen vector or matrix as list. If there are
     more than N, then simply end the prnting with a '...'.

    Ex:
    [1, 2, 3, 4, ...]

    [  8,   2,   7,   1,   2, ...]
    [  1,   2,   3,   4,   9, ...]
    [  1,   2,   3,   4,   5, ...]
    [  0,   2,   3,   4,   5, ...]
    [..., ..., ..., ..., ..., ...]
    */
    template <typename T>
    std::string fancy_eigen_vector_str(const T& obj, int max_cols = 5, int decimal_places = 2) {
        std::stringstream sstream;

        sstream << "[";
        for (int i = 0; i < std::min(static_cast<int>(obj.size()), max_cols); ++i) {
            sstream << std::fixed << std::setprecision(decimal_places) << obj(i)
                    << (i < std::min(static_cast<int>(obj.size()), max_cols) - 1 ? ", " : "");
        }
        sstream << (static_cast<int>(obj.size()) > max_cols ? " ...]" : "]");

        return sstream.str();
    }

    template <typename T>
    std::string fancy_eigen_matrix_str(const T& obj, int max_rows = 5, int max_cols = 5,
                                       int decimal_places = 2) {
        std::stringstream sstream;

        for (int i = 0; i < std::min(static_cast<int>(obj.rows()), max_rows); ++i) {
            sstream << "[";
            for (int j = 0; j < std::min(static_cast<int>(obj.cols()), max_cols); ++j) {
                sstream << std::fixed << std::setprecision(decimal_places)
                        << std::setw(4 + decimal_places + 1) << obj(i, j)
                        << (j < std::min(static_cast<int>(obj.cols()), max_cols) - 1 ? ", " : "");
            }
            sstream << (static_cast<int>(obj.cols()) > max_cols ? " ...]" : "]")
                    << (i < std::min(static_cast<int>(obj.rows()), max_rows) - 1 ? "\n" : "");
        }
        sstream << (static_cast<int>(obj.rows()) > max_rows ? "\n..." : "");

        return sstream.str();
    }
} // namespace lc

#endif
