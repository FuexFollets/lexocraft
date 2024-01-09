#include <iostream>
#include <string>

#include <Eigen/Eigen>
#include <tsl/robin_map.h>

int main() {
    tsl::robin_map<std::string, int> map = {
        {"a", 1},
        {"b", 2},
        {"c", 3}
    };

    std::cout << "int:\n";
    std::cout << "a: " << map ["a"] << "\n";
    std::cout << "b: " << map ["b"] << "\n";
    std::cout << "c: " << map ["c"] << "\n";
    std::cout << "d: " << map ["d"] << "\n";

    tsl::robin_map<std::string, Eigen::Vector3f> vector_map = {
        {"a", Eigen::Vector3f(1, 2, 3)},
        {"b", Eigen::Vector3f(4, 5, 6)},
        {"c", Eigen::Vector3f(7, 8, 9)}
    };

    std::cout << "Vector3f:\n";
    std::cout << "a: " << vector_map ["a"] << "\n";
    std::cout << "b: " << vector_map ["b"] << "\n";
    std::cout << "c: " << vector_map ["c"] << "\n";
    std::cout << "d: " << vector_map ["d"] << "\n";
}
