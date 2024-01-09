#include <iostream>
#include <string>

#include <tsl/robin_map.h>

int main() {
    tsl::robin_map<std::string, int> map = {
        {"a", 1},
        {"b", 2},
        {"c", 3}
    };

    using RobinMapStorehash =
        tsl::robin_map<std::string, int, std::hash<std::string>, std::equal_to<>,
                       std::allocator<std::pair<std::string, int>>, true>;
}
