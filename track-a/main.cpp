#include "common.hpp"
#include <array>
#include <numeric>
#include <unordered_map>

struct Group {
    std::vector<uint32_t> latencies;
    uint32_t errors = 0;
};
struct Endpoint {
    std::string_view name;
    std::array<Group, 72> hours;
};

int main(int argc, char** argv) {
    try {
        Paths paths(argc, argv);
        MappedFile input(paths.data + "/requests.log");
        std::unordered_map<std::string_view, uint32_t> ids;
        ids.reserve(2048);
        std::vector<Endpoint> endpoints;
        endpoints.reserve(2048);
        std::array<std::string, 72> hour_names;
        const char* p = input.data;
        while (p < input.end()) {
            // Fixed timestamp layout and the 72-hour span are spec guarantees.
            unsigned day = (p[8] - '0') * 10 + p[9] - '0';
            unsigned hour = (p[11] - '0') * 10 + p[12] - '0';
            unsigned day_index = p[6] == '0' ? day - 30 : day + 1;
            unsigned bucket = day_index * 24 + hour - 18;
            if (hour_names[bucket].empty()) hour_names[bucket] = std::string(p, 13) + ":00:00Z";
            p += 28;
            while (*p++ != ' ') {} // skip method
            const char* start = p;
            while (*p != ' ') ++p;
            std::string_view key(start, p - start);
            auto [it, inserted] = ids.try_emplace(key, endpoints.size());
            if (inserted) endpoints.push_back(Endpoint{key, {}});
            Group& group = endpoints[it->second].hours[bucket];
            ++p;
            group.errors += *p == '5';
            p += 4;
            group.latencies.push_back(static_cast<uint32_t>(integer(p)));
            ++p;
        }
        std::vector<uint32_t> order(endpoints.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](auto a, auto b) { return endpoints[a].name < endpoints[b].name; });
        std::string output;
        output.reserve(12 * 1024 * 1024);
        for (unsigned hour = 0; hour < 72; ++hour) {
            for (auto id : order) {
                auto& group = endpoints[id].hours[hour];
                auto& values = group.latencies;
                uint64_t n = values.size();
                if (!n) continue;
                size_t r50 = (50 * n + 99) / 100 - 1;
                size_t r95 = (95 * n + 99) / 100 - 1;
                size_t r99 = (99 * n + 99) / 100 - 1;
                if (n <= 128) std::sort(values.begin(), values.end());
                else {
                    // Partition in-place
                    std::nth_element(values.begin(), values.begin() + r50, values.end());
                    if (r95 > r50) std::nth_element(values.begin() + r50 + 1, values.begin() + r95, values.end());
                    if (r99 > r95) std::nth_element(values.begin() + r95 + 1, values.begin() + r99, values.end());
                }
                output += hour_names[hour];
                output += ' ';
                output += endpoints[id].name;
                for (uint64_t value : {n, uint64_t(values[r50]), uint64_t(values[r95]), uint64_t(values[r99]), uint64_t(group.errors), uint64_t(group.errors) * 1000000 / n}) {
                    output += ' ';
                    number(output, value);
                }
                output += '\n';
            }
        }
        write_output(paths.out + "/summary.txt", output);
    } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
