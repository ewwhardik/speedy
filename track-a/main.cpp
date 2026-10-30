// Speed Demon 2026, Track A (Ingest): per (UTC hour, endpoint) request count,
// nearest-rank p50/p95/p99 latency and 5xx count/ppm. See specs/a-ingest.md.
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct MappedFile {
  const char *data = "";
  size_t size = 0;
  explicit MappedFile(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    struct stat st {};
    if (fstat(fd, &st) == 0 && (size = st.st_size) > 0) {
      void *p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (p != MAP_FAILED) data = static_cast<const char *>(p);
    }
    close(fd);
  }
  ~MappedFile() {
    if (size && data) munmap(const_cast<char *>(data), size);
  }
  const char *end() const { return data + size; }
};

inline uint64_t integer(const char *&p) {
  uint64_t value = 0;
  while (static_cast<unsigned>(*p - '0') < 10)
    value = value * 10 + (*p++ - '0');
  return value;
}

template <class T> inline void number(std::string &out, T value) {
  char buffer[32];
  auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  out.append(buffer, result.ptr);
}

struct Group {
  std::vector<uint32_t> latencies;
  uint32_t errors = 0;
};
struct Endpoint {
  std::string_view name;
  std::array<Group, 72> hours;
};

int main(int argc, char **argv) {
  const char *data_path = "/data/requests.log", *out_path = "/out/summary.txt";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::string_view(argv[i]) == "--data") data_path = argv[i + 1];
    else if (std::string_view(argv[i]) == "--out") out_path = argv[i + 1];
  }

  MappedFile input(data_path);
  if (!input.size) return 1;

  std::unordered_map<std::string_view, uint32_t> ids;
  ids.reserve(2048);
  std::vector<Endpoint> endpoints;
  endpoints.reserve(2048);
  std::array<std::string, 72> hour_names;

  const char *p = input.data, *end = input.end();
  while (p < end) {
    unsigned day = (p[8] - '0') * 10 + p[9] - '0';
    unsigned hour = (p[11] - '0') * 10 + p[12] - '0';
    unsigned day_index = p[6] == '0' ? day - 30 : day + 1;
    unsigned bucket = day_index * 24 + hour - 18;
    if (hour_names[bucket].empty())
      hour_names[bucket] = std::string(p, 13) + ":00:00Z";
    p += 28;
    while (*p++ != ' ') {}
    const char *start = p;
    while (*p != ' ') p++;
    std::string_view key(start, p - start);
    auto [it, inserted] = ids.try_emplace(key, endpoints.size());
    if (inserted)
      endpoints.push_back(Endpoint{key, {}});
    Group &group = endpoints[it->second].hours[bucket];
    p++;
    group.errors += (*p == '5');
    p += 4;
    group.latencies.push_back(static_cast<uint32_t>(integer(p)));
    p++;
  }

  std::vector<uint32_t> order(endpoints.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto a, auto b) {
    return endpoints[a].name < endpoints[b].name;
  });

  std::string output;
  output.reserve(12 * 1024 * 1024);
  for (unsigned hour = 0; hour < 72; ++hour) {
    for (auto id : order) {
      auto &group = endpoints[id].hours[hour];
      auto &values = group.latencies;
      uint64_t n = values.size();
      if (!n) continue;
      size_t r50 = (50 * n + 99) / 100 - 1;
      size_t r95 = (95 * n + 99) / 100 - 1;
      size_t r99 = (99 * n + 99) / 100 - 1;
      if (n <= 128)
        std::sort(values.begin(), values.end());
      else {
        std::nth_element(values.begin(), values.begin() + r50, values.end());
        if (r95 > r50)
          std::nth_element(values.begin() + r50 + 1, values.begin() + r95, values.end());
        if (r99 > r95)
          std::nth_element(values.begin() + r95 + 1, values.begin() + r99, values.end());
      }
      output += hour_names[hour];
      output += ' ';
      output += endpoints[id].name;
      for (uint64_t value : {n, uint64_t(values[r50]), uint64_t(values[r95]),
                             uint64_t(values[r99]), uint64_t(group.errors),
                             uint64_t(group.errors) * 1000000 / n}) {
        output += ' ';
        number(output, value);
      }
      output += '\n';
    }
  }

  FILE *out_f = fopen(out_path, "wb");
  if (!out_f) return 1;
  fwrite(output.data(), 1, output.size(), out_f);
  fclose(out_f);
  return 0;
}
