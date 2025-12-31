// NOLINTBEGIN
#pragma once
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Signal {
  const char *name;
  unsigned offset;
  unsigned numBits;
  enum Type { Input, Output, Register, Memory, Wire } type;
  // for memories:
  unsigned stride;
  unsigned depth;
};

struct Hierarchy {
  const char *name;
  unsigned numStates;
  unsigned numChildren;
  Signal *states;
  Hierarchy *children;
};

template <unsigned N>
struct Bytes {
  uint8_t byte[N];
};
template <typename T, unsigned Stride, unsigned Depth>
struct Memory {
  union {
    T data;
    uint8_t stride[Stride];
  } words[Depth];
};

template <class ModelLayout>
class ValueChangeDump {
public:
  ValueChangeDump(std::basic_ostream<char> &os, const uint8_t *state)
      : os(os), state(state) {}

  void writeHeader(bool withHierarchy = true) {
    os << "$date\n    October 21, 2015\n$end\n";
    os << "$version\n    Some cryptic MLIR magic\n$end\n";
    os << "$timescale 1ns $end\n";

    os << "$scope module " << ModelLayout::name << " $end\n";

    auto writeSignal = [&](const Signal &state) {
      if (state.type != Signal::Memory) {
        auto &signal =
            allocSignal(state, state.offset, (state.numBits + 7) / 8);
        if (state.type == Signal::Register) {
          os << "$var reg " << state.numBits << " " << signal.abbrev << " "
             << state.name;
        } else {
          os << "$var wire " << state.numBits << " " << signal.abbrev << " "
             << state.name;
        }
        if (state.numBits > 1)
          os << " [" << (state.numBits - 1) << ":0]";
        os << " $end\n";
      } else {
        for (unsigned i = 0; i < state.depth; ++i) {
          auto &signal = allocSignal(state, state.offset + i * state.stride,
                                     (state.numBits + 7) / 8);
          os << "$var reg " << state.numBits << " " << signal.abbrev << " "
             << state.name << "[" << i << "]";
          if (state.numBits > 1)
            os << " [" << (state.numBits - 1) << ":0]";
          os << " $end\n";
        }
      }
    };

    std::function<void(const Hierarchy &)> writeHierarchy =
        [&](const Hierarchy &hierarchy) {
          os << "$scope module " << hierarchy.name << " $end\n";
          for (unsigned i = 0; i < hierarchy.numStates; ++i)
            writeSignal(hierarchy.states[i]);
          for (unsigned i = 0; i < hierarchy.numChildren; ++i)
            writeHierarchy(hierarchy.children[i]);
          os << "$upscope $end\n";
        };

    for (auto &port : ModelLayout::io)
      writeSignal(port);
    if (withHierarchy)
      writeHierarchy(ModelLayout::hierarchy);

    os << "$upscope $end\n";
    os << "$enddefinitions $end\n";
  }

  void writeValues(bool includeUnchanged = false) {
    for (auto &signal : signals) {
      const uint8_t *valNew = state + signal.offset;
      uint8_t *valOld = &previousValues[0] + signal.previousOffset;
      size_t numBytes = (signal.state.numBits + 7) / 8;
      bool unchanged = std::equal(valNew, valNew + numBytes, valOld);
      if (unchanged && !includeUnchanged)
        continue;
      if (signal.state.numBits > 1)
        os << 'b';
      for (unsigned n = signal.state.numBits; n > 0; --n)
        os << (valNew[(n - 1) / 8] & (1 << ((n - 1) % 8)) ? '1' : '0');
      if (signal.state.numBits > 1)
        os << ' ';
      os << signal.abbrev << "\n";
      std::copy(valNew, valNew + numBytes, valOld);
    }
  }

  void writeDumpvars() {
    os << "$dumpvars\n";
    writeValues(true);
  }

  void writeTimestep(size_t timeIncrement) {
    time += timeIncrement;
    os << "#" << time << "\n";
    writeValues();
  }

  size_t time = 0;

private:
  struct VcdSignal {
    std::string abbrev;
    unsigned offset;
    const Signal &state;
    unsigned previousOffset;
  };

  VcdSignal &allocSignal(const Signal &state, unsigned offset,
                         unsigned numBytes) {
    std::string abbrev;
    unsigned rest = signals.size() + 1;
    while (rest != 0) {
      uint8_t c = (rest % 84) + 33;
      if (c >= '0')
        c += 10;
      abbrev += c;
      rest /= 84;
    }
    signals.push_back(
        VcdSignal{abbrev, offset, state, unsigned(previousValues.size())});
    previousValues.resize(previousValues.size() + numBytes);
    return signals.back();
  }

  std::basic_ostream<char> &os;
  const uint8_t *state;
  std::vector<VcdSignal> signals;
  std::vector<uint8_t> previousValues;
};

// NOLINTEND

extern "C" int32_t circt_sv_strcmp(const char *lhs, const char *rhs) {
  if (!lhs)
    lhs = "";
  if (!rhs)
    rhs = "";
  return std::strcmp(lhs, rhs);
}

extern "C" int32_t circt_sv_string_len(const char *str) {
  if (!str)
    return 0;
  return static_cast<int32_t>(std::strlen(str));
}

extern "C" uint8_t circt_sv_string_getc(const char *str, int32_t idx) {
  if (!str)
    return 0;
  if (idx < 0)
    return 0;
  size_t len = std::strlen(str);
  size_t pos = static_cast<size_t>(idx);
  if (pos >= len)
    return 0;
  return static_cast<uint8_t>(static_cast<unsigned char>(str[pos]));
}

extern "C" const char *circt_sv_string_substr(const char *str, int32_t start,
                                              int32_t end) {
  if (!str)
    str = "";
  if (end < start)
    return "";
  if (start < 0)
    start = 0;

  size_t len = std::strlen(str);
  size_t startPos = static_cast<size_t>(start);
  if (startPos >= len)
    return "";

  size_t endPos = static_cast<size_t>(end);
  if (endPos >= len)
    endPos = len - 1;
  if (endPos < startPos)
    return "";

  size_t outLen = endPos - startPos + 1;
  char *out = static_cast<char *>(std::malloc(outLen + 1));
  if (!out)
    return "";
  std::memcpy(out, str + startPos, outLen);
  out[outLen] = '\0';
  return out;
}

//===----------------------------------------------------------------------===//
// Minimal SV randomization runtime (deterministic, seedable)
//===----------------------------------------------------------------------===//

static uint64_t circt_sv_rand_state = 1;

static inline uint64_t circt_sv_splitmix64_next() {
  // SplitMix64 (public domain).
  uint64_t z = (circt_sv_rand_state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

extern "C" void circt_sv_rand_seed(uint64_t seed) { circt_sv_rand_state = seed; }

extern "C" uint64_t circt_sv_rand_get_state_u64() { return circt_sv_rand_state; }

extern "C" void circt_sv_rand_set_state_u64(uint64_t state) {
  circt_sv_rand_state = state;
}

extern "C" uint32_t circt_sv_urandom_u32() {
  return static_cast<uint32_t>(circt_sv_splitmix64_next());
}

extern "C" uint32_t circt_sv_urandom_seed_u32(uint32_t seed) {
  circt_sv_rand_seed(static_cast<uint64_t>(seed));
  return circt_sv_urandom_u32();
}

extern "C" int32_t circt_sv_random_i32() {
  return static_cast<int32_t>(circt_sv_urandom_u32());
}

extern "C" int32_t circt_sv_random_seed_i32(int32_t seed) {
  circt_sv_rand_seed(static_cast<uint64_t>(static_cast<uint32_t>(seed)));
  return circt_sv_random_i32();
}

extern "C" int32_t circt_sv_rand_range_i32(int32_t lo, int32_t hi) {
  if (hi < lo)
    return 0;
  uint64_t range = static_cast<uint64_t>(
      static_cast<int64_t>(hi) - static_cast<int64_t>(lo) + 1);
  if (range == 0)
    return lo;

  const uint64_t limit = (~0ull) - ((~0ull) % range);
  uint64_t r = 0;
  do {
    r = circt_sv_splitmix64_next();
  } while (r >= limit);
  int64_t out = static_cast<int64_t>(lo) + static_cast<int64_t>(r % range);
  return static_cast<int32_t>(out);
}

extern "C" void circt_sv_srandom_i32(int32_t seed) {
  circt_sv_rand_seed(static_cast<uint64_t>(static_cast<uint32_t>(seed)));
}

extern "C" const char *circt_sv_get_randstate_str() {
  // Return a stable decimal representation of the RNG state.
  char buf[32];
  int n = std::snprintf(buf, sizeof(buf), "%llu",
                        static_cast<unsigned long long>(circt_sv_rand_state));
  if (n <= 0)
    return "";
  size_t len = static_cast<size_t>(n);
  char *out = static_cast<char *>(std::malloc(len + 1));
  if (!out)
    return "";
  std::memcpy(out, buf, len + 1);
  return out;
}

extern "C" void circt_sv_set_randstate_str(const char *state) {
  if (!state || !*state)
    return;
  char *end = nullptr;
  unsigned long long v = std::strtoull(state, &end, 0);
  if (end == state)
    return;
  circt_sv_rand_state = static_cast<uint64_t>(v);
}

//===----------------------------------------------------------------------===//
// Minimal SV class runtime shims
//===----------------------------------------------------------------------===//

struct CirctSvClassObject {
  int32_t typeId = 0;
  std::unordered_map<int32_t, int32_t> i32Fields;
  std::unordered_map<int32_t, std::string> strFields;
};

static std::unordered_map<int32_t, CirctSvClassObject> circt_sv_class_objects;

static int32_t circt_sv_class_alloc_handle() {
  static uint32_t nextHandle = 1;
  uint32_t handle = nextHandle++;
  if (handle == 0)
    handle = nextHandle++;
  return static_cast<int32_t>(handle);
}

extern "C" int32_t circt_sv_class_alloc(int32_t typeId) {
  int32_t handle = circt_sv_class_alloc_handle();
  circt_sv_class_objects.emplace(handle, CirctSvClassObject{typeId, {}});
  return handle;
}

extern "C" int32_t circt_sv_class_new() { return circt_sv_class_alloc(0); }

extern "C" int32_t circt_sv_class_get_type(int32_t handle) {
  auto it = circt_sv_class_objects.find(handle);
  if (it == circt_sv_class_objects.end())
    return 0;
  return it->second.typeId;
}

extern "C" int32_t circt_sv_class_get_i32(int32_t handle, int32_t fieldId) {
  auto it = circt_sv_class_objects.find(handle);
  if (it == circt_sv_class_objects.end())
    return 0;
  auto jt = it->second.i32Fields.find(fieldId);
  if (jt == it->second.i32Fields.end())
    return 0;
  return jt->second;
}

extern "C" const char *circt_sv_class_get_str(int32_t handle, int32_t fieldId) {
  auto it = circt_sv_class_objects.find(handle);
  if (it == circt_sv_class_objects.end())
    return "";
  auto jt = it->second.strFields.find(fieldId);
  if (jt == it->second.strFields.end())
    return "";
  return jt->second.c_str();
}

extern "C" void circt_sv_class_set_i32(int32_t handle, int32_t fieldId,
                                       int32_t value) {
  auto it = circt_sv_class_objects.find(handle);
  if (it == circt_sv_class_objects.end())
    it = circt_sv_class_objects.emplace(handle, CirctSvClassObject{}).first;
  it->second.i32Fields[fieldId] = value;
}

extern "C" void circt_sv_class_set_str(int32_t handle, int32_t fieldId,
                                       const char *value) {
  auto it = circt_sv_class_objects.find(handle);
  if (it == circt_sv_class_objects.end())
    it = circt_sv_class_objects.emplace(handle, CirctSvClassObject{}).first;
  it->second.strFields[fieldId] = value ? value : "";
}

//===----------------------------------------------------------------------===//
// Minimal SV dynamic container runtime shims
//===----------------------------------------------------------------------===//

static int32_t circt_sv_alloc_handle() {
  static uint32_t nextHandle = 1;
  uint32_t handle = nextHandle++;
  if (handle == 0)
    handle = nextHandle++;
  return static_cast<int32_t>(handle);
}

static std::unordered_map<int32_t, std::vector<int32_t>> circt_sv_dynarrays_i32;
static std::unordered_map<int32_t, std::deque<int32_t>> circt_sv_queues_i32;
static std::unordered_map<int32_t, std::unordered_map<std::string, int32_t>>
    circt_sv_assoc_str_i32;

extern "C" int32_t circt_sv_dynarray_alloc_i32(int32_t size) {
  if (size < 0)
    size = 0;
  int32_t handle = circt_sv_alloc_handle();
  circt_sv_dynarrays_i32.emplace(handle,
                                std::vector<int32_t>(static_cast<size_t>(size)));
  return handle;
}

extern "C" int32_t circt_sv_dynarray_size_i32(int32_t handle) {
  auto it = circt_sv_dynarrays_i32.find(handle);
  if (it == circt_sv_dynarrays_i32.end())
    return 0;
  return static_cast<int32_t>(it->second.size());
}

extern "C" int32_t circt_sv_dynarray_get_i32(int32_t handle, int32_t idx) {
  auto it = circt_sv_dynarrays_i32.find(handle);
  if (it == circt_sv_dynarrays_i32.end())
    return 0;
  if (idx < 0)
    return 0;
  size_t pos = static_cast<size_t>(idx);
  if (pos >= it->second.size())
    return 0;
  return it->second[pos];
}

extern "C" void circt_sv_dynarray_set_i32(int32_t handle, int32_t idx,
                                         int32_t value) {
  auto it = circt_sv_dynarrays_i32.find(handle);
  if (it == circt_sv_dynarrays_i32.end())
    return;
  if (idx < 0)
    return;
  size_t pos = static_cast<size_t>(idx);
  if (pos >= it->second.size())
    return;
  it->second[pos] = value;
}

extern "C" int32_t circt_sv_queue_alloc_i32() {
  int32_t handle = circt_sv_alloc_handle();
  circt_sv_queues_i32.emplace(handle, std::deque<int32_t>{});
  return handle;
}

extern "C" int32_t circt_sv_queue_size_i32(int32_t handle) {
  auto it = circt_sv_queues_i32.find(handle);
  if (it == circt_sv_queues_i32.end())
    return 0;
  return static_cast<int32_t>(it->second.size());
}

extern "C" void circt_sv_queue_push_back_i32(int32_t handle, int32_t value) {
  auto it = circt_sv_queues_i32.find(handle);
  if (it == circt_sv_queues_i32.end())
    return;
  it->second.push_back(value);
}

extern "C" void circt_sv_queue_push_front_i32(int32_t handle, int32_t value) {
  auto it = circt_sv_queues_i32.find(handle);
  if (it == circt_sv_queues_i32.end())
    return;
  it->second.push_front(value);
}

extern "C" int32_t circt_sv_queue_pop_front_i32(int32_t handle) {
  auto it = circt_sv_queues_i32.find(handle);
  if (it == circt_sv_queues_i32.end() || it->second.empty())
    return 0;
  int32_t value = it->second.front();
  it->second.pop_front();
  return value;
}

extern "C" int32_t circt_sv_queue_pop_back_i32(int32_t handle) {
  auto it = circt_sv_queues_i32.find(handle);
  if (it == circt_sv_queues_i32.end() || it->second.empty())
    return 0;
  int32_t value = it->second.back();
  it->second.pop_back();
  return value;
}

extern "C" int32_t circt_sv_assoc_alloc_str_i32() {
  int32_t handle = circt_sv_alloc_handle();
  circt_sv_assoc_str_i32.emplace(handle,
                                std::unordered_map<std::string, int32_t>{});
  return handle;
}

extern "C" int32_t circt_sv_assoc_exists_str_i32(int32_t handle,
                                                 const char *key) {
  auto it = circt_sv_assoc_str_i32.find(handle);
  if (it == circt_sv_assoc_str_i32.end())
    return 0;
  std::string k = key ? key : "";
  return it->second.count(k) ? 1 : 0;
}

extern "C" int32_t circt_sv_assoc_get_str_i32(int32_t handle, const char *key) {
  auto it = circt_sv_assoc_str_i32.find(handle);
  if (it == circt_sv_assoc_str_i32.end())
    return 0;
  std::string k = key ? key : "";
  auto jt = it->second.find(k);
  if (jt == it->second.end())
    return 0;
  return jt->second;
}

extern "C" void circt_sv_assoc_set_str_i32(int32_t handle, const char *key,
                                          int32_t value) {
  auto it = circt_sv_assoc_str_i32.find(handle);
  if (it == circt_sv_assoc_str_i32.end())
    return;
  std::string k = key ? key : "";
  it->second[k] = value;
}

//===----------------------------------------------------------------------===//
// Minimal UVM runtime shims
//===----------------------------------------------------------------------===//

static bool circt_env_truthy(const char *name, bool defaultValue) {
  const char *val = std::getenv(name);
  if (!val)
    return defaultValue;
  while (*val == ' ' || *val == '\t')
    ++val;
  if (!*val)
    return defaultValue;
  if ((val[0] == '0' && val[1] == '\0') || (val[0] == 'n' || val[0] == 'N') ||
      (val[0] == 'f' || val[0] == 'F') || (val[0] == 'o' || val[0] == 'O'))
    return false;
  return true;
}

static bool circt_uvm_shims_enabled() {
  static bool enabled = circt_env_truthy("CIRCT_UVM_SHIMS", true);
  return enabled;
}

static bool circt_uvm_phase_all_done_state = false;
static std::unordered_map<std::string, int32_t> circt_uvm_resource_db;

static std::string circt_uvm_resource_db_key(const char *scope, const char *name) {
  std::string key;
  if (scope)
    key += scope;
  key.push_back('\0');
  if (name)
    key += name;
  return key;
}

extern "C" void circt_uvm_run_test(const char *test_name) {
  (void)test_name;
  // CIRCT does not yet execute the class-based UVM scheduler. This is a minimal
  // hook used by the sv-tests UVM M0 wrapper to make top-level `run_test()`
  // calls measurable (non-vacuous).
  if (!circt_uvm_shims_enabled())
    return;
  circt_uvm_phase_all_done_state = true;
}

extern "C" int32_t circt_uvm_coreservice_get() { return 1; }

extern "C" int32_t circt_uvm_coreservice_get_root(int32_t coreservice) {
  (void)coreservice;
  return 1;
}

extern "C" void circt_uvm_root_run_test(int32_t root, const char *test_name) {
  (void)root;
  (void)test_name;
  if (!circt_uvm_shims_enabled())
    return;
  circt_uvm_phase_all_done_state = true;
}

extern "C" void circt_uvm_resource_db_set(const char *scope, const char *name,
                                         int32_t value) {
  circt_uvm_resource_db[circt_uvm_resource_db_key(scope, name)] = value;
}

extern "C" int32_t circt_uvm_report_server_get_server() { return 1; }

extern "C" int32_t circt_uvm_root_get() { return 1; }

extern "C" int32_t circt_uvm_get_severity_count(int32_t severity) {
  // The UVM M0 wrapper only queries UVM_ERROR and UVM_FATAL. Until CIRCT lowers
  // the real report server, treat the counts as zero.
  (void)severity;
  return 0;
}

extern "C" bool circt_uvm_phase_all_done() {
  return circt_uvm_phase_all_done_state;
}
