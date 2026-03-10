#pragma once

#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>

namespace torch::aot_inductor {

struct KernelContext {
  std::string kernel_name;
  std::string python_stack;
  std::string compressed_python_stack;

  KernelContext(std::string name, std::string stack)
      : kernel_name(std::move(name)) {
    python_stack = trim_python_stack(stack);
    compressed_python_stack = compress_python_stack(python_stack);
  }

  KernelContext(const KernelContext&) = default;
  KernelContext& operator=(const KernelContext&) = default;
  KernelContext(KernelContext&&) = default;
  KernelContext& operator=(KernelContext&&) = default;

 private:
  static std::string trim_python_stack(std::string stack) {
    auto pos = stack.find_first_not_of('\n');
    if (pos != stack.npos) {
      stack.erase(0, pos);
      pos = stack.find_last_not_of('\n');
      if (pos != stack.npos) {
        stack.erase(pos + 1);
      }
    }
    return stack;
  }

  static std::string compress_python_stack(const std::string& stack) {
    namespace fs = std::filesystem;
    char function[1025];
    char filename[1025];
    uint32_t fileline;
    int ret, n, ws;
    const char* p;
    std::string compressed_stack;
    std::stringstream stream{stack};
    std::string line;
    std::string fmt = "File \"%1024[^\"]\", line %u, in %1024[^\n]\n%n";
    while (std::getline(stream, line)) {
      // check if new stack
      if (line.empty()) {
        compressed_stack += '\n';
        continue;
      }
      p = line.c_str();
      ws = 0;
      while (*p == ' ') {
        ++p;
        ++ws;
      }
      // check if new file
      if (ws != 0 && ws != 2) {
        return {};
      }
      ret = sscanf(p, fmt.c_str(), filename, &fileline, function, &n);
      if (ret != 3) {
        return {};
      }
      if (!std::getline(stream, line)) {
        return {};
      }
      p = line.c_str();
      ws = 0;
      while (*p == ' ') {
        ++p;
        ++ws;
      }
      // check if command
      if (ws != 4) {
        return {};
      }
      compressed_stack += std::string{function} + '[' + std::string{p} + ']';
      compressed_stack += '\n';
      compressed_stack += fs::path{filename}.filename();
      compressed_stack += '\n';
      compressed_stack += std::to_string(fileline);
      compressed_stack += '\n';
    }
    return compressed_stack;
  }
};

// Thread-local pointer
extern thread_local KernelContext* tls_kernel_context;

inline KernelContext* current_kernel_context() {
  return tls_kernel_context;
}

inline void set_kernel_context(KernelContext* ctx) {
  tls_kernel_context = ctx;
}

inline void clear_kernel_context() {
  tls_kernel_context = nullptr;
}

struct KernelContextGuard {
  KernelContextGuard(const std::string& name, const std::string& stack)
      : owned_context_(name, stack) {
    set_kernel_context(&owned_context_);
  }
  ~KernelContextGuard() {
    clear_kernel_context();
  }

  // Delete copy constructor and copy assignment operator
  KernelContextGuard(const KernelContextGuard&) = delete;
  KernelContextGuard& operator=(const KernelContextGuard&) = delete;

  KernelContextGuard(KernelContextGuard&&) = default;
  KernelContextGuard& operator=(KernelContextGuard&&) = delete;

 private:
  KernelContext owned_context_;
};

} // namespace torch::aot_inductor
