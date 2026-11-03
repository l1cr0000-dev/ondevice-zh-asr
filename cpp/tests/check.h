// 極簡測試巨集（不引入第三方測試框架）。
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: CHECK 失敗：%s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

#define CHECK_EQ(a, b)                                                                  \
  do {                                                                                  \
    const auto va = (a);                                                                \
    const auto vb = (b);                                                                \
    if (!(va == vb)) {                                                                  \
      std::fprintf(stderr, "%s:%d: CHECK_EQ 失敗：%s != %s\n", __FILE__, __LINE__, #a, #b); \
      ++g_failures;                                                                     \
    }                                                                                   \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= (tol))

#define TEST_MAIN_END() \
  return g_failures ? (std::fprintf(stderr, "%d 項失敗\n", g_failures), 1) : (std::puts("OK"), 0)
