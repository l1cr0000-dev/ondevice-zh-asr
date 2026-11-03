#include "fbank.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ozasr {

namespace {

constexpr double kPi = 3.14159265358979323846;

double mel_scale(double hz) { return 1127.0 * std::log(1.0 + hz / 700.0); }

}  // namespace

Fbank::Fbank(const FbankOptions& opts) : opts_(opts) {
  const int n = opts_.frame_length;
  window_.resize(n);
  for (int i = 0; i < n; ++i) window_[i] = 0.54 - 0.46 * std::cos(2.0 * kPi * i / (n - 1));

  const int fft_n = opts_.fft_size;
  cos_.resize(fft_n / 2);
  sin_.resize(fft_n / 2);
  for (int i = 0; i < fft_n / 2; ++i) {
    cos_[i] = std::cos(2.0 * kPi * i / fft_n);
    sin_[i] = -std::sin(2.0 * kPi * i / fft_n);
  }
  bitrev_.resize(fft_n);
  int bits = 0;
  while ((1 << bits) < fft_n) ++bits;
  for (int i = 0; i < fft_n; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b)
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    bitrev_[i] = r;
  }

  // mel 濾波器組：三角形在 mel 軸上等寬，只用 0..fft/2-1 的 bin（不含 Nyquist），與 kaldi 相同
  const int num_fft_bins = fft_n / 2;
  const double nyquist = 0.5 * opts_.sample_rate;
  const double high_freq = nyquist;  // high_freq = 0 → Nyquist
  const double mel_low = mel_scale(opts_.low_freq);
  const double mel_high = mel_scale(high_freq);
  const double delta = (mel_high - mel_low) / (opts_.num_bins + 1);
  const double bin_width = static_cast<double>(opts_.sample_rate) / fft_n;
  filters_.resize(opts_.num_bins);
  for (int b = 0; b < opts_.num_bins; ++b) {
    const double left = mel_low + b * delta;
    const double center = mel_low + (b + 1) * delta;
    const double right = mel_low + (b + 2) * delta;
    Filter& f = filters_[b];
    f.first = -1;
    for (int i = 0; i < num_fft_bins; ++i) {
      const double mel = mel_scale(bin_width * i);
      double w = 0.0;
      if (mel > left && mel < right) {
        w = mel <= center ? (mel - left) / (center - left) : (right - mel) / (right - center);
      }
      if (w > 0.0) {
        if (f.first < 0) f.first = i;
        f.weights.push_back(w);
      } else if (f.first >= 0) {
        break;  // 三角形連續，離開後不會再回來
      }
    }
    if (f.first < 0) f.first = 0;
  }
}

int Fbank::num_frames(size_t num_samples) const {
  if (num_samples < static_cast<size_t>(opts_.frame_length)) return 0;
  return static_cast<int>(1 + (num_samples - opts_.frame_length) / opts_.frame_shift);
}

void Fbank::fft(std::vector<double>& re, std::vector<double>& im) const {
  const int n = opts_.fft_size;
  for (int i = 0; i < n; ++i) {
    const int j = bitrev_[i];
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len / 2;
    const int step = n / len;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < half; ++k) {
        const double wr = cos_[k * step], wi = sin_[k * step];
        const int a = i + k, b = i + k + half;
        const double tr = re[b] * wr - im[b] * wi;
        const double ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

void Fbank::compute(const float* samples, size_t n, std::vector<float>& out) const {
  const int frames = num_frames(n);
  out.assign(static_cast<size_t>(frames) * opts_.num_bins, 0.0f);
  const int len = opts_.frame_length;
  const int fft_n = opts_.fft_size;
  const double eps = std::numeric_limits<float>::epsilon();
  std::vector<double> re(fft_n), im(fft_n), x(len);
  for (int f = 0; f < frames; ++f) {
    const float* src = samples + static_cast<size_t>(f) * opts_.frame_shift;
    double mean = 0.0;
    for (int i = 0; i < len; ++i) {
      x[i] = src[i];
      mean += x[i];
    }
    mean /= len;
    for (int i = 0; i < len; ++i) x[i] -= mean;  // remove_dc_offset
    for (int i = len - 1; i > 0; --i) x[i] -= opts_.preemph * x[i - 1];
    x[0] -= opts_.preemph * x[0];
    std::fill(re.begin(), re.end(), 0.0);
    std::fill(im.begin(), im.end(), 0.0);
    for (int i = 0; i < len; ++i) re[i] = x[i] * window_[i];
    fft(re, im);
    float* dst = out.data() + static_cast<size_t>(f) * opts_.num_bins;
    for (int b = 0; b < opts_.num_bins; ++b) {
      const Filter& flt = filters_[b];
      double e = 0.0;
      for (size_t k = 0; k < flt.weights.size(); ++k) {
        const int i = flt.first + static_cast<int>(k);
        e += flt.weights[k] * (re[i] * re[i] + im[i] * im[i]);
      }
      dst[b] = static_cast<float>(std::log(std::max(e, eps)));
    }
  }
}

void lfr_cmvn(const std::vector<float>& feats, int num_frames, int dim, int m, int shift,
              const std::vector<float>& neg_mean, const std::vector<float>& inv_std,
              std::vector<float>& out, int& out_frames) {
  if (num_frames <= 0) {
    out.clear();
    out_frames = 0;
    return;
  }
  out_frames = 1 + (num_frames - 1) / shift;
  const int width = dim * m;
  out.resize(static_cast<size_t>(out_frames) * width);
  for (int i = 0; i < out_frames; ++i) {
    for (int j = 0; j < m; ++j) {
      int idx = i * shift - (m - 1) / 2 + j;
      idx = std::min(std::max(idx, 0), num_frames - 1);
      const float* src = feats.data() + static_cast<size_t>(idx) * dim;
      float* dst = out.data() + static_cast<size_t>(i) * width + static_cast<size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) {
        const int c = j * dim + d;
        dst[d] = (src[d] + neg_mean[c]) * inv_std[c];
      }
    }
  }
}

}  // namespace ozasr
