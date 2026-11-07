#pragma once

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace engine::models::qwen3_tts::detail {

inline std::vector<float> read_decoder_output(const ggml_tensor * output, size_t sample_count) {
    // ggml_backend_tensor_get aborts the process on invalid tensor state.
    // Report a failed synthesis so other requests can still use the server.
    if (output == nullptr || output->data == nullptr ||
        (output->view_src ? output->view_src->buffer : output->buffer) == nullptr ||
        output->type != GGML_TYPE_F32 || !ggml_is_contiguous(output) ||
        sample_count > ggml_nbytes(output) / sizeof(float)) {
        throw std::runtime_error("Qwen3 speech decoder output tensor is unavailable");
    }
    std::vector<float> audio(sample_count, 0.0F);
    ggml_backend_tensor_get(output, audio.data(), 0, audio.size() * sizeof(float));
    return audio;
}

// Offline, windowed streaming and stateful streaming all retain graphs. Apply
// the same failure policy to each slot so a failed graph is rebuilt next time.
// Stateful request history is reset by the session before the next utterance.
template<class Graph>
std::vector<float> run_cached_decoder_graph(
    std::unique_ptr<Graph> & graph, const int32_t * codes, size_t code_count) {
    try {
        return graph->run(codes, code_count);
    } catch (...) {
        // Never reuse a decoder graph whose execution or readback failed.
        graph.reset();
        throw;
    }
}

} // namespace engine::models::qwen3_tts::detail
