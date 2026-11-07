#include "decoder_graph_helpers.h"
#include "test_assert.h"

#include <ggml-alloc.h>
#include <ggml-cpu.h>

#include <iostream>
#include <limits>
#include <string>

using engine::test::require_eq;
namespace decoder = engine::models::qwen3_tts::detail;

namespace {

template<class Callback>
void require_unavailable(Callback callback, const std::string & label) {
    bool failed = false;
    try {
        callback();
    } catch (const std::runtime_error & error) {
        failed = std::string(error.what()) == "Qwen3 speech decoder output tensor is unavailable";
    }
    require_eq(failed, true, label.c_str());
}

struct TestGraph {
    const ggml_tensor * output;
    int * destroyed;
    bool fail_compute = false;
    TestGraph(const ggml_tensor * tensor, int & count) : output(tensor), destroyed(&count) {}
    ~TestGraph() { ++*destroyed; }
    std::vector<float> run(const int32_t *, size_t) {
        if (fail_compute) throw std::runtime_error("synthetic compute failure");
        return decoder::read_decoder_output(output, 4);
    }
};

} // namespace

int main() {
    try {
        auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
            ggml_backend_cpu_init(), ggml_backend_free);
        require_eq(backend != nullptr, true, "CPU backend created");
        auto context = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
            ggml_init({64 * 1024, nullptr, true}), ggml_free);
        require_eq(context != nullptr, true, "tensor metadata context created");
        auto * output = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        auto * matrix = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 2, 2);
        auto * noncontiguous = ggml_transpose(context.get(), matrix);
        auto * view = ggml_view_1d(context.get(), output, 2, sizeof(float));
        auto * half_output = ggml_new_tensor_1d(context.get(), GGML_TYPE_F16, 4);
        require_unavailable([] { decoder::read_decoder_output(nullptr, 4); }, "null output throws instead of aborting");
        require_unavailable([&] { decoder::read_decoder_output(output, 4); }, "unallocated output throws");
        auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
            ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
        require_eq(buffer != nullptr, true, "CPU tensor buffer allocated");
        const std::vector<float> expected{0.25F, -0.5F, 0.75F, -1.0F};
        ggml_backend_tensor_set(output, expected.data(), 0, expected.size() * sizeof(float));
        require_eq(decoder::read_decoder_output(output, 4) == expected, true, "valid output unchanged");
        require_eq(decoder::read_decoder_output(view, 2) == std::vector<float>{-0.5F, 0.75F},
                   true, "valid view uses its source buffer");
        require_eq(decoder::read_decoder_output(output, 0).empty(), true, "zero sample read is safe");
        require_unavailable([&] { decoder::read_decoder_output(output, 5); }, "out-of-bounds read throws");
        require_unavailable([&] { decoder::read_decoder_output(output, std::numeric_limits<size_t>::max()); },
                            "sample count overflow cannot bypass guard");
        require_unavailable([&] { decoder::read_decoder_output(half_output, 2); }, "non-F32 waveform throws");
        require_unavailable([&] { decoder::read_decoder_output(noncontiguous, 4); }, "strided waveform throws");
        auto * original_data = output->data;
        output->data = nullptr;
        require_unavailable([&] { decoder::read_decoder_output(output, 4); }, "null data throws");
        output->data = original_data;
        auto * original_buffer = output->buffer;
        output->buffer = nullptr;
        require_unavailable([&] { decoder::read_decoder_output(output, 4); }, "null buffer throws");
        require_unavailable([&] { decoder::read_decoder_output(view, 2); }, "view with missing source buffer throws");
        output->buffer = original_buffer;

        // Exercise the exact helper used by all three production cache paths.
        // The failure is synthetic; no model or backend fault injection is needed.
        {
            int destroyed = 0;
            auto graph = std::make_unique<TestGraph>(nullptr, destroyed);
            require_unavailable([&] { decoder::run_cached_decoder_graph(graph, nullptr, 0); }, "cached null readback");
            require_eq(graph == nullptr, true, "readback failure invalidates graph");
            require_eq(destroyed, 1, "failed graph destroyed once");
            graph = std::make_unique<TestGraph>(output, destroyed);
            require_eq(decoder::run_cached_decoder_graph(graph, nullptr, 0) == expected, true,
                       "next graph succeeds without stale failure");
            auto * reused = graph.get();
            require_eq(decoder::run_cached_decoder_graph(graph, nullptr, 0) == expected, true,
                       "successful graph remains reusable");
            require_eq(graph.get() == reused, true, "successful cache is retained");
            graph->fail_compute = true;
            bool failed = false;
            try { decoder::run_cached_decoder_graph(graph, nullptr, 0); }
            catch (const std::runtime_error & error) {
                failed = std::string(error.what()) == "synthetic compute failure";
            }
            require_eq(failed, true, "compute error propagated unchanged");
            require_eq(graph == nullptr, true, "compute failure invalidates graph");
        }
        std::cout << "qwen3_tts_decoder_recovery_test: readback guards and shared cached-graph recovery policy passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
