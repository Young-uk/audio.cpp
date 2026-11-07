// Exercise the actual load/publication and idle sweep paths in both runtimes.
// A controlled loader needs no model weights, GPU, socket or production sleeps.
#include "runtime.h"
#include "parallel_runtime.h"

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace std::chrono_literals;
namespace rt = engine::runtime;
namespace srv = minitts::server;

namespace {
void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Control {
    std::function<void()> on_load;
    bool fail_load = false;
    std::atomic<int> loads{0}, destructions{0}, runs{0};
};
class Session final : public rt::IOfflineVoiceTaskSession {
public:
    explicit Session(std::shared_ptr<Control> control) : control_(std::move(control)) {}
    std::string family() const override { return "idle_fixture"; }
    rt::VoiceTaskKind task_kind() const override { return rt::VoiceTaskKind::Tts; }
    rt::RunMode run_mode() const override { return rt::RunMode::Offline; }
    void prepare(const rt::SessionPreparationRequest &) override {}
    rt::TaskResult run(const rt::TaskRequest &) override {
        ++control_->runs;
        rt::TaskResult result;
        result.audio_output = rt::AudioBuffer{24000, 1, {0.1F, -0.1F}};
        return result;
    }
private:
    std::shared_ptr<Control> control_;
};
class Model final : public rt::ILoadedVoiceModel {
public:
    explicit Model(std::shared_ptr<Control> control) : control_(std::move(control)) {
        metadata_.family = "idle_fixture";
    }
    ~Model() override { ++control_->destructions; }
    const rt::ModelMetadata & metadata() const noexcept override { return metadata_; }
    const rt::CapabilitySet & capabilities() const noexcept override { return capabilities_; }
    std::unique_ptr<rt::IVoiceTaskSession> create_task_session(
        const rt::TaskSpec &, const rt::SessionOptions &) const override {
        return std::make_unique<Session>(control_);
    }
private:
    std::shared_ptr<Control> control_;
    rt::ModelMetadata metadata_;
    rt::CapabilitySet capabilities_;
};
class Loader final : public rt::IVoiceModelLoader {
public:
    std::shared_ptr<Control> control = std::make_shared<Control>();
    std::string family() const override { return "idle_fixture"; }
    bool can_load(const rt::ModelLoadRequest &) const override { return true; }
    rt::ModelInspection inspect(const rt::ModelLoadRequest &) const override { return {}; }
    std::unique_ptr<rt::ILoadedVoiceModel> load(const rt::ModelLoadRequest &) const override {
        ++control->loads;
        if (control->on_load) control->on_load();
        if (control->fail_load) throw std::runtime_error("controlled load failure");
        return std::make_unique<Model>(control);
    }
};
} // namespace

namespace minitts::server {
class ServerRuntimeTestAccess {
public:
    template<class State>
    static void configure(State & state, const std::shared_ptr<Loader> & loader,
                          const std::filesystem::path & path) {
        state.registry_factory_ = [loader] {
            rt::ModelRegistry registry;
            registry.register_loader(loader);
            return registry;
        };
        ServerModelConfig config;
        config.id = "slow"; config.family = loader->family(); config.path = path;
        config.lazy = true; config.task = "tts"; config.mode = "offline";
        auto model = state.make_model(std::move(config));
        state.model_index_.emplace("slow", state.models_.size());
        state.models_.push_back(std::move(model));
        // Keep the background thread off for deterministic scheduling. Tests
        // invoke the real sweep exactly where a stale watchdog decision resumes.
        state.config_.idle_unload_ms = 1000;
    }
    template<class State> static void load(State & state) {
        auto & model = *state.models_.front();
        auto lease = state.acquire_model_run(model, std::nullopt);
        state.ensure_model_loaded_locked(model);
    }
    template<class State> static auto clocks(State & state) {
        return std::make_pair(state.last_activity_ms_.load(), state.models_.front()->last_used_ms.load());
    }
    template<class State> static void age_activity(State & state) {
        state.last_activity_ms_.store(now_ms() - state.config_.idle_unload_ms - 1);
    }
    template<class State> static void sweep(State & state) { state.unload_idle_models(); }
    template<class State> static bool loaded(State & state) { return state.models_.front()->loaded.load(); }
};
} // namespace minitts::server
using Access = srv::ServerRuntimeTestAccess;

namespace {
template<class State> struct Fixture {
    std::filesystem::path root;
    std::shared_ptr<Loader> loader = std::make_shared<Loader>();
    std::unique_ptr<State> state;
    Fixture() {
        root = std::filesystem::temp_directory_path() /
            ("audiocpp-idle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        std::ofstream(root / "fixture") << "controlled fixture";
        srv::ServerConfig config;
        config.backend = engine::core::BackendType::Cpu;
        config.ui_enabled = false; config.idle_unload_ms = 0;
        state = std::make_unique<State>(config, root);
        Access::configure(*state, loader, root / "fixture");
    }
    ~Fixture() {
        state.reset();
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

template<class State> void load_completion() {
    Fixture<State> fixture;
    auto & state = *fixture.state;
    auto control = fixture.loader->control;
    std::promise<void> entered, release;
    auto entered_future = entered.get_future();
    auto released = release.get_future().share();
    control->on_load = [&] { entered.set_value(); released.wait(); };
    auto loading = std::async(std::launch::async, [&] { Access::load(state); });
    const bool started = entered_future.wait_for(2s) == std::future_status::ready;
    if (started) std::this_thread::sleep_for(1100ms); // Longer than the 1000ms idle timeout.
    const auto completed_not_before = now_ms();
    release.set_value();
    loading.get();
    require(started, "controlled load never started");
    const auto clocks = Access::clocks(state);
    require(clocks.first >= completed_not_before, "global idle clock predates successful load completion");
    require(clocks.second >= completed_not_before, "model last-used clock predates successful load completion");
    Access::sweep(state); // Resume an idle decision made while the load was busy.
    require(Access::loaded(state) && control->destructions == 0, "freshly completed load was unloaded immediately");
    std::this_thread::sleep_for(1100ms);
    Access::sweep(state);
    require(!Access::loaded(state) && control->destructions == 1, "actual post-load idle expiry did not unload");
    control->on_load = {};
    Access::load(state);
    require(Access::loaded(state) && control->loads == 2, "model did not reload after idle expiry");
}

template<class State> void idle_recheck() {
    Fixture<State> fixture;
    auto & state = *fixture.state;
    Access::load(state);
    Access::age_activity(state); // The watchdog could decide to unload here.
    srv::HttpRequest request;
    request.method = "POST"; request.path = "/v1/audio/speech";
    request.body = R"({"model":"slow","input":"hello","response_format":"wav"})";
    request.headers["content-type"] = "application/json";
    const auto response = state.handle(request); // Complete new activity before its stale sweep.
    require(response.status == 200 && fixture.loader->control->runs == 1, "controlled speech request failed");
    Access::sweep(state);
    require(Access::loaded(state), "stale idle decision unloaded a newly used model");
    Access::age_activity(state);
    Access::sweep(state);
    require(!Access::loaded(state), "idle recheck prevented genuine expiry");
}

template<class State> void load_failure() {
    Fixture<State> fixture;
    auto & state = *fixture.state;
    auto control = fixture.loader->control;
    std::pair<int64_t, int64_t> at_load_start;
    control->on_load = [&] {
        at_load_start = Access::clocks(state);
        std::this_thread::sleep_for(20ms);
    };
    control->fail_load = true;
    bool failed = false;
    try { Access::load(state); }
    catch (const std::runtime_error & error) { failed = std::string(error.what()) == "controlled load failure"; }
    require(failed && !Access::loaded(state), "failed load was published or changed its error");
    require(Access::clocks(state) == at_load_start, "failed load incorrectly recorded successful completion");
}

template<class State> void run_case(const std::string & name) {
    if (name == "load-completion") load_completion<State>();
    else if (name == "idle-recheck") idle_recheck<State>();
    else if (name == "load-failure") load_failure<State>();
    else throw std::runtime_error("unknown idle test case");
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) { std::cerr << "expected one idle test case\n"; return 1; }
    int failures = 0;
    try { run_case<srv::ServerState>(argv[1]); }
    catch (const std::exception & error) {
        std::cerr << "legacy: " << error.what() << '\n'; ++failures;
    }
    try { run_case<srv::ParallelServerState>(argv[1]); }
    catch (const std::exception & error) {
        std::cerr << "parallel: " << error.what() << '\n'; ++failures;
    }
    if (failures == 0) std::cout << "PASS both server runtimes: " << argv[1] << '\n';
    return failures == 0 ? 0 : 1;
}
