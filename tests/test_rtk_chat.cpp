#include "llm/rtk.h"
#include "llm/plugin.h"
#include "llm/tools.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace llm;

static void test_rtk_v0_observer() {
    auto rtk = make_rtk_v0();
    assert(rtk != nullptr);
    assert(std::string(rtk->name()) == "rtk-v0-observer");

    ModelConfig cfg;
    cfg.n_layers = 16;
    cfg.dim = 2048;

    RtkKvView v0;
    v0.n_layers = 16;
    v0.kv_dim = 256;
    v0.seq_len = 10;
    v0.capacity = 128;
    v0.bytes = 10 * 256 * sizeof(float);

    bool ok = rtk->init(cfg, v0);
    assert(ok);

    RtkMetrics m = rtk->metrics();
    assert(m.steps == 0);
    assert(m.max_seq == 10);
    assert(m.peak_kv_bytes == v0.bytes);

    // Simulate prefill with larger context
    RtkKvView v_pref = v0;
    v_pref.seq_len = 32;
    v_pref.bytes = 32 * 256 * sizeof(float);
    rtk->on_prefill(v_pref);

    m = rtk->metrics();
    assert(m.max_seq == 32);
    assert(m.peak_kv_bytes == v_pref.bytes);

    // Simulate decode steps
    for (int step = 1; step <= 5; ++step) {
        RtkKvView v_step = v_pref;
        v_step.seq_len = 32 + step;
        v_step.bytes = v_step.seq_len * 256 * sizeof(float);
        rtk->on_step(v_step);
    }

    m = rtk->metrics();
    assert(m.steps == 5);
    assert(m.max_seq == 37);
    assert(m.peak_kv_bytes == 37 * 256 * sizeof(float));

    printf("  PASS  test_rtk_v0_observer\n");
}

static void test_plugin_host_rtk() {
    PluginHost host;
    host.set_rtk(make_rtk_v0());

    ModelConfig cfg;
    RtkKvView v0{4, 64, 5, 32, 512, 1280};

    bool ok = host.init(cfg, v0);
    assert(ok);
    assert(host.inited());
    assert(host.rtk() != nullptr);
    assert(host.kosh() == nullptr);

    host.rtk()->on_step(RtkKvView{4, 64, 6, 32, 512, 1536});
    assert(host.rtk()->metrics().steps == 1);
    assert(host.rtk()->metrics().max_seq == 6);

    host.shutdown();
    printf("  PASS  test_plugin_host_rtk\n");
}

static void test_rtk_chat_templates() {
    std::vector<ChatMessage> msgs = {
        {ChatMessage::Role::System, "You are a helpful assistant.", ""},
        {ChatMessage::Role::User, "Hello world!", ""},
        {ChatMessage::Role::Assistant, "Hi there!", ""},
        {ChatMessage::Role::User, "What is 2+2?", ""}
    };
    ToolRegistry tools;

    // ChatML style
    std::string chatml = render_chat(msgs, tools, ChatTemplateStyle::ChatML, true);
    assert(chatml.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>") != std::string::npos);
    assert(chatml.find("<|im_start|>user\nHello world!<|im_end|>") != std::string::npos);
    assert(chatml.find("<|im_start|>assistant\nHi there!<|im_end|>") != std::string::npos);
    assert(chatml.find("<|im_start|>assistant\n") != std::string::npos);

    // LLaMA-3 style
    std::string llama3 = render_chat(msgs, tools, ChatTemplateStyle::Llama3, true);
    assert(llama3.find("<|start_header_id|>system<|end_header_id|>") != std::string::npos);
    assert(llama3.find("<|start_header_id|>user<|end_header_id|>") != std::string::npos);
    assert(llama3.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos);

    // Mistral style
    std::string mistral = render_chat(msgs, tools, ChatTemplateStyle::Mistral, true);
    assert(mistral.find("[INST]") != std::string::npos);
    assert(mistral.find("[/INST]") != std::string::npos);

    // Gemma style
    std::string gemma = render_chat(msgs, tools, ChatTemplateStyle::Gemma, true);
    assert(gemma.find("<start_of_turn>user") != std::string::npos);
    assert(gemma.find("<end_of_turn>") != std::string::npos);
    assert(gemma.find("<start_of_turn>model") != std::string::npos);

    printf("  PASS  test_rtk_chat_templates\n");
}

int main() {
    printf("== test_rtk_chat ==\n");
    test_rtk_v0_observer();
    test_plugin_host_rtk();
    test_rtk_chat_templates();
    printf("All RTK chat tests passed!\n");
    return 0;
}
