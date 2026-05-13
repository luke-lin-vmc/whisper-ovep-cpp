// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "whisper.hpp"
#define NOMINMAX
#include <Windows.h>
#include <codecvt>
#include <chrono>
#include "base64.h" // Include a base64 library
#include <iostream>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

// WinML 2.0 API
#include <WinMLEpCatalog.h>
#include <onnxruntime_cxx_api.h>

std::map<int, std::string> token_table;

torch::Tensor create_attention_mask(int current_index, size_t max_ctx) {
    torch::Tensor attention_mask = torch::full(
        { 1, 1, 1, (long long)max_ctx },
        -std::numeric_limits<float>::infinity(),
        torch::kFloat32
    );

    int end_index = current_index + 1;
    if (end_index > max_ctx) {
        end_index = max_ctx;
    }
    attention_mask.index_put_({ 0, 0, 0, torch::indexing::Slice(0, end_index) }, 0.0f);

    return attention_mask;
}

void whisper_run(OnnxModel& model, const torch::Tensor& mel) {
    auto [n_layer_cross_k, n_layer_cross_v] = model.runEncoder(mel);

    int lang = model.detectLanguage(n_layer_cross_k, n_layer_cross_v);
    model.sot_sequence[1] = lang;
    model.sot_sequence[2] = model.transcribe;

    auto [n_layer_self_k_cache, n_layer_self_v_cache] = model.getSelfCache();

    torch::Tensor current_index = torch::zeros({ 1 }, torch::dtype(torch::kInt64));
    size_t max_ctx = model.n_text_ctx;

    std::vector<int> results;
    results.reserve(max_ctx);
    int max_token_id = -1;

    for (auto element : model.sot_sequence) {
        std::vector<int64_t> first_sot_sequence = { element };
        torch::Tensor first_tokens = torch::tensor(first_sot_sequence, torch::dtype(torch::kInt64)).unsqueeze(0);

        torch::Tensor attention_mask = create_attention_mask(current_index.item<int>(), max_ctx);

        auto [logits, self_k_cache, self_v_cache] = model.runDecoder(
            first_tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            current_index,
            attention_mask
        );

        n_layer_self_k_cache = self_k_cache;
        n_layer_self_v_cache = self_v_cache;

        current_index += 1;

        auto logits_last = logits[0][-1];
        model.suppressTokens(logits_last, true);
        max_token_id = logits_last.argmax().item<int>();
    }

    for (int i = 0; i < max_ctx; ++i) {
        if (max_token_id == model.eot)
            break;
        results.push_back(max_token_id);

        torch::Tensor tokens = torch::tensor({ {results.back()} }, torch::kInt64);

        torch::Tensor attention_mask = create_attention_mask(current_index.item<int>(), max_ctx);


        auto [new_logits, new_self_k_cache, new_self_v_cache] = model.runDecoder(
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            current_index,
            attention_mask
        );

        n_layer_self_k_cache = new_self_k_cache;
        n_layer_self_v_cache = new_self_v_cache;

        current_index += 1;
        new_logits = new_logits[0][-1];
        model.suppressTokens(new_logits, false);

        max_token_id = new_logits.argmax().item<int>();
    }

    std::string s;
    for (int i : results) {
        if (token_table.find(i) != token_table.end()) {
            s += base64_decode(token_table[i]);
        }
    }

    std::cout << "\nTranscribed:\n" << s << std::endl;
    std::cout << "\nProcessing complete. Exiting." << std::endl;
}

// Windows ML Execution Provider Catalog Sample
// https://github.com/microsoft/WindowsAppSDK-Samples/blob/main/Samples/WindowsML/cmake/WinMLEpCatalog/main.cpp
struct RegistrationContext
{
    Ort::Env* env;
    std::vector<std::string> registeredProviders;
};

static BOOL CALLBACK RegisterCertifiedProvider(WinMLEpHandle ep, const WinMLEpInfo* info, void* context)
{
    if (!info || !info->name)
    {
        return TRUE;
    }

    auto* ctx = static_cast<RegistrationContext*>(context);

    std::cout << std::format("Provider: {}\n", info->name);

    if (info->certification != WinMLEpCertification_Certified)
    {
        std::cout << "  Skipping (not certified)\n\n";
        return TRUE;
    }

    HRESULT hr = WinMLEpEnsureReady(ep);
    if (FAILED(hr))
    {
        std::cout << std::format("  EnsureReady failed: 0x{:08X}\n\n", static_cast<uint32_t>(hr));
        return TRUE;
    }

    size_t pathSize = 0;
    hr = WinMLEpGetLibraryPathSize(ep, &pathSize);
    if (FAILED(hr) || pathSize == 0)
    {
        std::cout << "  Failed to get library path\n\n";
        return TRUE;
    }

    std::string libraryPathUtf8(pathSize, '\0');
    size_t used = 0;
    hr = WinMLEpGetLibraryPath(ep, pathSize, libraryPathUtf8.data(), &used);
    if (FAILED(hr) || used == 0)
    {
        std::cout << "  Failed to get library path\n\n";
        return TRUE;
    }
    libraryPathUtf8.resize(used - 1);

    std::cout << std::format("  Library: {}\n", libraryPathUtf8);

    try
    {
        std::filesystem::path libraryPath(libraryPathUtf8);
        ctx->env->RegisterExecutionProviderLibrary(info->name, libraryPath.wstring());
        ctx->registeredProviders.push_back(info->name);
        std::cout << "  Registered!\n\n";
    }
    catch (const Ort::Exception& e)
    {
        std::cout << std::format("  Registration failed: {}\n\n", e.what());
    }

    return TRUE;
}

int main(int argc, char* argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    if (argc < 4)
    {
        std::cout << "usage: whisper_onnx.exe <audio_file> <model_type> <device>" << std::endl;
        return -1;
    }

    std::string audio_file = std::string(argv[1]);
    
    std::string model_type = std::string(argv[2]);
    std::string whisper_encoder_onnx = model_type + "-encoder.onnx";
    std::wstring whisper_encoder_path = converter.from_bytes(whisper_encoder_onnx);
    std::string whisper_decoder_onnx = model_type + "-decoder.onnx";
    std::wstring whisper_decoder_path = converter.from_bytes(whisper_decoder_onnx);
    std::string whisper_tokens = model_type + "-tokens.txt";
    token_table = loadTokens(whisper_tokens);
   
    std::string device = std::string(argv[3]);
    std::transform(device.begin(), device.end(), device.begin(), [](unsigned char c){ return std::toupper(c); }); // make device uppercase

    std::cout << "Whisper encoder model: " << whisper_encoder_onnx << std::endl;
    std::cout << "Whisper decoder model: " << whisper_decoder_onnx << std::endl;
    std::cout << "Whisper tokens: " << whisper_tokens << std::endl;
    std::cout << "Inference device: " << device << std::endl;

    try {
        // Enable more detailed logging for ONNX Runtime
        //Ort::Env env(ORT_LOGGING_LEVEL_INFO, "Default");
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "Default");

        // Windows ML Execution Provider Catalog Sample
        // https://github.com/microsoft/WindowsAppSDK-Samples/blob/main/Samples/WindowsML/cmake/WinMLEpCatalog/main.cpp
        WinMLEpCatalogHandle catalog = nullptr;
        HRESULT hr = WinMLEpCatalogCreate(&catalog);
        if (FAILED(hr))
        {
            std::cerr << std::format("Failed to create catalog: 0x{:08X}\n", static_cast<uint32_t>(hr));
            return 1;
        }

        RegistrationContext ctx{ &env, {} };
        hr = WinMLEpCatalogEnumProviders(catalog, RegisterCertifiedProvider, &ctx);
        if (FAILED(hr))
        {
            std::cerr << std::format("Enumeration failed: 0x{:08X}\n", static_cast<uint32_t>(hr));
        }

        // The catalog is no longer needed after enumeration.
        WinMLEpCatalogRelease(catalog);

        auto epDevices = env.GetEpDevices();
        std::cout << std::format("Available EP devices ({}):\n", epDevices.size());
        for (const auto& device : epDevices)
        {
            const char* epName = device.EpName();
            if (epName)
            {
                std::cout << std::format("  - {}\n", epName);
            }
        }

        std::cout << std::format("\nRegistered {} certified provider(s).\n", ctx.registeredProviders.size());


        const ORTCHAR_T* encoder_path = whisper_encoder_path.c_str();
        const ORTCHAR_T* decoder_path = whisper_decoder_path.c_str();

        OnnxModel model(env, encoder_path, decoder_path, device);

        // Handle audio file
        auto [audioData, sampleRate] = load_audio(audio_file);
        if (!audioData.empty()) {
            torch::Tensor mel = compute_features(audioData, sampleRate, (int)model.n_mels);
            whisper_run(model, mel); // Pass mel directly
        }

        // Registered EP libraries hold references to resources owned by the
        // Ort::Env.  They must be explicitly unregistered before the Env
        // destructor runs, otherwise the provider DLLs may be unloaded while
        // still in use, leading to access violations during shutdown.
        for (const auto& name : ctx.registeredProviders)
        {
            env.UnregisterExecutionProviderLibrary(name.c_str());
            std::cout << "Successfully unregistered: " << name << std::endl;
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception occurred: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception occurred: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }
    return 0;
}
