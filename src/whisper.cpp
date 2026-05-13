// Reference: https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-whisper-model.cc

#define NOMINMAX // To avoid macro clash with 'min' and 'max'

#include <fstream> // Include this for file stream functionality
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

// Other includes...
#include <onnxruntime_cxx_api.h>
#include "whisper.hpp"
#include <sndfile.hh>
#include <torch/torch.h>
#include <filesystem>
#include <limits>
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/whisper-feature.h"
#include "base64.h"

// Definitions of functions and methods

std::unique_ptr<Ort::Session> encoder;
std::unique_ptr<Ort::Session> decoder;
Ort::SessionOptions session_opts;

// Function to load audio file
std::pair<std::vector<float>, int> load_audio(const std::string& filename) {
    SndfileHandle file(filename);

    if (file.error()) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        return {};
    }

    int sample_rate = file.samplerate();
    int channels = file.channels();

    std::vector<float> data(file.frames() * channels);
    file.readf(data.data(), file.frames());

    // Return mono channel samples for simplicity
    std::vector<float> samples(file.frames());
    for (size_t i = 0; i < file.frames(); ++i) {
        samples[i] = data[i * channels];
    }

    return { samples, sample_rate };
}

// Function to compute features
torch::Tensor compute_features(const std::vector<float>& samples, int sample_rate, int dim) {
    std::vector<torch::Tensor> features;
    knf::WhisperFeatureOptions opts;
    opts.dim = dim;
    knf::OnlineWhisperFbank online_whisper_fbank(opts);

    // The actual sample rate passed to AcceptWaveform must match the dataset
    online_whisper_fbank.AcceptWaveform(sample_rate, samples.data(), samples.size());
    online_whisper_fbank.InputFinished();

    for (int i = 0; i < online_whisper_fbank.NumFramesReady(); ++i) {
        const float* f = online_whisper_fbank.GetFrame(i);
        auto tensor = torch::from_blob(const_cast<float*>(f), { dim }, torch::kFloat32).clone();
        features.push_back(tensor);
    }

    torch::Tensor features_tensor = torch::stack(features);

    // Use log10() for logarithmic scaling
    torch::Tensor log_spec = torch::clamp(features_tensor, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    torch::Tensor mel = (log_spec + 4.0) / 4.0;

    // Pad feature array to a specific length
    mel = torch::constant_pad_nd(mel, { 0, 0, 0, 1500 }, 0);

    int target = 3000;
    if (mel.size(0) > target) {
        mel = mel.narrow(0, 0, target - 50);
        mel = torch::constant_pad_nd(mel, { 0, 0, 0, 50 }, 0);
    }
    else if (mel.size(0) < target) {
        int padding = target - mel.size(0);
        mel = torch::constant_pad_nd(mel, { 0, 0, 0, padding }, 0);
    }

    // Transpose and unsqueeze
    mel = mel.t().unsqueeze(0);

    return mel;
}

torch::Tensor compute_features_audio(const std::vector<float>& wave, int sample_rate, int dim) {
    std::vector<torch::Tensor> features;
    knf::WhisperFeatureOptions opts;
    opts.dim = dim;
    knf::OnlineWhisperFbank online_whisper_fbank(opts);

    online_whisper_fbank.AcceptWaveform(sample_rate, wave.data(), wave.size());
    online_whisper_fbank.InputFinished();

    for (int i = 0; i < online_whisper_fbank.NumFramesReady(); ++i) {
        const float* f = online_whisper_fbank.GetFrame(i);
        auto tensor = torch::from_blob(const_cast<float*>(f), { dim }, torch::kFloat32).clone();
        features.push_back(tensor);
    }

    torch::Tensor features_tensor = torch::stack(features);

    torch::Tensor log_spec = torch::clamp(features_tensor, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    torch::Tensor mel = (log_spec + 4.0) / 4.0;

    mel = torch::constant_pad_nd(mel, { 0, 0, 0, 1500 }, 0);

    int target = 3000;
    if (mel.size(0) > target) {
        mel = mel.narrow(0, 0, target - 50);
        mel = torch::constant_pad_nd(mel, { 0, 0, 0, 50 }, 0);
    }
    else if (mel.size(0) < target) {
        int padding = target - mel.size(0);
        mel = torch::constant_pad_nd(mel, { 0, 0, 0, padding }, 0);
    }

    mel = mel.t().unsqueeze(0);

    return mel;
}

// Function to load tokens
std::map<int, std::string> loadTokens(const std::string& filename) {
    std::map<int, std::string> tokens;
    std::ifstream file(filename);
    std::string line;
    if (file.is_open()) {
        while (getline(file, line)) {
            std::istringstream iss(line);
            int i;
            std::string t;
            iss >> t >> i;
            tokens[i] = t;
        }
        file.close();
    }
    return tokens;
}

Ort::ConstEpDevice get_cpu_ep_device(Ort::Env& env) {
    auto ep_devices = env.GetEpDevices();
    for (Ort::ConstEpDevice& device : ep_devices) {
        if (std::string_view(device.EpName()).find("CPUExecutionProvider") != std::string::npos) {
            return device;
        }
    }
    return Ort::ConstEpDevice{};
}

Ort::ConstEpDevice get_ov_ep_device(Ort::Env& env, const std::string& ov_device) {
    auto ep_devices = env.GetEpDevices();
    // AUTO
    if (ov_device == "AUTO") {
        for (Ort::ConstEpDevice& device : ep_devices) {
            if (std::string_view(device.EpName()).find("OpenVINOExecutionProvider.AUTO") != std::string::npos) {
                const auto& meta_kv = device.EpMetadata().GetKeyValuePairs();
                auto device_type_it = meta_kv.find("ov_device");
                if (device_type_it != meta_kv.end()) {
                    std::cout << "AUTO Selected inference device: " << device_type_it->second << std::endl;
                }
                return device;
            }
        }
    }
    else // NPU, GPU, CPU
    {
        for (Ort::ConstEpDevice& device : ep_devices) {
            if (std::string_view(device.EpName()).find("OpenVINOExecutionProvider") != std::string::npos) {
                const auto& meta_kv = device.EpMetadata().GetKeyValuePairs();
                auto device_type_it = meta_kv.find("ov_device");
                if (device_type_it != meta_kv.end()) {
                    if (device_type_it->second == ov_device) {
                        return device;
                    }
                }
            }
        }
    }
    return Ort::ConstEpDevice{};
}

// OnnxModel Implementation
OnnxModel::OnnxModel(Ort::Env& env, const ORTCHAR_T* encoder_path, const ORTCHAR_T* decoder_path, std::string inference_device) {
    std::vector<std::string> allowed_ov_device_types =
    {
        "NPU", "GPU", "CPU", "AUTO"
    };

    bool is_ov_device = false;
    for (auto d : allowed_ov_device_types)
    {
        if (inference_device == d)
        {
            is_ov_device = true;
            break;
        }
    }

    if (is_ov_device)
    {
        std::cout << "Inference device: OpenVINO EP with device = " << inference_device << std::endl;

        Ort::ConstEpDevice ep_device{};
        std::vector<Ort::ConstEpDevice> ep_devices;
        ep_device = get_ov_ep_device(env, inference_device);
        if (!ep_device)
            throw std::runtime_error("Cannot find OpenVINOExecutionProvider with device " + inference_device);
        else
			ep_devices.push_back(ep_device);

        // For NPU to enable model caching
        std::unordered_map<std::string, std::string> provider_options;
        if (inference_device == "AUTO" || inference_device == "NPU")
        {
            // Reference: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#load_config
            provider_options["load_config"] = "{\"AUTO\":{\"DEVICE_PROPERTIES\":{\"NPU\":{\"CACHE_DIR\":\"cache\"}}},\"NPU\":{\"CACHE_DIR\":\"cache\"}}";
        }

        session_opts.AppendExecutionProvider_V2(env, ep_devices, provider_options);
    }
    else
    {
        std::cout << "Inference device: Using default CPUExecutionProvider" << std::endl;

        Ort::ConstEpDevice ep_device{};
        std::vector<Ort::ConstEpDevice> ep_devices;
        ep_device = get_cpu_ep_device(env);
        if (!ep_device)
            throw std::runtime_error("Cannot find CPUExecutionProvider");
        else
            ep_devices.push_back(ep_device);

        std::unordered_map<std::string, std::string> provider_options;

        session_opts.AppendExecutionProvider_V2(env, ep_devices, provider_options);
    }

    /* Set more session options if required */
    //session_opts.SetIntraOpNumThreads(4);
    //session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    initEncoder(env, encoder_path);
    initDecoder(env, decoder_path);
}

// Initialize Encoder
void OnnxModel::initEncoder(Ort::Env& env, const ORTCHAR_T* encoder_path) {

    // Create the ONNX Runtime session
    encoder = std::make_unique<Ort::Session>(env, encoder_path, session_opts);

    // Additional debug logging
    std::wcout << L"Encoder path: " << encoder_path << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;

    // Debug and validate input and output node names
    size_t num_input_nodes = encoder->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto input_name = encoder->GetInputNameAllocated(i, allocator);
        std::cout << "Encoder input name " << i << ": " << input_name.get() << std::endl;
    }

    size_t num_output_nodes = encoder->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto output_name = encoder->GetOutputNameAllocated(i, allocator);
        std::cout << "Encoder output name " << i << ": " << output_name.get() << std::endl;
    }

    Ort::ModelMetadata meta = encoder->GetModelMetadata();

    auto custom_metadata_keys = meta.GetCustomMetadataMapKeysAllocated(allocator);
    std::unordered_map<std::string, std::string> meta_data;

    for (const auto& key : custom_metadata_keys) {
        std::string key_str = key.get();
        auto value = meta.LookupCustomMetadataMapAllocated(key_str.c_str(), allocator);
        meta_data[key_str] = value.get();
    }

    n_text_layer = std::stoi(meta_data.at("n_text_layer"));
    n_text_ctx = std::stoi(meta_data.at("n_text_ctx"));
    n_text_state = std::stoi(meta_data.at("n_text_state"));
    n_mels = std::stoi(meta_data.at("n_mels"));
    sot = std::stoi(meta_data.at("sot"));
    eot = std::stoi(meta_data.at("eot"));
    translate = std::stoi(meta_data.at("translate"));
    transcribe = std::stoi(meta_data.at("transcribe"));
    no_timestamps = std::stoi(meta_data.at("no_timestamps"));
    no_speech = std::stoi(meta_data.at("no_speech"));
    blank = std::stoi(meta_data.at("blank_id"));

    parseElements(meta_data.at("sot_sequence"), sot_sequence);
    sot_sequence.push_back(no_timestamps);

    parseElements(meta_data.at("all_language_tokens"), all_language_tokens);
    split(meta_data.at("all_language_codes"), ',', all_language_codes);

    for (size_t i = 0; i < all_language_tokens.size(); ++i) {
        lang2id[all_language_codes[i]] = all_language_tokens[i];
        id2lang[all_language_tokens[i]] = all_language_codes[i];
    }

    is_multilingual = std::stoi(meta_data.at("is_multilingual")) == 1;
}

// Initialize Decoder
void OnnxModel::initDecoder(Ort::Env& env, const ORTCHAR_T* decoder_path) {
    decoder = std::make_unique<Ort::Session>(env, decoder_path, session_opts);

    // Check decoder path
    std::wcout << L"Decoder path: " << decoder_path << std::endl;

    // Retrieve input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = decoder->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto input_name = decoder->GetInputNameAllocated(i, allocator);
        //std::cout << "Decoder input name " << i << ": " << input_name.get() << std::endl;
    }

    size_t num_output_nodes = decoder->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto output_name = decoder->GetOutputNameAllocated(i, allocator);
        //std::cout << "Decoder output name " << i << ": " << output_name.get() << std::endl;
    }
}

// Encoder and Decoder Run Methods
std::pair<torch::Tensor, torch::Tensor> OnnxModel::runEncoder(torch::Tensor mel) {
    using namespace std::chrono;
    auto start = steady_clock::now();

    mel = mel.contiguous();
    float* input_value = mel.data_ptr<float>();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = { mel.size(0), mel.size(1), mel.size(2) };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_value,
        mel.numel(),
        input_shape.data(),
        input_shape.size()
    );

    std::vector<const char*> input_names = { "mel" };
    std::vector<const char*> output_names = { "n_layer_cross_k", "n_layer_cross_v" };

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = encoder->Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Error during ONNX Runtime execution: " << e.what() << std::endl;
        return {};
    }

    // If output_tensors is empty, return an error
    if (output_tensors.empty()) {
        std::cerr << "Error: No outputs from encoder." << std::endl;
        return {};
    }

    auto n_layer_cross_k = std::move(output_tensors[0]);
    auto n_layer_cross_v = std::move(output_tensors[1]);

    auto n_layer_cross_k_data = n_layer_cross_k.GetTensorMutableData<float>();
    auto n_layer_cross_v_data = n_layer_cross_v.GetTensorMutableData<float>();

    // Measuring time
    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    std::cout << "Encoder processing time: " << diff.count() << " ms" << std::endl;

    // Convert ONNX output to torch::Tensor
    torch::Tensor torch_n_layer_cross_k = torch::from_blob(n_layer_cross_k_data, n_layer_cross_k.GetTensorTypeAndShapeInfo().GetShape());
    torch::Tensor torch_n_layer_cross_v = torch::from_blob(n_layer_cross_v_data, n_layer_cross_v.GetTensorTypeAndShapeInfo().GetShape());

    torch::Tensor cross_k_permuted = torch_n_layer_cross_k.permute({ 0, 1, 2, 3 });
    torch::Tensor cross_v_permuted = torch_n_layer_cross_v.permute({ 0, 1, 2, 3 });

    return { cross_k_permuted.clone(), cross_v_permuted.clone() };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnnxModel::runDecoder(
    const torch::Tensor& tokens,
    torch::Tensor& n_layer_self_k_cache,
    torch::Tensor& n_layer_self_v_cache,
    const torch::Tensor& n_layer_cross_k,
    const torch::Tensor& n_layer_cross_v,
    const torch::Tensor& current_index,
    const torch::Tensor& attention_mask) {

    using namespace std::chrono;
    auto start = steady_clock::now();

    std::vector<const char*> input_names = {
        "tokens",
        "in_n_layer_self_k_cache",
        "in_n_layer_self_v_cache",
        "n_layer_cross_k",
        "n_layer_cross_v",
        "current_index",
        "attention_mask"
    };
    std::vector<const char*> output_names = { "logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache" };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;

    // tokens (int64)
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, tokens.data_ptr<int64_t>(), tokens.numel(), tokens.sizes().data(), tokens.sizes().size()));
    // in_n_layer_self_k_cache (float)
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, n_layer_self_k_cache.data_ptr<float>(), n_layer_self_k_cache.numel(), n_layer_self_k_cache.sizes().data(), n_layer_self_k_cache.sizes().size()));
    // in_n_layer_self_v_cache (float)
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, n_layer_self_v_cache.data_ptr<float>(), n_layer_self_v_cache.numel(), n_layer_self_v_cache.sizes().data(), n_layer_self_v_cache.sizes().size()));
    // n_layer_cross_k (float)
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, n_layer_cross_k.data_ptr<float>(), n_layer_cross_k.numel(), n_layer_cross_k.sizes().data(), n_layer_cross_k.sizes().size()));
    // n_layer_cross_v (float)
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, n_layer_cross_v.data_ptr<float>(), n_layer_cross_v.numel(), n_layer_cross_v.sizes().data(), n_layer_cross_v.sizes().size()));
    // 【新增】current_index (int64)
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, current_index.data_ptr<int64_t>(), current_index.numel(), current_index.sizes().data(), current_index.sizes().size()));
    // 【新增】attention_mask (float)
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, attention_mask.data_ptr<float>(), attention_mask.numel(), attention_mask.sizes().data(), attention_mask.sizes().size()));


    std::vector<Ort::Value> output_tensors;

    try {
        // Correct usage of Run method
        output_tensors = decoder->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensors.data(),  // Use .data() to get a pointer to the array
            input_names.size(),  // Size of the input
            output_names.data(),
            output_names.size()     // Size of the output
        );
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    auto end = steady_clock::now();
    auto diff = duration_cast<milliseconds>(end - start);
    std::cout << "Decoder processing time: " << diff.count() << " ms" << std::endl;

    auto logits = torch::from_blob(output_tensors[0].GetTensorMutableData<float>(), output_tensors[0].GetTensorTypeAndShapeInfo().GetShape());
    auto new_self_k_cache = torch::from_blob(output_tensors[1].GetTensorMutableData<float>(), output_tensors[1].GetTensorTypeAndShapeInfo().GetShape());
    auto new_self_v_cache = torch::from_blob(output_tensors[2].GetTensorMutableData<float>(), output_tensors[2].GetTensorTypeAndShapeInfo().GetShape());

    return std::make_tuple(logits.clone(), new_self_k_cache.clone(), new_self_v_cache.clone());
}

// Get Self Cache
std::pair<torch::Tensor, torch::Tensor> OnnxModel::getSelfCache() {
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    /*
    auto n_layer_self_k_cache = torch::zeros({ 1, 4, 448, 1280 }, options); //Turbo
    auto n_layer_self_v_cache = torch::zeros({ 1, 4, 448, 1280 }, options); //Turbo
    auto n_layer_self_k_cache = torch::zeros({ 1, 12, 448, 768 }, options); //Small
    auto n_layer_self_v_cache = torch::zeros({ 1, 12, 448, 768 }, options); //Small
    auto n_layer_self_k_cache = torch::zeros({ 1, 6, 448, 512 }, options); //base
    auto n_layer_self_v_cache = torch::zeros({ 1, 6, 448, 512 }, options); //base
    */
    auto n_layer_self_k_cache = torch::zeros({ 1, (int)n_text_layer, (int)n_text_ctx, (int)n_text_state }, options);
    auto n_layer_self_v_cache = torch::zeros({ 1, (int)n_text_layer, (int)n_text_ctx, (int)n_text_state }, options);

    return { n_layer_self_k_cache, n_layer_self_v_cache };
}

// Suppress Tokens
void OnnxModel::suppressTokens(torch::Tensor& logits, bool is_initial) {
    if (is_initial) {
        logits.index_put_({ eot }, -std::numeric_limits<float>::infinity());
        logits.index_put_({ blank }, -std::numeric_limits<float>::infinity());
    }
    logits.index_put_({ no_timestamps }, -std::numeric_limits<float>::infinity());
    logits.index_put_({ sot }, -std::numeric_limits<float>::infinity());
    logits.index_put_({ no_speech }, -std::numeric_limits<float>::infinity());
    logits.index_put_({ translate }, -std::numeric_limits<float>::infinity());
}

// Language Detection
int OnnxModel::detectLanguage(const torch::Tensor& n_layer_cross_k, const torch::Tensor& n_layer_cross_v) {
    //token: <|startoftranscript|> (sot)
    auto tokens = torch::full({ 1, 1 }, sot, torch::kInt64);

    torch::Tensor current_index = torch::zeros({ 1 }, torch::kInt64);

    int max_text_ctx = 448;
    if (n_text_ctx > 0) { max_text_ctx = n_text_ctx; }

    torch::Tensor attention_mask = torch::full(
        { 1, 1, 1, max_text_ctx },
        -std::numeric_limits<float>::infinity(),
        torch::kFloat32
    );
    attention_mask.index_put_({ 0, 0, 0, 0 }, 0.0f);

    auto [n_layer_self_k_cache, n_layer_self_v_cache] = getSelfCache();

    auto [logits, _, __] = runDecoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        current_index,
        attention_mask
    );

    logits = logits.view(-1);
    auto mask = torch::ones(logits.sizes(), torch::kInt64);
    for (auto lang_token : all_language_tokens) {
        mask.index_put_({ lang_token }, 0);
    }
    logits.masked_fill_(mask != 0, -std::numeric_limits<float>::infinity());
    auto lang_id = logits.argmax().item<int>();
    std::cout << "Detected language: " << id2lang[lang_id] << std::endl;
    return lang_id;
}

void OnnxModel::parseElements(const std::string& str, std::vector<int>& vec) {
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        vec.push_back(std::stoi(item));
    }
}

void OnnxModel::split(const std::string& str, char delim, std::vector<std::string>& tokens) {
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        tokens.push_back(item);
    }
}
