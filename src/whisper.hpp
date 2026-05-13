// whisper.hpp

#ifndef WHISPER_HPP
#define WHISPER_HPP

#include <string>
#include <vector>
#include <map>
#include <torch/torch.h>

#include <onnxruntime_cxx_api.h>
#include <sndfile.hh> // For sound file handling
#include <utility>

// Forward Declarations
class OnnxModel;

// Function Declarations
std::pair<std::vector<float>, int> load_audio(const std::string& filename);
torch::Tensor compute_features(const std::vector<float>& samples, int sample_rate, int dim);
torch::Tensor compute_features_audio(const std::vector<float>& wave, int sample_rate, int dim);
std::map<int, std::string> loadTokens(const std::string& filename);

// Class Declaration
class OnnxModel {
public:
    // Other Member variables
    size_t n_text_layer, n_text_ctx, n_text_state, n_mels;
    int sot, eot, translate, transcribe, no_timestamps, no_speech, blank;
    std::vector<int> sot_sequence, all_language_tokens;
    std::vector<std::string> all_language_codes;
    std::unordered_map<std::string, int> lang2id;
    std::unordered_map<int, std::string> id2lang;
    bool is_multilingual;
    std::string whisper_encoder_device;
    std::string whisper_decoder_device;

    OnnxModel(Ort::Env& env, const ORTCHAR_T* encoder_path, const ORTCHAR_T* decoder_path, std::string inference_device);
    void initEncoder(Ort::Env& env, const ORTCHAR_T* encoder_path);
    void initDecoder(Ort::Env& env, const ORTCHAR_T* decoder_path);
    std::pair<torch::Tensor, torch::Tensor> runEncoder(torch::Tensor mel);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> runDecoder(
        const torch::Tensor& tokens, torch::Tensor& n_layer_self_k_cache,
        torch::Tensor& n_layer_self_v_cache, const torch::Tensor& n_layer_cross_k,
        const torch::Tensor& n_layer_cross_v,
        const torch::Tensor& current_index,
        const torch::Tensor& attention_mask);

    std::pair<torch::Tensor, torch::Tensor> getSelfCache();
    void suppressTokens(torch::Tensor& logits, bool is_initial);
    int detectLanguage(const torch::Tensor& n_layer_cross_k, const torch::Tensor& n_layer_cross_v);

private:
    // Private Members and Utility Functions
    void parseElements(const std::string& str, std::vector<int>& vec);
    void split(const std::string& str, char delim, std::vector<std::string>& tokens);
};

#endif // WHISPER_HPP
