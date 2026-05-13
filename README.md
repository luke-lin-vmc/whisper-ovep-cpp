# About whisper-ovep-cpp (WindowsML 2.0)
This C++ pipeline is to show how to run Whisper on Intel CPU/GPU/NPU thru [Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is derived from sherpa-onnx project
https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/whisper

# Features
This pipeline supports WindowsML 2.0
* No need to install ONNX runtime (already built-in in Windows 11 24H2 or later)
* No need to install OpenVINO Execution Provider (already built-in in Windows 11 24H2 or later, if the system is with INTEL CPU)

# Quick Steps
## Prepare models
Run the following commands to export models
```
cd Script
pip install -r requirements.txt
python export-onnx.py --model small
```
* Supported models: ```tiny``` ```base``` ```small``` ```medium``` ```large-v1``` ```large-v2``` ```large```(aka large v3) and ```turbo```(aka large v3 turbo)<br>

Models (```*.onnx``` and ```*.weights``` if model is big) and tokenizer (```*tokens.txt```) will be exported under .\Script directory
```
small-encoder.onnx
small-decoder.onnx
small-tokens.txt
```
Log file [export_model.log](https://github.com/luke-lin-vmc/whisper-ovep-cpp/blob/main/log/export_model.log) is provided for reference

## Build
Make sure the internet connection is working. The build script will download [Microsoft.WindowsAppSDK.ML v2.0.300](https://www.nuget.org/packages/Microsoft.WindowsAppSDK.ML/2.0.300/) and [LibTorch v2.8.0](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip) from internet

```
cd ..
mkdir build
cd build 
cmake ..
cmake --build . --config Release
```
```whisper_onnx.exe``` will be generated under .\build\Release

Log file [cmake_and_build.log](https://github.com/luke-lin-vmc/whisper-ovep-cpp/blob/main/log/cmake_and_build.log) is provided for reference

## Make a package
Run below command to make a package
```
cmake --install . --config Release
```
This command will copy the following files into ```..\package``` directory

* Execution file ```whisper_onnx.exe```
* Models and tokenizer under ```.\Script``` (```*.onnx```, ```*.weights``` and ```*tokens.txt```)
* Simple example ```simple_run.bat```
* Audio sample ```how_are_you_doing_today.wav``` ([source](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav))
* Required ```DLLs```

The ```..\pacakge``` directory should look like
```
C:\GitHub\whisper-ovep-cpp\build>dir ..\package /o
 Volume in drive C has no label.
 Volume Serial Number is 6E19-CE53

 Directory of C:\GitHub\whisper-ovep-cpp\package

05/12/2026  03:41 PM    <DIR>          .
05/12/2026  03:41 PM    <DIR>          ..
07/23/2025  03:35 PM           367,104 asmjit.dll
07/23/2025  03:42 PM         1,046,528 c10.dll
04/23/2026  07:33 AM        18,710,800 DirectML.dll
07/23/2025  03:35 PM         5,721,600 fbgemm.dll
05/08/2026  03:58 PM            78,806 how_are_you_doing_today.wav
05/08/2026  03:58 PM           112,128 kaldi-native-fbank-core.dll
07/23/2025  03:29 PM         1,613,680 libiomp5md.dll
07/23/2025  03:29 PM            43,888 libiompstubs5md.dll
04/23/2026  07:33 AM           890,400 Microsoft.Windows.AI.MachineLearning.dll
04/23/2026  07:33 AM        21,670,840 onnxruntime.dll
05/08/2026  04:43 PM            10,126 README.md
05/08/2026  04:00 PM                54 simple_run.bat
05/07/2026  12:00 PM       557,868,083 small-decoder.onnx
05/07/2026  12:00 PM       409,406,960 small-encoder.onnx
05/07/2026  11:59 AM           866,987 small-tokens.txt
05/08/2026  03:58 PM         2,603,520 sndfile.dll
07/23/2025  04:47 PM             9,728 torch.dll
07/23/2025  04:46 PM       255,948,800 torch_cpu.dll
07/23/2025  03:42 PM             9,728 torch_global_deps.dll
12/18/2020  02:53 PM           195,072 uv.dll
05/12/2026  03:38 PM           438,784 whisper_onnx.exe
              21 File(s)  1,277,613,616 bytes
               2 Dir(s)  520,396,967,936 bytes free
```
Log file [make_package.log](https://github.com/luke-lin-vmc/whisper-ovep-cpp/blob/main/log/make_package.log) is provided for reference

## Run
```
cd ..\package
```
### Usage
```
usage: whisper_onnx.exe <audio_file> <model_type> <device>
```
### Run on CPU
```
# you may simply input "simple_run.bat"
whisper_onnx.exe how_are_you_doing_today.wav small CPU
```
### Run on GPU
```
whisper_onnx.exe how_are_you_doing_today.wav small GPU
```
### Run on NPU
```
whisper_onnx.exe how_are_you_doing_today.wav small NPU
```
:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been [enabled for NPU](https://github.com/luke-lin-vmc/whisper-ovep-cpp/blob/main/src/whisper.cpp#L223) to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be much faster as model compilation has been skipped.

### Run on default CPUExecutionProvider
```
whisper_onnx.exe how_are_you_doing_today.wav small DEFAULT
```
### AUTO Select Inference Device
Set device to `AUTO` to select the inference device automatically. The selection priority is `NPU`, `GPU`, then `CPU`
```
whisper_onnx.exe how_are_you_doing_today.wav small AUTO
```
:warning:[NOTE] If the selected inference device is NPU, the 1st running will take long time (about 3 minutes) on model compiling. See [Run on NPU](https://github.com/luke-lin-vmc/whisper-ovep-cpp#run-on-npu) for the detail.


### Sample run log
```
C:\GitHub\whisper-ovep-cpp\package>whisper_onnx.exe how_are_you_doing_today.wav small AUTO
Whisper encoder model: small-encoder.onnx
Whisper decoder model: small-decoder.onnx
Whisper tokens: small-tokens.txt
Inference device: AUTO
Provider: OpenVINOExecutionProvider
  Library: C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.74.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll
  Registered!

Available EP devices (8):
  - CPUExecutionProvider
  - DmlExecutionProvider
  - OpenVINOExecutionProvider
  - OpenVINOExecutionProvider
  - OpenVINOExecutionProvider
  - OpenVINOExecutionProvider.AUTO
  - OpenVINOExecutionProvider.AUTO
  - OpenVINOExecutionProvider.AUTO

Registered 1 certified provider(s).
Inference device: OpenVINO EP with device = AUTO
AUTO Selected inference device: NPU
Encoder path: small-encoder.onnx
Encoder input name 0: mel
Encoder output name 0: n_layer_cross_k
Encoder output name 1: n_layer_cross_v
Decoder path: small-decoder.onnx
Encoder processing time: 239 ms
Decoder processing time: 125 ms
Detected language: en
Decoder processing time: 31 ms
Decoder processing time: 30 ms
Decoder processing time: 32 ms
Decoder processing time: 29 ms
Decoder processing time: 29 ms
Decoder processing time: 29 ms
Decoder processing time: 27 ms
Decoder processing time: 25 ms
Decoder processing time: 35 ms
Decoder processing time: 27 ms

Transcribed:
 How are you doing today?

Processing complete. Exiting.
Successfully unregistered: OpenVINOExecutionProvider
```
Full log [run.log](https://github.com/luke-lin-vmc/whisper-ovep-cpp/blob/main/log/run.log) is provided for reference

## Tested Models and Devices
The test was done on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system, with
* ```OS: Windows 11 25H2```
* ```iGPU: Intel(R) Arc(TM) 130V GPU (16GB), driver 32.0.101.8508 (1/30/2026)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4724 (3/19/2026)```
### Result
| Model                     | CPU    | GPU    | NPU    |
|---------------------------|--------|--------|--------|
| tiny                      | OK     | OK     | OK     |
| base                      | OK     | OK     | OK     |
| small                     | OK     | OK     | OK     |
| medium                    | OK     | OK     | OK     |
| large-v1                  | OK*    | OK*    | Fail** |
| large-v2                  | OK     | OK     | Fail** |
| large<br>(large v3)       | OK     | OK     | Fail** |
| turbo<br>(large v3 turbo) | OK     | OK     | OK     |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*&nbsp;Pileline worked fine but the EN speech was misdetected as PL<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**&nbsp;Pileline didn't worked due to insufficient memory

## Known issues
* Only Arc iGPUs (Meteor Lake, Lunar Lake, Panther Lake and Arrow Lake H-series) are supported. Running on unsupported iGPU (such like Iris Xe or UHD) may lead to incorrect output, such as "!!!!!!!!!!!!!!".
