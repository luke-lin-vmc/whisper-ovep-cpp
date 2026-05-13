This script is modified from https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/whisper/export-onnx.py

### Steps:
Run the following commands to export models
```
pip install -r requirements.txt
python export-onnx.py --model small
```
* Supported models: ```tiny``` ```base``` ```small``` ```medium``` ```large-v1``` ```large-v2``` ```large```(aka large v3) and ```turbo```(aka large v3 turbo)<br>

Models (```*.onnx``` and ```*.weights``` if model is big) and tokenizer (```*tokens.txt```) will be exported under the same directory
```
small-encoder.onnx
small-decoder.onnx
small-tokens.txt
```
