# sd15cpp
Stable Diffusion with C++ and ONNX Runtime

## how to use

1. download sd1.5 onnx models

```bash
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx
# or
git clone https://huggingface.co/jackos/stable-diffusion-1.5-onnx
```

2. change config in program.cs

```cpp
// txt2img_cli.cpp
config.TextEncoderOnnxPath = L"G:/wrokspace2/ML/stable-diffusion-v1-5/text_encoder/model.onnx";
config.UnetOnnxPath        = L"G:/wrokspace2/ML/stable-diffusion-v1-5/unet/model.onnx";
config.VaeDecoderOnnxPath  = L"G:/wrokspace2/ML/stable-diffusion-v1-5/vae_decoder/model.onnx";
config.SafetyModelPath     = L"G:/wrokspace2/ML/stable-diffusion-v1-5/safety_checker/model.onnx";
```

3. change prompt and run

```cpp
// txt2img_cli.cpp
std::string prompt = "a photo of an astronaut riding a horse on mars";
```

## references
- https://github.com/axodox/axodox-machinelearning
- https://github.com/cassiebreviu/StableDiffusion
- https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
- https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx

# Summary

Date : 2024-02-07 14:58:46

Directory hello_onnx\\src\\sd15cpp

Total : 18 files,  1316 codes, 92 comments, 308 blanks, all 1716 lines

## Languages
| language | files | code | comment | blank | total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| C++ | 17 | 1,291 | 92 | 299 | 1,682 |
| Markdown | 1 | 25 | 0 | 9 | 34 |

## Directories
| path | files | code | comment | blank | total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| . | 18 | 1,316 | 92 | 308 | 1,716 |
| . (Files) | 2 | 60 | 12 | 23 | 95 |
| StableDiffusion | 16 | 1,256 | 80 | 285 | 1,621 |
