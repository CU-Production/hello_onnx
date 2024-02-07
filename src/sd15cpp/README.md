# sd15cpp
Stable Diffusion with C++ and ONNX Runtime

## how to use

1. download sd1.5 onnx models

```bash
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx
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
