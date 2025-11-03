#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_GLCORE
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"
#include "imgui.h"
#include "util/sokol_imgui.h"
#include "fonts/Cousine-Regular.cpp"

#include "StableDiffusion/UNet.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <nfd.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <filesystem>

sg_pass_action pass_action{};

// Global state for the application
struct AppState {
    // UI inputs
    char prompt[512] = "a photo of an astronaut riding a horse on mars";
    int numInferenceSteps = 15;
    float guidanceScale = 7.5f;
    int deviceId = 0;
    
    // Model paths
    char textEncoderPath[512] = "E:/SW/ML/stable-diffusion-1.5-onnx/text_encoder/model.onnx";
    char unetPath[512] = "E:/SW/ML/stable-diffusion-1.5-onnx/unet/model.onnx";
    char vaeDecoderPath[512] = "E:/SW/ML/stable-diffusion-1.5-onnx/vae_decoder/model.onnx";
    char safetyModelPath[512] = "E:/SW/ML/stable-diffusion-1.5-onnx/safety_checker/model.onnx";
    
    // Generation state
    std::atomic<bool> isGenerating{false};
    std::atomic<bool> hasNewImage{false};
    std::string statusMessage = "Ready";
    std::string lastGenerationTime = "";
    
    // Image data
    std::vector<uint8_t> imageData;
    sg_image generatedImage = {0};
    sg_view generatedImageView = {0};
    bool imageValid = false;
    
    // Save state
    std::string lastPrompt = "";
    
    // Thread
    std::thread* generationThread = nullptr;
    std::mutex dataMutex;
} appState;

// Function to save the current image
void saveImage() {
    if (appState.imageData.empty()) {
        appState.statusMessage = "No image to save!";
        return;
    }
    
    // Initialize NFD
    NFD_Init();
    
    // Generate default filename
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", now);
    
    char defaultName[256];
    snprintf(defaultName, sizeof(defaultName), "sd_image_%s_Steps%d_Scale%.1f", 
        buf, appState.numInferenceSteps, appState.guidanceScale);
    
    // Define file filters
    nfdfilteritem_t filters[2] = {
        { "PNG Image", "png" },
        { "JPEG Image", "jpg,jpeg" }
    };
    
    nfdchar_t* outPath = nullptr;
    nfdresult_t result = NFD_SaveDialog(&outPath, filters, 2, nullptr, defaultName);
    
    if (result == NFD_OKAY) {
        std::string savePath(outPath);
        NFD_FreePath(outPath);
        
        // Get file extension
        std::filesystem::path filePath(savePath);
        std::string extension = filePath.extension().string();
        
        // Convert extension to lowercase
        for (auto& c : extension) {
            c = std::tolower(c);
        }
        
        // Save based on extension
        bool saveSuccess = false;
        if (extension == ".png") {
            saveSuccess = stbi_write_png(savePath.c_str(), 512, 512, 4, 
                appState.imageData.data(), 512*4) != 0;
        } else if (extension == ".jpg" || extension == ".jpeg") {
            saveSuccess = stbi_write_jpg(savePath.c_str(), 512, 512, 4, 
                appState.imageData.data(), 100) != 0;
        } else {
            // Default to PNG if no extension or unknown extension
            savePath += ".png";
            saveSuccess = stbi_write_png(savePath.c_str(), 512, 512, 4, 
                appState.imageData.data(), 512*4) != 0;
        }
        
        if (saveSuccess) {
            char statusStr[512];
            snprintf(statusStr, sizeof(statusStr), "Image saved as %s", savePath.c_str());
            appState.statusMessage = statusStr;
        } else {
            appState.statusMessage = "Failed to save image!";
        }
    } else if (result == NFD_CANCEL) {
        appState.statusMessage = "Save cancelled";
    } else {
        char statusStr[512];
        snprintf(statusStr, sizeof(statusStr), "Error: %s", NFD_GetError());
        appState.statusMessage = statusStr;
    }
    
    NFD_Quit();
}

// Helper function to convert std::string to std::wstring
std::wstring stringToWString(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

// Generation function to run in background thread
void generateImage() {
    try {
        appState.statusMessage = "Initializing...";
        auto timeStart = std::chrono::high_resolution_clock::now();
        
        std::string prompt;
        StableDiffusionConfig config{};
        
        {
            std::lock_guard<std::mutex> lock(appState.dataMutex);
            prompt = std::string(appState.prompt);
            config.NumInferenceSteps = appState.numInferenceSteps;
            config.GuidanceScale = appState.guidanceScale;
            config.ExecutionProviderTarget = StableDiffusionConfig::ExecutionProvider::DirectML;
            config.DeviceId = appState.deviceId;
            
            config.TextEncoderOnnxPath = stringToWString(std::string(appState.textEncoderPath));
            config.UnetOnnxPath = stringToWString(std::string(appState.unetPath));
            config.VaeDecoderOnnxPath = stringToWString(std::string(appState.vaeDecoderPath));
            config.SafetyModelPath = stringToWString(std::string(appState.safetyModelPath));
            
            config.memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        }
        
        appState.statusMessage = "Generating image...";
        auto rgbaData = UNet::Inference(prompt, config);
        
        auto timeEnd = std::chrono::high_resolution_clock::now();
        uint64_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
        
        {
            std::lock_guard<std::mutex> lock(appState.dataMutex);
            appState.imageData = std::move(rgbaData);
            appState.hasNewImage = true;
            appState.lastPrompt = prompt;
            
            char timeStr[64];
            snprintf(timeStr, sizeof(timeStr), "%.2fs", milliseconds / 1000.0f);
            appState.lastGenerationTime = timeStr;
            
            char statusStr[512];
            snprintf(statusStr, sizeof(statusStr), "Image generated in %.2fs (click Save to export)", 
                milliseconds / 1000.0f);
            appState.statusMessage = statusStr;
        }
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(appState.dataMutex);
        appState.statusMessage = std::string("Error: ") + e.what();
    }
    
    appState.isGenerating = false;
}

void init() {
    sg_desc desc = {};
    desc.environment = sglue_environment();
    desc.logger.func = slog_func;
    sg_setup(&desc);

    simgui_desc_t simgui_desc = {};
    simgui_desc.no_default_font = true;
    simgui_setup(&simgui_desc);

    ImGui::CreateContext();
    ImGuiIO* io = &ImGui::GetIO();
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    //io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    //io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
    //io->ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;


    // IMGUI Font texture init
    if( !ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 18.f) )
    {
        ImGui::GetIO().Fonts->AddFontDefault();
    }

    pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
    pass_action.colors[0].clear_value = {0.45f, 0.55f, 0.60f, 1.00f};
}

void frame() {
    const int width = sapp_width();
    const int height = sapp_height();

    sg_pass pass{};
    pass.action = pass_action;
    pass.swapchain = sglue_swapchain();
    sg_begin_pass(&pass);

    simgui_new_frame({ width, height, sapp_frame_duration(), sapp_dpi_scale() });

    // Check if there's a new image to upload to GPU
    if (appState.hasNewImage.load()) {
        std::lock_guard<std::mutex> lock(appState.dataMutex);
        if (!appState.imageData.empty()) {
            // Destroy old image and view if exists
            if (appState.imageValid) {
                if (appState.generatedImageView.id != 0) {
                    sg_destroy_view(appState.generatedImageView);
                    appState.generatedImageView.id = 0;
                }
                if (appState.generatedImage.id != 0) {
                    sg_destroy_image(appState.generatedImage);
                    appState.generatedImage.id = 0;
                }
            }
            
            // Create new image
            sg_image_desc img_desc{};
            img_desc.width = 512;
            img_desc.height = 512;
            img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
            img_desc.data.mip_levels[0].ptr = appState.imageData.data();
            img_desc.data.mip_levels[0].size = appState.imageData.size();
            
            appState.generatedImage = sg_make_image(&img_desc);
            
            // Create view for the image
            sg_view_desc view_desc{};
            view_desc.texture.image = appState.generatedImage;
            appState.generatedImageView = sg_make_view(&view_desc);
            
            appState.imageValid = true;
            appState.hasNewImage = false;
        }
    }

    // Main window
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(450, height), ImGuiCond_Always);
    ImGui::Begin("Stable Diffusion Text to Image", nullptr, 
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    
    ImGui::SeparatorText("Prompt");
    ImGui::InputTextMultiline("##prompt", appState.prompt, sizeof(appState.prompt), 
        ImVec2(-1, 80), ImGuiInputTextFlags_WordWrap);
    
    ImGui::SeparatorText("Generation Parameters");
    ImGui::SliderInt("Inference Steps", &appState.numInferenceSteps, 1, 100);
    ImGui::SliderFloat("Guidance Scale", &appState.guidanceScale, 1.0f, 20.0f, "%.1f");
    ImGui::SliderInt("Device ID", &appState.deviceId, 0, 3);
    
    ImGui::SeparatorText("Model Paths");
    if (ImGui::TreeNode("Model Configuration")) {
        ImGui::InputText("Text Encoder", appState.textEncoderPath, sizeof(appState.textEncoderPath));
        ImGui::InputText("UNet", appState.unetPath, sizeof(appState.unetPath));
        ImGui::InputText("VAE Decoder", appState.vaeDecoderPath, sizeof(appState.vaeDecoderPath));
        ImGui::InputText("Safety Model", appState.safetyModelPath, sizeof(appState.safetyModelPath));
        ImGui::TreePop();
    }
    
    ImGui::Separator();
    
    // Generate button
    bool generating = appState.isGenerating.load();
    if (generating) {
        ImGui::BeginDisabled();
    }
    
    if (ImGui::Button("Generate Image", ImVec2(-1, 40))) {
        // Start generation in background thread
        if (appState.generationThread != nullptr) {
            if (appState.generationThread->joinable()) {
                appState.generationThread->join();
            }
            delete appState.generationThread;
        }
        
        appState.isGenerating = true;
        appState.generationThread = new std::thread(generateImage);
    }
    
    if (generating) {
        ImGui::EndDisabled();
    }
    
    // Save button
    if (appState.imageData.empty() || generating) {
        ImGui::BeginDisabled();
    }
    
    if (ImGui::Button("Save Image", ImVec2(-1, 30))) {
        saveImage();
    }
    
    if (appState.imageData.empty() || generating) {
        ImGui::EndDisabled();
    }
    
    // Status
    ImGui::Separator();
    if (generating) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Status: %s", appState.statusMessage.c_str());
        ImGui::ProgressBar(-1.0f * ImGui::GetTime(), ImVec2(-1, 0));
    } else {
        ImGui::Text("Status: %s", appState.statusMessage.c_str());
    }
    
    if (!appState.lastGenerationTime.empty()) {
        ImGui::Text("Last generation time: %s", appState.lastGenerationTime.c_str());
    }
    
    ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("FPS: %.1f", io.Framerate);
    
    ImGui::End();
    
    // Image display window
    ImGui::SetNextWindowPos(ImVec2(450, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(width - 450, height), ImGuiCond_Always);
    ImGui::Begin("Generated Image", nullptr, 
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    
    if (appState.imageValid && appState.generatedImageView.id != 0) {
        ImVec2 windowSize = ImGui::GetContentRegionAvail();
        float imageSize = std::min(windowSize.x, windowSize.y);
        
        // Center the image
        ImVec2 imagePos = ImGui::GetCursorPos();
        imagePos.x += (windowSize.x - imageSize) * 0.5f;
        imagePos.y += (windowSize.y - imageSize) * 0.5f;
        ImGui::SetCursorPos(imagePos);

        // Use the stored view to get ImTextureID
        ImTextureID imtex_id = simgui_imtextureid(appState.generatedImageView);

        ImGui::Image(imtex_id, ImVec2(imageSize, imageSize));
    } else {
        ImVec2 windowSize = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos(ImVec2(windowSize.x * 0.5f - 100, windowSize.y * 0.5f - 10));
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No image generated yet");
    }
    
    ImGui::End();

    simgui_render();

    sg_end_pass();
    sg_commit();
}

void cleanup() {
    // Wait for generation thread to finish
    if (appState.generationThread != nullptr) {
        if (appState.generationThread->joinable()) {
            appState.generationThread->join();
        }
        delete appState.generationThread;
        appState.generationThread = nullptr;
    }
    
    // Clean up image and view
    if (appState.imageValid) {
        if (appState.generatedImageView.id != 0) {
            sg_destroy_view(appState.generatedImageView);
        }
        if (appState.generatedImage.id != 0) {
            sg_destroy_image(appState.generatedImage);
        }
    }
    
    simgui_shutdown();
    sg_shutdown();
}

void input(const sapp_event* event) {
    simgui_handle_event(event);
}

int main(int argc, const char* argv[]) {
    sapp_desc desc = {};
    desc.init_cb = init;
    desc.frame_cb = frame;
    desc.cleanup_cb = cleanup;
    desc.event_cb = input;
    desc.width = 1280;
    desc.height = 720;
    desc.high_dpi = true;
    desc.window_title = "Stable Diffusion Text to Image";
    desc.icon.sokol_default = true;
    desc.logger.func = slog_func;
    sapp_run(desc);

    return 0;
}
