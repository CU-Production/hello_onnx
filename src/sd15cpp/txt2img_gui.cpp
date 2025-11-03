#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_GLCORE33
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
#include <iostream>
#include <chrono>
#include <ctime>

sg_pass_action pass_action{};

void init() {
    sg_desc desc = {};
    desc.context = sapp_sgcontext();
    desc.logger.func = slog_func;
    sg_setup(&desc);

    simgui_desc_t simgui_desc = {};
    simgui_desc.no_default_font = true;
    simgui_setup(&simgui_desc);

    ImGui::CreateContext();
    ImGuiIO* io = &ImGui::GetIO();
    io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    //io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io->ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoTaskBarIcons;
    //io->ConfigFlags |= ImGuiConfigFlags_ViewportsNoMerge;


    // IMGUI Font texture init
    if( !ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(Cousine_Regular_compressed_data, Cousine_Regular_compressed_size, 18.f) )
    {
        ImGui::GetIO().Fonts->AddFontDefault();
    }
    {

        ImGuiIO& io = ImGui::GetIO();

        // Build texture atlas
        unsigned char* pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);   // Load as RGBA 32-bit (75% of the memory is wasted, but default font is so small) because it is more likely to be compatible with user's existing shaders. If your ImTextureId represent a higher-level concept than just a GL texture id, consider calling GetTexDataAsAlpha8() instead to save on GPU memory.

        sg_image_desc img_desc{};
        img_desc.width = width;
        img_desc.height = height;
        img_desc.label = "font-texture";
        img_desc.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
        img_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
        img_desc.min_filter = SG_FILTER_LINEAR;
        img_desc.mag_filter = SG_FILTER_LINEAR;
        img_desc.data.subimage[0][0].ptr = pixels;
        img_desc.data.subimage[0][0].size = (width * height) * sizeof(uint32_t);

        sg_image img = sg_make_image(&img_desc);

        io.Fonts->TexID = (ImTextureID)(uintptr_t)img.id;
    }

    pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
    pass_action.colors[0].clear_value = {0.45f, 0.55f, 0.60f, 1.00f};
}

bool show_demo_window = true;
bool show_another_window = false;
ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
void frame() {
    const int width = sapp_width();
    const int height = sapp_height();

    sg_begin_default_pass(&pass_action, width, height);

    simgui_new_frame({ width, height, sapp_frame_duration(), sapp_dpi_scale() });

    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();
    }

    simgui_render();

    sg_end_pass();
    sg_commit();
}

void cleanup() {
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
    desc.window_title = "Mini C IDE";
    desc.icon.sokol_default = true;
    desc.logger.func = slog_func;
    sapp_run(desc);

    return 0;
}
