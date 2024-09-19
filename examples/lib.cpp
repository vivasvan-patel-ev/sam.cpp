#include "sam.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstring>
#include <vector>

extern "C" {
  // Define a function to generate the mask
  std::vector<unsigned char> generate_mask(const sam_image_u8 &img, float x,
                                          float y, const sam_params &params,
                                          sam_state &state) {
    sam_point pt{x, y};

    if (!sam_compute_embd_img(img, params.n_threads, state)) {
      printf("failed to compute encoded image\n");
    }
    printf("t_compute_img_ms = %d ms\n", state.t_compute_img_ms);

    std::vector<sam_image_u8> masks =
        sam_compute_masks(img, params.n_threads, pt, state);

    if (masks.empty()) {
      fprintf(stderr, "No mask generated\n");
      return {};
    }
    const sam_image_u8 &mask = masks[0]; // Using the first mask

    // Return the mask data directly as a 1-channel image
    return mask.data;
  }
void generate_mask_wrapper(const unsigned char *image_data, int width,
                            int height, float x, float y,
                            int32_t seed, int32_t n_threads,
                            const char *model, const char *fname_inp,
                            const char *fname_out,
                            unsigned char **output_data, int *output_size) {

    // Create sam_params struct inside the function
    sam_params params;
    params.seed = seed;
    params.n_threads = n_threads;
    params.model = model;
    params.fname_inp = fname_inp;
    params.fname_out = fname_out;
    
    sam_image_u8 img;
    img.nx = width;
    img.ny = height;
    img.data.resize(width * height * 3);
    memcpy(img.data.data(), image_data, width * height * 3);

    // Load SAM model
    std::shared_ptr<sam_state> state = sam_load_model(params);
    if (!state) {
      fprintf(stderr, "Failed to load model\n");
      return;
    }
    fprintf(stderr, "%s: t_load_ms = %d ms\n", __func__, state->t_load_ms);

    // Call the original function
    std::vector<unsigned char> mask_data =
        generate_mask(img, x, y, params, *state);

    // Allocate memory for output_data
    *output_size = mask_data.size();
    *output_data = new unsigned char[*output_size];

    // Copy data to output buffer
    memcpy(*output_data, mask_data.data(), *output_size);
  }
}