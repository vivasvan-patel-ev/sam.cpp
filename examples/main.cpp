#include "sam.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "httplib.h" // For the HTTP server

#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

httplib::Server svr; // Declare server globally for access during shutdown

// Signal handler for graceful shutdown
void signal_handler(int signum) {
  printf("Interrupt signal (%d) received. Shutting down gracefully...\n",
         signum);
  svr.stop();
}

// Function to print usage information
static void print_usage(int argc, char **argv, const sam_params &params) {
  std::cerr << "usage: " << argv[0] << " [options]\n";
  std::cerr << "\n";
  std::cerr << "options:\n";
  std::cerr << "  -h, --help            show this help message and exit\n";
  std::cerr << "  -s SEED, --seed SEED  RNG seed (default: -1)\n";
  std::cerr << "  -t N, --threads N     number of threads to use during "
               "computation (default: "
            << params.n_threads << ")\n";
  std::cerr << "  -m FNAME, --model FNAME\n";
  std::cerr << "                        model path (default: " << params.model
            << ")\n";
  std::cerr << "  -i FNAME, --inp FNAME\n";
  std::cerr << "                        input file (default: "
            << params.fname_inp << ")\n";
  std::cerr << "  -o FNAME, --out FNAME\n";
  std::cerr << "                        output file (default: "
            << params.fname_out << ")\n";
  std::cerr << "\n";
}

// Function to parse command-line arguments and fill the params structure
static bool params_parse(int argc, char **argv, sam_params &params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-s" || arg == "--seed") {
      params.seed = std::stoi(argv[++i]);
    } else if (arg == "-t" || arg == "--threads") {
      params.n_threads = std::stoi(argv[++i]);
    } else if (arg == "-m" || arg == "--model") {
      params.model = argv[++i];
    } else if (arg == "-i" || arg == "--inp") {
      params.fname_inp = argv[++i];
    } else if (arg == "-o" || arg == "--out") {
      params.fname_out = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argc, argv, params);
      exit(0);
    } else {
      std::cerr << "error: unknown argument: " << arg << "\n";
      print_usage(argc, argv, params);
      return false;
    }
  }

  return true;
}

// Helper function to decode image data from Base64
std::vector<unsigned char> base64_decode(const std::string &encoded) {
  std::string cmd = "echo " + encoded + " | base64 --decode > decoded_img.jpg";
  system(cmd.c_str());
  std::ifstream file("decoded_img.jpg", std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
  file.close();
  return buffer;
}

// Downscale image with nearest-neighbor interpolation
static sam_image_u8 downscale_img(sam_image_u8 &img, float scale) {
  sam_image_u8 new_img;

  int width = img.nx;
  int height = img.ny;

  int new_width = img.nx / scale + 0.5f;
  int new_height = img.ny / scale + 0.5f;

  new_img.nx = new_width;
  new_img.ny = new_height;
  new_img.data.resize(new_img.nx * new_img.ny * 3);

  for (int y = 0; y < new_height; ++y) {
    for (int x = 0; x < new_width; ++x) {
      int src_x = (x + 0.5f) * scale - 0.5f;
      int src_y = (y + 0.5f) * scale - 0.5f;

      int src_index = (src_y * width + src_x) * 3;
      int dest_index = (y * new_width + x) * 3;

      for (int c = 0; c < 3; ++c) {
        new_img.data[dest_index + c] = img.data[src_index + c];
      }
    }
  }

  return new_img;
}

// Load image from binary data
static bool load_image_from_memory(const std::vector<unsigned char> &data,
                                   sam_image_u8 &img) {
  int nx, ny, nc;
  auto data_ptr =
      stbi_load_from_memory(data.data(), data.size(), &nx, &ny, &nc, 3);
  if (!data_ptr) {
    fprintf(stderr, "Failed to load image from memory\n");
    return false;
  }

  img.nx = nx;
  img.ny = ny;
  img.data.resize(nx * ny * 3);
  memcpy(img.data.data(), data_ptr, nx * ny * 3);

  stbi_image_free(data_ptr);
  return true;
}

// Generate mask based on image and point
std::vector<unsigned char> generate_mask(const sam_image_u8 &img, float x,
                                         float y, const sam_params &params,
                                         sam_state &state) {
  sam_point pt{x, y};

  // // save the original image to disk
  // stbi_write_jpg("original_image.jpg", img.nx, img.ny, 3, img.data.data(), 100);


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
  std::vector<unsigned char> output_img(mask.nx * mask.ny * 3);

  // Convert mask to RGB image
  for (int i = 0; i < mask.nx * mask.ny; ++i) {
    output_img[3 * i + 0] = mask.data[i];
    output_img[3 * i + 1] = mask.data[i];
    output_img[3 * i + 2] = mask.data[i];
  }

  // Encode the RGB data to a PNG
  std::vector<unsigned char> png_data;
  stbi_write_png_to_func(
      [](void *context, void *data, int size) {
        std::vector<unsigned char> *png_data =
            static_cast<std::vector<unsigned char> *>(context);
        png_data->insert(png_data->end(), (unsigned char *)data,
                         (unsigned char *)data + size);
      },
      &png_data, mask.nx, mask.ny, 3, output_img.data(), mask.nx * 3);

  return png_data;
}

// Downscale the image to fit within a default screen size (1024x1024)
static bool downscale_img_to_screen(sam_image_u8 &img) {
  const int max_width = 1024;  // Default screen width
  const int max_height = 1024; // Default screen height

  fprintf(stderr, "%s: default screen size (%d x %d) \n", __func__, max_width,
          max_height);
  fprintf(stderr, "%s: img size (%d x %d) \n", __func__, img.nx, img.ny);

  // Check if the image exceeds the maximum allowed dimensions
  if (img.nx > max_width || img.ny > max_height) {

    printf("Scaling based on above values");

    // Calculate the scaling factor to fit within the 1024x1024 size while
    // preserving the aspect ratio
    const float scale_x = static_cast<float>(img.nx) / max_width;
    const float scale_y = static_cast<float>(img.ny) / max_height;
    const float scale =
        std::max(scale_x, scale_y); // Use the largest scaling factor

    fprintf(stderr, "%s: Scaling image by factor %f\n", __func__, scale);

    // Downscale the image using the calculated scaling factor
    img = downscale_img(img, scale);
  }

  // printf("returning true\n");

  //   flush
  fflush(stdout);

  return true;
}

int main(int argc, char **argv) {
  // Register signal handler for graceful shutdown
  signal(SIGINT, signal_handler);

  sam_params params;
  if (!params_parse(argc, argv, params)) {
    return 1;
  }

  if (params.seed < 0) {
    params.seed = time(NULL);
  }
  fprintf(stderr, "Seed = %d\n", params.seed);

  // Load SAM model
  std::shared_ptr<sam_state> state = sam_load_model(params);
  if (!state) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
  }
  fprintf(stderr, "%s: t_load_ms = %d ms\n", __func__, state->t_load_ms);

  // Create HTTP server
  svr.Post("/generate_mask", [&](const httplib::Request &req,
                                 httplib::Response &res) {
    // Get current time in IST
    time_t now = time(0);
    tm *ltm = localtime(&now);
    printf("Received request @ %02d-%02d-%04d %02d:%02d:%02d IST\n",
           ltm->tm_mday, ltm->tm_mon + 1, ltm->tm_year + 1900, ltm->tm_hour,
           ltm->tm_min, ltm->tm_sec);
    // print request params
    printf("Request params:\n");
    for (const auto &param : req.params) {
      printf("  %s: %s\n", param.first.c_str(), param.second.c_str());
    }

    // Print information about the file received
    if (req.has_file("image")) {
      auto image_file = req.get_file_value("image");
      printf("Received file: %s, size: %zu bytes\n",
             image_file.filename.c_str(), image_file.content.size());
    } else {
      printf("No image file received.\n");
      res.status = 400;
      res.set_content("No image file in request", "text/plain");
      return;
    }

    // Check if 'x' and 'y' params exist
    if (!req.has_param("x") || !req.has_param("y")) {
      printf("Missing 'x' or 'y' parameters.\n");
      res.status = 400;
      res.set_content("Missing 'x' or 'y' parameters", "text/plain");
      return;
    }

    // Get point
    float x = std::stof(req.get_param_value("x"));
    float y = std::stof(req.get_param_value("y"));
    printf("x: %f, y: %f\n", x, y);

    if (req.has_file("image") && req.has_param("x") && req.has_param("y")) {
      // Get image file data
      auto image_file = req.get_file_value("image");
      std::vector<unsigned char> image_data(image_file.content.begin(),
                                            image_file.content.end());

      // Load image
      sam_image_u8 img;
      if (!load_image_from_memory(image_data, img)) {
        res.status = 400;
        res.set_content("Failed to load image", "text/plain");
        return;
      }

      // Downscale image if necessary
      downscale_img_to_screen(img);

      printf("Downscale successful!\n");

      // Generate mask
      fprintf(stderr, "Generating mask...\n");
      int start_time = clock();
      auto mask_data = generate_mask(img, x, y, params, *state);
      int end_time = clock();
      fprintf(stderr, "%s: Mask generation took %d ms\n", __func__,
             (end_time - start_time) * 1000 / CLOCKS_PER_SEC);
      if (mask_data.empty()) {
        res.status = 500;
        res.set_content("Failed to generate mask", "text/plain");
        return;
      }

      // Set response
      res.set_content(reinterpret_cast<const char *>(mask_data.data()),
                      mask_data.size(), "image/png");
    } else {
      res.status = 400;
      res.set_content("Invalid request", "text/plain");
    }

    // flush
    fflush(stdout);
  });

  svr.Get("/stop", [&](const auto & /*req*/, auto & /*res*/) {
    printf("Received stop request. Shutting down server...\n");
    svr.stop();
  });

  printf("Server listening on port 42069...\n");

  try {
    svr.listen("0.0.0.0", 42069);
  } catch (const std::exception &e) {
    fprintf(stderr, "Server encountered an error: %s\n", e.what());
  }

  printf("Server stopped. Cleaning up resources...\n");

  // Clean up
  sam_deinit(*state);

  printf("Cleanup done.\n");

  return 0;
}