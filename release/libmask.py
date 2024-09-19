import ctypes
import numpy as np
import cv2  # Assuming you want to use OpenCV for image loading and manipulation
import os
import platform

# Determine the architecture
architecture = platform.machine()

# Set the correct path based on architecture
if architecture == "x86_64":
    lib_path = "x64"
elif architecture == "arm64" or architecture == "aarch64":
    lib_path = "arm"
else:
    raise ValueError(f"Unsupported architecture: {architecture}")

lib = ctypes.CDLL(f"./release/{lib_path}/libmask.so")


# Define the params structure
class SamParams(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("model", ctypes.c_char * 256),
        ("fname_inp", ctypes.c_char * 256),
        ("fname_out", ctypes.c_char * 256),
    ]

    def __init__(
        self,
        seed=-1,
        n_threads=4,
        model="./checkpoints/ggml-model-f16.bin",
        fname_inp="./img.jpg",
        fname_out="img.png",
    ):
        self.seed = seed
        self.n_threads = n_threads
        self.model = model.encode("utf-8")
        self.fname_inp = fname_inp.encode("utf-8")
        self.fname_out = fname_out.encode("utf-8")


# Function to load an image using OpenCV
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img


# Example usage
def main():
    # Load an image (replace 'input_image.jpg' with your image file)
    image_path = "./img.jpg"  # Path to your input image
    image = load_image(image_path)

    # Convert image to RGB format and flatten to a list
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_data = image_rgb.flatten().tolist()  # Flatten to 1D list
    width, height = image.shape[1], image.shape[0]

    # Create an instance of SamParams and set values
    params = SamParams()

    # Prepare output variables
    output_data = ctypes.POINTER(ctypes.c_ubyte)()
    output_size = ctypes.c_int()

    # Hard code data as correct types
    # image_data = (ctypes.c_ubyte * len(image_data))(*image_data)
    c_width = ctypes.c_int(width)
    c_height = ctypes.c_int(height)
    x = ctypes.c_float(100.0)  # Replace with actual value if needed
    y = ctypes.c_float(150.0)  # Replace with actual value if needed
    seed = ctypes.c_int(params.seed)
    n_threads = ctypes.c_int(params.n_threads)
    model = ctypes.c_char_p(params.model)
    fname_inp = ctypes.c_char_p(params.fname_inp)
    fname_out = ctypes.c_char_p(params.fname_out)
    output_data = ctypes.POINTER(ctypes.c_ubyte)()
    output_size = ctypes.c_int()
    # Call the function to generate the mask
    lib.generate_mask_wrapper(
        (ctypes.c_ubyte * len(image_data))(*image_data),
        c_width,
        c_height,
        x,  # x coordinate (replace with actual value)
        y,  # y coordinate (replace with actual value)
        seed,
        n_threads,
        model,  # Pass model as bytes
        fname_inp,  # Pass fname_inp as bytes
        fname_out,  # Pass fname_out as bytes
        ctypes.byref(output_data),
        ctypes.byref(output_size),
    )

    lib.generate_mask_wrapper.restype = None

    # print size of output_data
    print(output_size.value)

    if output_size.value == 0:
        print("Error in generating mask")
        return

    # Convert the pointer to bytes
    bytes_data = ctypes.string_at(output_data, output_size.value)

    # size of bytes_data
    print(len(bytes_data))

    mask_data = np.frombuffer(bytes_data, dtype=np.uint8)
    print(type(mask_data), mask_data.shape)
    # Convert the output to a numpy array
    # mask_data = np.frombuffer(output_data[: output_size.value], dtype=np.uint8)

    # Reshape the output data to the correct dimensions
    mask_image = mask_data.reshape(
        -1, width
    )  # Adjust if necessary based on output format
    print(mask_image.shape)

    # Save or display the mask image
    output_mask_path = "output_mask.png"
    cv2.imwrite(output_mask_path, mask_image)
    print(f"Mask saved to: {output_mask_path}")


if __name__ == "__main__":
    main()
