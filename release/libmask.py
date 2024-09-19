import ctypes
import numpy as np
import cv2  # Assuming you want to use OpenCV for image loading and manipulation

# Load the shared library
lib = ctypes.CDLL("./release/x64/libmask.so")


# Define the params structure
class SamParams(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("model", ctypes.c_char * 256),
        ("fname_inp", ctypes.c_char * 256),
        ("fname_out", ctypes.c_char * 256),
    ]


# Define the function signature for generate_mask_wrapper
lib.generate_mask_wrapper.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # Image data
    ctypes.c_int,  # Width
    ctypes.c_int,  # Height
    ctypes.c_float,  # X coordinate
    ctypes.c_float,  # Y coordinate
    SamParams,  # Params
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # Output data
    ctypes.POINTER(ctypes.c_int),  # Output size
]
lib.generate_mask_wrapper.restype = None


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
    params.seed = 42
    params.n_threads = 4
    params.model = (
        b"./checkpoints/ggml-model-f16.bin"  # Replace with the actual model path
    )
    params.fname_inp = b"img.jpg"
    params.fname_out = b"output_mask.png"

    # Prepare output variables
    output_data = ctypes.POINTER(ctypes.c_ubyte)()
    output_size = ctypes.c_int()

    # Call the function to generate the mask
    lib.generate_mask_wrapper(
        (ctypes.c_ubyte * len(image_data))(*image_data),
        width,
        height,
        100.0,  # x coordinate (replace with actual value)
        150.0,  # y coordinate (replace with actual value)
        params,
        ctypes.byref(output_data),
        ctypes.byref(output_size),
    )
    # print size of output_data
    print(output_size.value)

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
