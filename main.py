import argparse
import moondream as md
from async_tracking import run_async_tracking  # <-- import from the new file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add a command-line switch: --display-ui
    parser.add_argument("--display-ui", action="store_true",
                        help="If set, display the OpenCV UI with trackbars and bounding box visualization.")
    args = parser.parse_args()

    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()

    model = md.vl(api_key=api_key)
    tracking_prompt = input("What object would you like to track? ")

    # Call our newly abstracted function, passing the display_ui argument.
    run_async_tracking(model, tracking_prompt, display_ui=True)
