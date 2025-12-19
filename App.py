import os
from flask import Flask, render_template, request, send_file
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

PIXEL_TO_MM = 0.197889

# --------------------------------------------------
# Route for Abdominal Circumference (AC)
# --------------------------------------------------

MODEL_PATH = "D:/Alyan/Final-Year-Project/FYP/FYP-FINAL-Server-webapp/models/AC_MODEL.tflite"
PLOT_IMAGE_PATH = "static/Ac_Bpd_result_plot_images/ac_plot_image.png"


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def estimate_ellipse_length(major_axis, minor_axis, pixel_to_mm):
    a = max(major_axis, minor_axis)
    b = min(major_axis, minor_axis)
    perimeter_pixels = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    return perimeter_pixels * pixel_to_mm


def draw_ellipse(image, mask):
    modified_image = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    major_axis = minor_axis = estimated_length = 0

    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(modified_image, ellipse, (255, 255, 255), 3)

            major_axis = ellipse[1][0]
            minor_axis = ellipse[1][1]
            estimated_length = estimate_ellipse_length(
                major_axis, minor_axis, PIXEL_TO_MM
            )

    return modified_image, major_axis, minor_axis, estimated_length


def predict_mask_tflite(image):
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def save_plot_as_image_ac(image_with_ellipse, binary_mask):
    plt.figure(figsize=(30, 10))

    # REMOVE extra white space
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05
    )

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Image with Ellipse", fontsize=25, pad=15)

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")
    plt.title("Predicted Mask", fontsize=25, pad=15)

    plt.savefig(PLOT_IMAGE_PATH)
    plt.close()

    return PLOT_IMAGE_PATH


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/plot_image_ac")
def plot_image_ac():
    return send_file(PLOT_IMAGE_PATH, mimetype="image/png")


@app.route("/ac", methods=["POST"])
def test_ac():

    if "file" not in request.files:
        return render_template("result.html", error="No file provided")

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return render_template("result.html", error="No selected file")

    # Read image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        return render_template("result.html", error="Invalid image")

    # For display
    display_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Normalize + reshape for model
    input_image = gray_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))

    # Predict mask (FAST)
    predicted_mask = predict_mask_tflite(input_image)

    # Threshold mask
    binary_mask = (predicted_mask[0] > 0.5).astype(np.uint8) * 255

    # Draw ellipse & calculate values
    image_with_ellipse, major_axis, minor_axis, estimated_length = draw_ellipse(
        display_image, binary_mask
    )

    # Round values to 2 decimal places
    major_axis = round(major_axis, 2)
    minor_axis = round(minor_axis, 2)
    estimated_length = round(estimated_length, 2)


    plot_path_ac = save_plot_as_image_ac(image_with_ellipse, binary_mask)

    return render_template(
        "result.html",
        plot_image_path=plot_path_ac,
        major_axis=major_axis,
        minor_axis=minor_axis,
        estimated_length=estimated_length,
        plot_route='plot_image_ac',
        measurement_name='Abdominal Circumference (AC)'
    )


# --------------------------------------------------
# Route for Biparetal Diameter and Head Circumference (BPD & HC)
# --------------------------------------------------


BPD_MODEL_PATH = "D:/Alyan/Final-Year-Project/FYP/FYP-FINAL-Server-webapp/models/BPD_MODEL.tflite"
BPD_PLOT_IMAGE_PATH = "static/Ac_Bpd_result_plot_images/bpd_plot_image.png"


bpd_interpreter = tf.lite.Interpreter(model_path=BPD_MODEL_PATH)
bpd_interpreter.allocate_tensors()

bpd_input_details = bpd_interpreter.get_input_details()
bpd_output_details = bpd_interpreter.get_output_details()


def predict_mask_bpd(image):
    bpd_interpreter.set_tensor(bpd_input_details[0]['index'], image.astype(np.float32))
    bpd_interpreter.invoke()
    return bpd_interpreter.get_tensor(bpd_output_details[0]['index'])


def save_plot_as_image_bpd(image_with_ellipse, binary_mask, image_with_line):
    plt.figure(figsize=(30, 10))

    # REMOVE extra white space
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05
    )

    # Original Image with Ellipse
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image with Ellipse", fontsize=25, pad=15)

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.title("Predicted Mask", fontsize=25, pad=15)

    # Detected Ellipse with Line
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Ellipse with Line', fontsize=25, pad=15)

    plt.savefig(BPD_PLOT_IMAGE_PATH)
    plt.close()

    return BPD_PLOT_IMAGE_PATH



@app.route("/bpd_plot_image")
def plot_image_bpd():
    return send_file(BPD_PLOT_IMAGE_PATH, mimetype="image/png")


@app.route('/bpd', methods=['POST'])
def test_bpd():
    if 'file' not in request.files:
        return render_template('result.html', error='No file provided')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result.html', error='No selected file')

    # Read image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        return render_template('result.html', error='Invalid image')

    # For display
    display_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Normalize + reshape for TFLite model
    input_image = gray_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))

    # Predict mask (BPD model)
    predicted_mask = predict_mask_bpd(input_image)

    # Threshold mask
    binary_mask = (predicted_mask[0] > 0.5).astype(np.uint8) * 255

    # Draw ellipse & calculate measurements
    image_with_ellipse, major_axis, minor_axis, estimated_length = draw_ellipse(display_image, binary_mask)

    # Draw line along major axis
    image_with_line = image_with_ellipse.copy()
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = 0
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            center_x, center_y = map(int, center)
            major_axis_px, minor_axis_px = map(int, axes)
            top_point = (center_x, center_y - major_axis_px // 2)
            bottom_point = (center_x, center_y + major_axis_px // 2)
            cv2.line(image_with_line, top_point, bottom_point, (255, 0, 0), 2)
            length = np.sqrt((bottom_point[0] - top_point[0])**2 + (bottom_point[1] - top_point[1])**2) * 0.3169

    # Round values to 2 decimal places
    major_axis = round(major_axis, 2)
    minor_axis = round(minor_axis, 2)
    estimated_length = round(estimated_length, 2)
    length = round(length,2)

    # Save plot image
    plot_path_bpd = save_plot_as_image_bpd(image_with_ellipse, binary_mask, image_with_line)

    # Render result.html
    return render_template(
        'result.html',
        plot_image_path=plot_path_bpd,
        major_axis=major_axis,
        minor_axis=minor_axis,
        estimated_length=estimated_length,
        length=length,
        plot_route='plot_image_bpd',
        measurement_name='Biparietal Diameter (BPD) and Head Circumference (HC)'
    )


#FEMUR Starts Here!

# Generic function for saving plots
def save_plot_image(fig, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    fig.savefig(file_path)
    plt.close(fig)
    return file_path


FEMUR_MODEL_PATH = "D:/Alyan/Final-Year-Project/FYP/FYP-FINAL-Server-webapp/models/VLE6_MODEL.tflite"
FEMUR_PLOT_DIR = "static/Femur_result_plot_images"

# Load TFLite model
femur_interpreter = tf.lite.Interpreter(model_path=FEMUR_MODEL_PATH)
femur_interpreter.allocate_tensors()
femur_input_details = femur_interpreter.get_input_details()
femur_output_details = femur_interpreter.get_output_details()


# --------------------------------------------------
# Route for VOLUSON E6 Machine
# --------------------------------------------------
@app.route('/volusonE6', methods=['POST'])
def test_voluson_e6_tflite():
    if 'file' not in request.files:
        return render_template('result_femur.html', error='No file provided')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result_femur.html', error='No selected file')

    # Read image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        return render_template('result_femur.html', error='Invalid image')

    # For display
    display_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)

    # Normalize + reshape for TFLite model
    input_image = gray_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))  # Add batch & channel dims

    # Predict mask using TFLite
    femur_interpreter.set_tensor(femur_input_details[0]['index'], input_image)
    femur_interpreter.invoke()
    predicted_mask = femur_interpreter.get_tensor(femur_output_details[0]['index'])

    # Threshold mask
    binary_mask = (predicted_mask[0] > 0.5).astype(np.uint8) * 255

    # Extract contours & measure length
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = 0
    for contour in contours:
        contour_length += cv2.arcLength(contour, True)

    pix_mm = contour_length * 0.08458333  # Pixel to mm conversion

    # Round the femur length to 2 decimal places
    pix_mm = round(pix_mm, 2)

    # Create images for plotting
    segmented_femur = cv2.bitwise_and(display_image, display_image, mask=binary_mask)
    contour_image = np.zeros_like(segmented_femur)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    # Plot all 3 images
    # Create figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # REMOVE extra white space
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05
    )

    # Plot images
    axs[0].imshow(display_image, cmap='gray')
    axs[0].set_title('Original Image',fontsize = 25,pad = 15)
    axs[0].axis('off')

    axs[1].imshow(segmented_femur, cmap='gray')
    axs[1].set_title('Segmented Femur',fontsize = 25,pad = 15)
    axs[1].axis('off')

    axs[2].imshow(contour_image, cmap='gray')
    axs[2].set_title('Contour Area',fontsize = 25,pad = 15)
    axs[2].axis('off')

    #code commented for length in 4th column of plot
   # axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f} mm', ha='center', va='center', fontsize=12)
    #axs[3].axis('off')

    # Save plot
    plot_image_path = save_plot_image(fig, FEMUR_PLOT_DIR, 'voluson_e6_plot.png')

    # Render result page
    return render_template(
        'result_femur.html',
        machine='Voluson E6',
        femur_length=pix_mm,
        image_path=os.path.join(FEMUR_PLOT_DIR, 'voluson_e6_plot.png'),
        plot_path=plot_image_path
    )



# --------------------------------------------------
# Route for VOLUSON S10 Machine
# --------------------------------------------------
FEMUR_S10_MODEL_PATH = "D:/Alyan/Final-Year-Project/FYP/FYP-FINAL-Server-webapp/models/VLS10_MODEL.tflite"
FEMUR_PLOT_DIR = "static/Femur_result_plot_images"

# Load TFLite model for Voluson S10
femur_s10_interpreter = tf.lite.Interpreter(model_path=FEMUR_S10_MODEL_PATH)
femur_s10_interpreter.allocate_tensors()
femur_s10_input_details = femur_s10_interpreter.get_input_details()
femur_s10_output_details = femur_s10_interpreter.get_output_details()


@app.route('/volusonS10', methods=['POST'])
def test_voluson_s10_tflite():
    if 'file' not in request.files:
        return render_template('result_femur.html', error='No file provided')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result_femur.html', error='No selected file')

    # Read image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        return render_template('result_femur.html', error='Invalid image')

    # For display
    display_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)

    # Normalize + reshape for TFLite model
    input_image = gray_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))  # Add batch & channel dims

    # Predict mask using TFLite
    femur_s10_interpreter.set_tensor(femur_s10_input_details[0]['index'], input_image)
    femur_s10_interpreter.invoke()
    predicted_mask = femur_s10_interpreter.get_tensor(femur_s10_output_details[0]['index'])

    # Threshold mask
    binary_mask = (predicted_mask[0] > 0.5).astype(np.uint8) * 255

    # Extract contours & measure length
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = 0
    for contour in contours:
        contour_length += cv2.arcLength(contour, True)

    pix_mm = contour_length * 0.08458333  # Pixel to mm conversion

    # Round the femur length to 2 decimal places
    pix_mm = round(pix_mm, 2)

    # Create images for plotting
    segmented_femur = cv2.bitwise_and(display_image, display_image, mask=binary_mask)
    contour_image = np.zeros_like(segmented_femur)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    # Plot all 3 images
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # REMOVE extra white space
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05
    )

    axs[0].imshow(display_image, cmap='gray')
    axs[0].set_title('Original Image', fontsize=25, pad=15)
    axs[0].axis('off')

    axs[1].imshow(segmented_femur, cmap='gray')
    axs[1].set_title('Segmented Femur', fontsize=25, pad=15)
    axs[1].axis('off')

    axs[2].imshow(contour_image, cmap='gray')
    axs[2].set_title('Contour Area', fontsize=25, pad=15)
    axs[2].axis('off')

    # code commented for length in 4th column of plot
    #axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f} mm', ha='center', va='center', fontsize=12)
    #axs[3].axis('off')


    # Save plot
    plot_image_path = save_plot_image(fig, FEMUR_PLOT_DIR, 'voluson_s10_plot.png')

    # Render result page
    return render_template(
        'result_femur.html',
        machine='Voluson S10',
        femur_length=pix_mm,
        image_path=os.path.join(FEMUR_PLOT_DIR, 'voluson_s10_plot.png'),
        plot_path=plot_image_path
    )


# --------------------------------------------------
# Route for VOLUSON S8 Machine
# --------------------------------------------------
FEMUR_S8_MODEL_PATH = "D:/Alyan/Final-Year-Project/FYP/FYP-FINAL-Server-webapp/models/VLS8_MODEL.tflite"
FEMUR_PLOT_DIR = "static/Femur_result_plot_images"

femur_s8_interpreter = tf.lite.Interpreter(model_path=FEMUR_S8_MODEL_PATH)
femur_s8_interpreter.allocate_tensors()
femur_s8_input_details = femur_s8_interpreter.get_input_details()
femur_s8_output_details = femur_s8_interpreter.get_output_details()


@app.route('/volusonS8', methods=['POST'])
def test_voluson_s8_tflite():

    if 'file' not in request.files:
        return render_template('result_femur.html', error='No file uploaded')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result_femur.html', error='No selected file')

    # Read image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        return render_template('result_femur.html', error='Invalid image')

    # For display
    display_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)

    # Preprocess for TFLite
    input_image = gray_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))  # (1, H, W, 1)

    # Run TFLite inference
    femur_s8_interpreter.set_tensor(
        femur_s8_input_details[0]['index'], input_image
    )
    femur_s8_interpreter.invoke()

    prediction = femur_s8_interpreter.get_tensor(
        femur_s8_output_details[0]['index']
    )

    # Threshold mask
    binary_mask = (prediction[0] > 0.5).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_length = sum(cv2.arcLength(cnt, True) for cnt in contours)

    # Pixel → mm conversion
    pix_mm = contour_length * 0.08458333

    # Round the femur length to 2 decimal places
    pix_mm = round(pix_mm, 2)

    # Visualization
    segmented_femur = cv2.bitwise_and(
        display_image, display_image, mask=binary_mask
    )

    contour_image = np.zeros_like(segmented_femur)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    # Plot all 3 images
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # REMOVE extra white space
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05
    )

    axs[0].imshow(display_image, cmap='gray')
    axs[0].set_title('Original Image', fontsize=25, pad=15)
    axs[0].axis('off')

    axs[1].imshow(segmented_femur, cmap='gray')
    axs[1].set_title('Segmented Femur', fontsize=25, pad=15)
    axs[1].axis('off')

    axs[2].imshow(contour_image, cmap='gray')
    axs[2].set_title('Contour Area', fontsize=25, pad=15)
    axs[2].axis('off')

    # code commented for length in 4th column of plot
    # axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f} mm', ha='center', va='center', fontsize=12)
    # axs[3].axis('off')

    # Save plot
    plot_image_path = save_plot_image(
        fig, FEMUR_PLOT_DIR, 'voluson_s8_plot.png'
    )

    # Render result
    return render_template(
        'result_femur.html',
        machine='VolusonS8',
        femur_length=pix_mm,
        image_path=f'{FEMUR_PLOT_DIR}/voluson_s8_plot.png',
        plot_path=plot_image_path
    )



# --------------------------------------------------
# Route for ALOKA Machine
# --------------------------------------------------
FEMUR_ALOKA_MODEL_PATH = "D:/Alyan/Final-Year-Project/FYP/FYP-FINAL-Server-webapp/models/ALOKA_MODEL.tflite"
FEMUR_PLOT_DIR = "static/Femur_result_plot_images"

aloka_interpreter = tf.lite.Interpreter(model_path=FEMUR_ALOKA_MODEL_PATH)
aloka_interpreter.allocate_tensors()
aloka_input_details = aloka_interpreter.get_input_details()
aloka_output_details = aloka_interpreter.get_output_details()


@app.route('/aloka', methods=['POST'])
def test_aloka_tflite():

    if 'file' not in request.files:
        return render_template('result_femur.html', error='No file uploaded')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result_femur.html', error='No selected file')

    # Read image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    gray_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        return render_template('result_femur.html', error='Invalid image')

    # For display
    display_image = cv2.cvtColor(gray_image.copy(), cv2.COLOR_GRAY2BGR)

    # Preprocess for TFLite
    input_image = gray_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))  # (1, H, W, 1)

    # Run TFLite inference
    aloka_interpreter.set_tensor(
        aloka_input_details[0]['index'], input_image
    )
    aloka_interpreter.invoke()

    prediction = aloka_interpreter.get_tensor(
        aloka_output_details[0]['index']
    )

    # Threshold mask
    binary_mask = (prediction[0] > 0.5).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_length = sum(cv2.arcLength(cnt, True) for cnt in contours)

    # Pixel → mm conversion (ALOKA specific)
    pix_mm = contour_length * 0.06858333

    # Round the femur length to 2 decimal places
    pix_mm = round(pix_mm, 2)

    # Visualization
    segmented_femur = cv2.bitwise_and(
        display_image, display_image, mask=binary_mask
    )

    contour_image = np.zeros_like(segmented_femur)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    # Plot results
    # Plot all 3 images
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # REMOVE extra white space
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05
    )

    axs[0].imshow(display_image, cmap='gray')
    axs[0].set_title('Original Image', fontsize=25, pad=15)
    axs[0].axis('off')

    axs[1].imshow(segmented_femur, cmap='gray')
    axs[1].set_title('Segmented Femur', fontsize=25, pad=15)
    axs[1].axis('off')

    axs[2].imshow(contour_image, cmap='gray')
    axs[2].set_title('Contour Area', fontsize=25, pad=15)
    axs[2].axis('off')

    # code commented for length in 4th column of plot
    # axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f} mm', ha='center', va='center', fontsize=12)
    # axs[3].axis('off')

    # Save plot
    plot_image_path = save_plot_image(
        fig, FEMUR_PLOT_DIR, 'aloka_plot.png'
    )

    # Render result
    return render_template(
        'result_femur.html',
        machine='ALOKA',
        femur_length=pix_mm,
        image_path=f'{FEMUR_PLOT_DIR}/aloka_plot.png',
        plot_path=plot_image_path
    )


@app.route('/femur', methods=['GET', 'POST'])
def femur_page():
    machine = request.args.get('machine', 'default_machine_value')
    return render_template('femur.html', machine=machine)


if __name__ == '__main__':
    app.run(debug=True)
