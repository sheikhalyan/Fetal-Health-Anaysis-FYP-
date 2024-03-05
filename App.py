import os
from flask import Flask, render_template, request, send_file, jsonify
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask
from flask_cors import CORS
matplotlib.use('Agg')  # Use the Agg backend to avoid GUI-related issues
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})



def estimate_ellipse_length(major_axis, minor_axis):
    a = max(major_axis, minor_axis)
    b = min(major_axis, minor_axis)
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    return perimeter

# Function to draw ellipse on the image
def draw_ellipse(image, mask):
    modified_image = image.copy()

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate and draw ellipse on the image
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(modified_image, ellipse, (255, 255, 255), 3)  # Draw ellipse in white

            major_axis = ellipse[1][0]
            minor_axis = ellipse[1][1]
            estimated_length = estimate_ellipse_length(major_axis, minor_axis)

    return modified_image, major_axis, minor_axis, estimated_length

# Function to preprocess the predicted mask
def preprocess_mask(predicted_mask, threshold_value):
    _, binary_mask = cv2.threshold(predicted_mask, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_mask

# Function to predict mask using the loaded model
def predict_mask(image, model_path):
    model = keras.models.load_model(model_path)
    predicted_mask = model.predict(image)
    return predicted_mask


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/plot_image')
def plot_image():
    return send_file('static/plot_image.png', mimetype='image/png')


@app.route('/ac', methods=['POST'])
def test_ac():
    print('Received a request for AC prediction')  # Print a message when a request is received

    if 'file' not in request.files:
        print('No file provided')
        return render_template('result.html', error='No file provided')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        print('No selected file')
        return render_template('result.html', error='No selected file')

    print('File received:', uploaded_file.filename)

    # Read the image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Convert to BGR for display
    test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)

    # Normalize the image
    test_image = test_image.astype("float") / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    model_path = 'C:/Users/alyan/PycharmProjects/FYP-FINAL/models/AC_MODEL.h5'
    print('Using model:', model_path)

    # Predict the mask
    predicted_mask = predict_mask(test_image, model_path)
    print('Prediction done')

    # Check if the predicted_mask is not None before proceeding
    if predicted_mask is not None:
        # Preprocess the predicted mask
        binary_mask = preprocess_mask(predicted_mask[0], threshold_value=0.5)

        # Draw ellipse on the image and get measurements
        image_with_ellipse, major_axis, minor_axis, estimated_length = draw_ellipse(test_image_for_display, binary_mask)
        print('Ellipse drawn')

        # Save the plot as an image
        plot_image_path = save_plot_as_image_AC(image_with_ellipse, binary_mask)
        print('Plot image saved')

        # Return the data in JSON format
        return jsonify({
            'plot_image_path': plot_image_path,
            'image_path': 'static/plot_images/AC_plot.png',
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'estimated_length': estimated_length
        })

    else:
        print('Model prediction failed')
        return render_template('result.html', error='Model prediction failed')


# Function to save plot as image AC
def save_plot_as_image_AC(image_with_ellipse, binary_mask):
    # Increase the size of the entire figure
    plt.figure(figsize=(12, 6))

    # Original Image with Ellipse
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image with Ellipse")

    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.title("Predicted Mask")

    # Save the plot as an image
    plot_image_pathh = 'static/plot_images/AC_plot.png'
    plt.savefig(plot_image_pathh)
    plt.close()

    return plot_image_pathh






@app.route('/bpd', methods=['POST'])
def test_bpd():
    if 'file' not in request.files:
        return render_template('result.html', error='No file provided')

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return render_template('result.html', error='No selected file')

    # Read the image
    image_stream = uploaded_file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Convert to BGR for display
    test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)

    # Normalize the image
    test_image = test_image.astype("float") / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    model_path = 'C:/Users/alyan/PycharmProjects/FYP-FINAL/models/BDP_MODEL.h5'

    # Predict the mask
    predicted_mask = predict_mask(test_image, model_path)

    # Check if the predicted_mask is not None before proceeding
    if predicted_mask is not None:
        # Preprocess the predicted mask
        binary_mask = preprocess_mask(predicted_mask[0], threshold_value=0.5)

        # Draw ellipse on the image and get measurements
        image_with_ellipse, major_axis, minor_axis, estimated_length = draw_ellipse(test_image_for_display, binary_mask)

        # Continue with the rest of your code
        image_with_line = image_with_ellipse.copy()
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                center_x, center_y = map(int, center)

                major_axis, minor_axis = map(int, axes)

                # Calculate top and bottom points of the major axis
                top_point = (center_x, center_y - major_axis // 2)
                bottom_point = (center_x, center_y + major_axis // 2)

                # Draw a line from top to bottom on the ellipse
                cv2.line(image_with_line, top_point, bottom_point, (255, 0, 0), 2)

                # Calculate the length of the line
                length = np.sqrt((bottom_point[0] - top_point[0]) ** 2 + (bottom_point[1] - top_point[1]) ** 2)


        # Save the plot as an image
        plot_image_path = save_plot_as_image_BPD(image_with_ellipse, binary_mask, image_with_line)

        # Render result.html with plot and measurements
        return render_template('result.html', plot_image_path=plot_image_path,image_path='static/plot_images/BPD_plot.png', major_axis=major_axis, minor_axis=minor_axis, estimated_length=estimated_length, length = length )

    else:
        return render_template('result.html', error='Model prediction failed')

# Function to save plot as image BPD
def save_plot_as_image_BPD(image_with_ellipse, binary_mask, image_with_line):
    # Increase the size of the entire figure
    plt.figure(figsize=(24, 8))

    # Original Image with Ellipse
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image with Ellipse")

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.title("Predicted Mask")

    # Detected Ellipse with Line
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Ellipse with Line')

    # Save the plot as an image
    plot_image_path = 'static/plot_images/BPD_plot.png'
    plt.savefig(plot_image_path)
    plt.close()

    return plot_image_path

#FEMUR Starts Here!

# save plot as image (ALL-FEMUR)
def save_plot_image(fig, directory, filename):
    file_path = os.path.join(directory, filename)
    fig.savefig(file_path)
    plt.close(fig)  # Close the plot to release resources
    return file_path


#Voluson-E6

@app.route('/volusonE6', methods=['POST'])
def test_voluson_e6():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/alyan/PycharmProjects/FYP-FINAL/models/vle6_updated.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.2645833333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'voluson_e6_plot.png')

                    # Render the result page with the specific details
                    return render_template('result_femur.html', machine='test_volusonE6', femur_length=pix_mm, image_path='static/plot_images/voluson_e6_plot.png', plot_path=plot_image_path)

    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')


#Voluson-S10
@app.route('/volusonS10', methods=['POST'])
def test_voluson_S10():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/alyan/PycharmProjects/FYP-FINAL/models/VLS10.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.2645833333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'voluson_s10_plot.png')

                    # Render the result page with the specific details
                    # Inside the test_voluson_s10 route
                    machine_name = request.args.get('machine', default='test_volusonS10')
                    return render_template('result_femur.html', machine=machine_name, femur_length=pix_mm,image_path='static/plot_images/voluson_s10_plot.png',plot_path=plot_image_path)



    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')



#Voluson-S8

@app.route('/volusonS8', methods=['POST'])
def test_voluson_s8():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/alyan/PycharmProjects/FYP-FINAL/models/vls8.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.2645833333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'voluson_s8_plot.png')

                    # Render the result page with the specific details
                    # Inside the test_voluson_s8 route
                    machine_name = request.args.get('machine', default='test_volusonS8')
                    return render_template('result_femur.html', machine=machine_name, femur_length=pix_mm,image_path='static/plot_images/voluson_s8_plot.png',plot_path=plot_image_path)



    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')


#ALOKA

@app.route('/aloka', methods=['POST'])
def test_aloka():
    # Check if 'file' exists in request.files
    if 'file' in request.files:
        uploaded_file = request.files['file']

        # Check if the filename is not empty
        if uploaded_file.filename != '':
            # Read the uploaded file as an image
            image_stream = uploaded_file.read()
            nparr = np.frombuffer(image_stream, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is not None:
                # Preprocess the image
                test_image_for_display = cv2.cvtColor(test_image.copy(), cv2.COLOR_GRAY2BGR)
                test_image = test_image.astype("float") / 255.0
                test_image = np.expand_dims(test_image, axis=0)

                # Load the pre-trained model
                loaded_model = keras.models.load_model('C:/Users/alyan/PycharmProjects/FYP-FINAL/models/Aloka_model.h5')

                # Make predictions
                predictions = loaded_model.predict(test_image)

                # Threshold the predictions to get binary masks if needed
                threshold = 0.5  # You can experiment with different threshold values
                thresholded_predictions = (predictions > threshold).astype(np.uint8)

                # Visualize the segmented head region
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                # Measure and show the length of the segmented mask in the image
                for idx, (image, mask) in enumerate(zip(test_image, thresholded_predictions)):
                    # Extract the head region using the binary mask
                    segmented_head = cv2.bitwise_and(image, image, mask=mask)

                    # Find contours of the segmented mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Calculate the length of contours
                    contour_length = 0
                    for contour in contours:
                        contour_length += cv2.arcLength(contour, True)

                    # Draw contours on the segmented head image
                    contour_image = np.zeros_like(segmented_head)
                    cv2.drawContours(contour_image, contours, -1, (255), 1)

                    pix = 0.2645833333
                    pix_mm = contour_length * pix

                    # Display the original image, the segmented head, and the measured length
                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    axs[0].imshow(image, cmap='gray')
                    axs[0].set_title('Original Image')
                    axs[0].axis('off')

                    axs[1].imshow(segmented_head, cmap='gray')
                    axs[1].set_title('Segmented Femur')
                    axs[1].axis('off')

                    axs[2].imshow(contour_image, cmap='gray')
                    axs[2].set_title('Contour Area')
                    axs[2].axis('off')

                    axs[3].text(0.5, 0.5, f'Length: {pix_mm:.2f}mm', ha='center', va='center', fontsize=12)
                    axs[3].axis('off')

                    # Save the plot as an image
                    plot_image_path = save_plot_image(fig, 'static/plot_images', 'aloka_plot.png')

                    # Render the result page with the specific details

                    machine_name = request.args.get('machine', default='test_aloka')
                    return render_template('result_femur.html', machine=machine_name, femur_length=pix_mm,image_path='static/plot_images/aloka_plot.png',plot_path=plot_image_path)



    # If no valid file or image processing fails, return an error
    return render_template('result_femur.html', error='Invalid file or image processing failed')


@app.route('/Femur.html', methods=['GET', 'POST'])
def femur_page():
    machine = request.args.get('machine', 'default_machine_value')
    return render_template('Femur.html', machine=machine)


if __name__ == '__main__':
    app.run(debug=True)
