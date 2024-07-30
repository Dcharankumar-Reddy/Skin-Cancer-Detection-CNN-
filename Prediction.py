# Predict on new images
def predict_image(image_path):
    image = load_and_resize_image(image_path, image_size)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    return le.inverse_transform([predicted_class])[0]

new_image_path = 'path_to_new_image.jpg'
predicted_class = predict_image(new_image_path)
print(f'Predicted class: {predicted_class}')
