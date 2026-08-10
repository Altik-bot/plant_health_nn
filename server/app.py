
# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_resized = image.resize((224, 224))

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = predicted.item()

    # Prepare image for Grad-CAM
    rgb_image = np.array(image_resized).astype(np.float32) / 255.0

    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
"app.py" 103L, 2478B                                                                                                                                       70,0-1        77%
