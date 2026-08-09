# Leaf Insight

make me a page for a small website using this description and give me design options use this as api url https://triage-kiln-savage.ngrok-free.dev/docs exactly as it is  and add CORS for FastAPI :

This page serves as a simple, clear interface for demonstrating a plant leaf disease detection system powered by a neural network. Its purpose is to let you test the model in a direct way while also helping you understand what the model does and how it makes decisions.

The page focuses on one core action. You upload an image of a plant leaf. The system processes the image and returns a prediction. This includes the detected disease class and a confidence score. The result appears immediately in a clean format, without unnecessary elements.

The design stays minimal on purpose. Each element has a clear role:

The upload section allows you to select an image from your device

The preview shows the exact image sent to the model

The predict button triggers the analysis

The results section displays the output in readable form

The visualization area shows where the model focused on the leaf

This structure reduces confusion. You always see input, action, and output in one place.

The system works in a simple pipeline:

You upload an image

The image is sent to the backend API

The model preprocesses the image to match training conditions

A pretrained convolutional neural network analyzes visual features such as texture, color patterns, and shapes

The model outputs probabilities for each disease class using softmax

The highest probability becomes the predicted class

A Grad-CAM heatmap highlights regions that influenced the decision

This approach follows standard image classification practice using transfer learning and cross-entropy optimization .

The Grad-CAM visualization adds transparency. Instead of giving only a label, the system shows which parts of the leaf contributed to the prediction. This helps you verify whether the model focuses on actual disease patterns rather than background noise. This step is important for debugging and trust in the model .

The page is designed as a demo, not a full product. Its goal is to validate three things:

The model produces correct predictions

The API works reliably

The results are understandable to a user

A clean interface improves testing speed. You upload, observe, adjust, and repeat. This short loop helps you refine both the model and the system.

The overall purpose is clarity. You see what goes in, what comes out, and why the model made that decision. This makes the demo useful for development, presentation, and evaluation.

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://plant-health-nn.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/687f91b5-b074-44ea-9f31-f1ab23f2e2b5).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
