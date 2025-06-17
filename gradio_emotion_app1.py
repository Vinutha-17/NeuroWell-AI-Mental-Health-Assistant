import random
import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Motivational quotes
quotes = [
    "You are stronger than you think.",
    "Every day is a fresh start.",
    "Keep going. Everything you need will come to you at the perfect time.",
    "You are capable of amazing things.",
    "Believe in yourself and all that you are.",
    "This too shall pass.",
    "Push yourself, because no one else is going to do it for you.",
    "Your mind is a powerful thing. When you fill it with positive thoughts, your life will start to change."
]

# Load the ONNX model
session = ort.InferenceSession("models/emotion_model.onnx")

# Emotion classes
emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Emotion prediction function
def predict_emotion(image):
    img_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})
    logits = outputs[0][0]

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    return {
        emotion_labels[i]: float(p)
        for i, p in enumerate(probs)
    }

# Chatbot logic with context tracking
def chat_interactive(message, history, context_state):
    message = message.strip().lower()
    history = history or []
    context_state = context_state or []

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    affirmatives = ["yes", "yeah", "yep", "sure"]
    negatives = ["no", "nah", "nope"]

    if context_state and context_state[-1] == "offer_tip":
        if any(yes in message for yes in affirmatives):
            reply = random.choice([
                "Try writing down 3 things you're grateful for today.",
                "Go for a 5-minute walk and take in your surroundings.",
                "Play your favorite song and let yourself relax for a moment.",
                "Take a deep breath and slowly count to 10. Repeat if needed."
            ])
            context_state.append("tip_given")
        elif any(no in message for no in negatives):
            reply = "That’s okay. I'm still here if you want to talk or need support."
            context_state.append("tip_declined")
        else:
            reply = "I didn’t quite catch that. Would you like a tip to lift your mood?"
            context_state.append("offer_tip_again")
    else:
        if message in greetings:
            reply = "Hello! I'm here to talk and listen. How are you feeling today?"
        elif "quote" in message or "motivate" in message:
            reply = random.choice(quotes)
        elif "sad" in message or "depress" in message:
            reply = "I'm really sorry you're feeling this way. You're not alone—I'm here for you. Would you like to try a simple mood-lifting tip?"
            context_state.append("offer_tip")
        elif "happy" in message:
            reply = "That's amazing! It's always good to acknowledge the happy moments in life 😊"
        elif "anxious" in message or "nervous" in message:
            reply = "It’s okay to feel anxious. Try closing your eyes and taking five deep breaths. Would you like a quick relaxation activity?"
            context_state.append("offer_tip")
        elif "lonely" in message:
            reply = "Feeling lonely can be tough. Talking helps, and I’m here for you. Maybe reconnecting with a friend might help too?"
        elif "bored" in message:
            reply = "Doing something creative or active can help. Maybe try drawing, journaling, or taking a walk!"
        elif "ok" in message or "okay" in message:
            reply = "Alright! Let me know if you'd like to talk more."
        elif "thank" in message:
            reply = "You're very welcome 😊 I'm always here if you need support."
        else:
            reply = "I'm listening. Feel free to share more. What’s on your mind?"

    history.append((message, reply))
    return history, "", context_state


# Gradio UI
with gr.Blocks(theme=gr.themes.Default(primary_hue="violet")) as app:
    gr.Markdown("## 🧠 NeuroWell - Emotion & Mental Wellness Assistant")
    gr.Markdown("Welcome! Detect your facial emotion and talk to our assistant to feel better.")

    with gr.Tab("📷 Emotion Detector"):
        with gr.Row():
            img_input = gr.Image(label="Upload Face Image", type="pil")
            detect_btn = gr.Button("Detect Emotion")
        emotion_result = gr.Label(label="Predicted Emotion", num_top_classes=7)
        detect_btn.click(fn=predict_emotion, inputs=img_input, outputs=emotion_result)

    with gr.Tab("💬 Mental Health Chatbot"):
        chatbot = gr.Chatbot(label="NeuroWell Assistant")
        user_input = gr.Textbox(placeholder="Share your thoughts here...", label="You", show_label=True)
        send_btn = gr.Button("Send")
        context_state = gr.State([])

        send_btn.click(
            chat_interactive,
            inputs=[user_input, chatbot, context_state],
            outputs=[chatbot, user_input, context_state]
        )
        user_input.submit(
            chat_interactive,
            inputs=[user_input, chatbot, context_state],
            outputs=[chatbot, user_input, context_state]
        )

# Launch the app
if __name__ == "__main__":
    app.launch()
