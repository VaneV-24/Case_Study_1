import gradio as gr
import spaces
from huggingface_hub import InferenceClient
from transformers import pipeline

LOCAL_MODEL = "Qwen/Qwen3-0.6B" #need to test different local models
REMOTE_MODEL = "openai/gpt-oss-20b" #need to test different remote models

pipe = pipeline(
    "text-generation",
    model=LOCAL_MODEL,
    dtype="auto",
    device="cpu", #changed to cpu (pre: cuda) as i am running locally only due to huggingface shared space issue
)

# CHANGED: Use darker brown for the page, linen for interactive areas, and enlarge the black time label.
fancy_css = """
body {
    background-color: #ead7bd !important;
}
.gradio-container {
    width: 96% !important;
    max-width: none !important;
    background-color: #ead7bd !important;
}
#app-title p {
    text-align: center;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin-bottom: 40px;
}
#app-subtitle p {
    text-align: center;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
    margin-bottom: 36px;
}
#time-required,
#time-required input,
#time-required .wrap {
    background-color: #faf0e6 !important;
}
#time-required label,
#time-required label span,
#time-required .label-wrap {
    color: #000000 !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}
#chat-container {
    width: 100%;
    background-color: #faf0e6 !important;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}
#chat-container .chatbot,
#chat-container .block,
#chat-container .accordion,
#chat-container .form,
#chat-container textarea,
#chat-container input,
#chat-container [role="listbox"] {
    background-color: #faf0e6 !important;
}
#model-note {
    font-size: 0.9em;
    color: var(--body-text-color-subdued);
    margin-top: 8px;
}
@media (max-width: 768px) {
    .gradio-container {
        width: 98% !important;
    }
    #chat-container {
        padding: 8px;
    }
}
"""


@spaces.GPU
def local_generate(
    messages,
    max_tokens,
    temperature,
    top_p,
):
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    return outputs[0]["generated_text"][-1]["content"]


# CHANGED: Accept the separately displayed time selection from ChatInterface.
def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    time_required,
    max_tokens,
    temperature,
    top_p,
    use_local_model,
    hf_token: gr.OAuthToken,
):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    # CHANGED: Include the selected cooking time in the prompt used by both models.
    messages.append(
        {
            "role": "user",
            "content": f"{message}\n\nTime Required: {time_required}",
        }
    )

    if use_local_model:
        print("[MODE] local")

        response = local_generate(
            messages,
            max_tokens,
            temperature,
            top_p,
        )

        yield response
        return

    print("[MODE] api")

    if hf_token is None or not getattr(hf_token, "token", None):
        yield "⚠️ Please log in with your Hugging Face account first."
        return

    client = InferenceClient(
        token=hf_token.token,
        model=REMOTE_MODEL,
    )

    response = ""

    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = chunk.choices
        token = ""

        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content

        response += token
        yield response

with gr.Blocks(css=fancy_css) as demo:
    with gr.Sidebar():
        gr.LoginButton()

    # CHANGED: Add a food emoji to the application title.
    gr.Markdown(
        "🍽️ What's For Dinner?",
        elem_id="app-title",
    )

    gr.Markdown(
        "Give the ingredients and time you have, and I'll suggest a recipe.",
        elem_id="app-subtitle",
    )

    # CHANGED: Render the clock-labeled time selector separately and identify it for tan styling.
    time_required = gr.Dropdown(
        choices=["20 minutes", "30 minutes", "1 hour", "2 hours"],
        value=None,
        label="🕒 Time Required",
        elem_id="time-required",
    )

    # CHANGED: Keep model controls unrendered until ChatInterface places them in its accordion.
    system_message = gr.Textbox(
        value="You are a recipe assistant Chatbot. Use the user's input ingredients and cooking time to suggest different recipes.",
        label="System message",
        render=False,
    )
    max_tokens = gr.Slider(
        minimum=1,
        maximum=2048,
        value=512,
        step=1,
        label="Max new tokens",
        render=False,
    )
    temperature = gr.Slider(
        minimum=0.1,
        maximum=2.0,
        value=0.7,
        step=0.1,
        label="Temperature",
        render=False,
    )
    top_p = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.95,
        step=0.05,
        label="Top-p (nucleus sampling)",
        render=False,
    )
    use_local_model = gr.Checkbox(
        label="Use Local Model",
        value=False,
        render=False,
    )

    with gr.Column(elem_id="chat-container"):
        # CHANGED: Build the chat in this Blocks context and wire in the external dropdown.
        chatbot = gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                system_message,
                time_required,
                max_tokens,
                temperature,
                top_p,
                use_local_model,
            ],
        )

        gr.Markdown(
            "Use **Additional inputs** to switch between the API model and the locally executed model.",
            elem_id="model-note",
        )


if __name__ == "__main__":
    demo.launch()
