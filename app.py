import gradio as gr
import spaces
from huggingface_hub import InferenceClient
from transformers import pipeline

LOCAL_MODEL = "Qwen/Qwen3-0.6B"
REMOTE_MODEL = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=LOCAL_MODEL,
    dtype="auto",
    device="cuda",
)

fancy_css = """
.gradio-container {
    width: 96% !important;
    max-width: none !important;
    background: url("/gradio_api/file=cute_kitchen_background.png") center / cover fixed !important;
}
#app-title {
    text-align: center;
    margin-bottom: 4px;
}
#app-subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-bottom: 24px;
}
#chat-container {
    width: 100%;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
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


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    time_required,
    pantry_staples,
    max_tokens,
    temperature,
    top_p,
    use_local_model,
    hf_token: gr.OAuthToken,
):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    pantry_text = ", ".join(pantry_staples) if pantry_staples else "None selected"
    messages.append(
    {
        "role": "user",
        "content": (
            f"{message}\n\n"
            f"Time Required: {time_required}\n"
            f"Pantry Staples Available: {pantry_text}"
        ),
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

    gr.Markdown(
        "# 🍽️ What's For Dinner?",
        elem_id="app-title",
    )

    gr.Markdown(
        "**Don't know what to make for dinner?  Plan your meals with our chatbot. Select the time you have and input your ingredients or special requests.**",
        elem_id="app-subtitle",
    )

    time_required = gr.Dropdown(
        choices=["20 minutes", "30 minutes", "1 hour", "2 hours"],
        value=None,
        label="**🕒 How Much Time Do You Have?**",
        elem_id="time-required",
    )

    pantry_staples = gr.CheckboxGroup(
    choices=[
        "Salt",
        "Pepper",
        "Olive Oil",
        "Butter",
        "Garlic",
        "Onions",
        "Peppers",
        "Rice",
        "Pasta",
        "Eggs",
    ],
    label="Which Pantry Staples Do You Have?",
    elem_id="pantry-staples",
)
    system_message = gr.Textbox(
        value="You are a recipe assistant Chatbot. Use the user's input ingredients, checked pantry staples, and cooking time to suggest different recipes.",
        label="System message",
        render=False,
    )
    max_tokens = gr.Slider(
        minimum=500,
        maximum=2048,
        value=900,
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
        chatbot = gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                system_message,
                time_required,
                pantry_staples,
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
    demo.launch(allowed_paths=["cute_kitchen_background.png"])
