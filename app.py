import gradio as gr
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
import tensorflow as tf

from huggingface_hub import from_pretrained_keras

model_v1 = from_pretrained_keras("cicada-project/cicada-v1.1")
model_v2 = from_pretrained_keras("cicada-project/cicada-v2.1")
hep.style.use("CMS")

def parse_input(et):
    if not et:
        raise gr.Error("Provide the input")

    et = [e.split(",") for e in et.split("\n")]
    et = np.array(et)
    et = et.astype(np.float32)

    if et.shape != (18, 14):
        raise gr.Error("The input shape has to be 18 rows and 14 columns")
    if np.any(et < 0) or np.any(et > 1023):
        raise gr.Error("The input has to be in a range (0, 1023)!")

    return et.reshape(1, 252)


def saliency(input_, version):
    x = tf.constant(input_)
    with tf.GradientTape() as tape:
        tape.watch(x)
        if version == "v1":
            predictions = model_v1(x)
        elif version == "v2":
            predictions = model_v2(x)
    gradient = tape.gradient(predictions, x)
    gradient = gradient.numpy()
    min_val, max_val = np.min(gradient), np.max(gradient)
    gradient = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())

    fig_s = plt.figure()
    im = plt.imshow(gradient.reshape(18, 14), vmin=0., vmax=1., cmap="Greys")
    ax = plt.gca()
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r"Calorimeter Saliency (a.u.)")
    plt.xticks(np.arange(14), labels=np.arange(4, 18))
    plt.yticks(
        np.arange(18),
        labels=np.arange(18)[::-1],
        rotation=90,
        va="center",
    )
    plt.xlabel(r"i$\eta$")
    plt.ylabel(r"i$\phi$")

    return fig_s


def draw_input(input_):
    fig_i = plt.figure()
    im = plt.imshow(input_.reshape(18, 14), vmin=0, vmax=input_.max(), cmap="Purples")
    ax = plt.gca()
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r"Calorimeter E$_T$ deposit (GeV)")
    plt.xticks(np.arange(14), labels=np.arange(4, 18))
    plt.yticks(
        np.arange(18),
        labels=np.arange(18)[::-1],
        rotation=90,
        va="center",
    )
    plt.xlabel(r"i$\eta$")
    plt.ylabel(r"i$\phi$")
    return fig_i


def inference(input_, version):
    if version == "v1":
        return model_v1.predict(input_)[0][0]
    elif version == "v2":
        return model_v2.predict(input_)[0][0]


def generate():
    matrix = np.clip(np.random.zipf(2, 252) - 1, 0, 1023)
    matrix = matrix.reshape(18, 14).astype(str)
    rows = [",".join(row) for row in matrix]
    return "\n".join(rows)


def process_request(input_):
    input_ = parse_input(input_)
    return (
        inference(input_, "v1"),
        inference(input_, "v2"),
        draw_input(input_),
        saliency(input_, "v1"),
        saliency(input_, "v2"),
    )


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_ = gr.Textbox(
                label="Calo Deposits",
                lines=18,
                placeholder="\n".join([",".join(["0"] * 14)] * 18),
            )
            with gr.Row():
                generate_input = gr.Button("Generate random input")
                magic = gr.Button("Do CICADA inference")

        with gr.Column():
            with gr.Row():
                label_v1 = gr.Number(label="CICADA Anomaly Score for CICADA v1")
            with gr.Row():
                label_v2 = gr.Number(label="CICADA Anomaly Score for CICADA v2")
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("Calorimeter Input"):
                        input_plot = gr.Plot()
                    with gr.TabItem("Saliency Map for CICADAv1"):
                        interpretation_plot_v1 = gr.Plot()
                    with gr.TabItem("Saliency Map for CICADAv2"):
                        interpretation_plot_v2 = gr.Plot()

    generate_input.click(generate, None, input_)
    magic.click(
        process_request,
        input_,
        [
            label_v1,
            label_v2,
            input_plot,
            interpretation_plot_v1,
            interpretation_plot_v2,
        ],
    )

demo.launch()
