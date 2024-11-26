import gradio as gr
from transformers import pipeline
from PIL import Image

# Função para remover o background da imagem
def remove_background(image):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

    # Obter a máscara da imagem
    pillow_mask = pipe(image, return_mask=True)

    # Aplicar máscara na imagem original
    pillow_image = pipe(image)

    return pillow_image

# Criar uma interface Gradio
app = gr.Interface(
    fn=remove_background,
    inputs=gr.components.Image(type="pil"),
    outputs=gr.components.Image(type="pil", format="png"),  # Especificar saída como PNG
    title="Remoção de Background de Imagens",
    description="Envie uma imagem e veja o background sendo removido automaticamente. A imagem resultante será no formato PNG."
)

app.launch(share=True)