import streamlit as st
import openai
import requests
from io import BytesIO
from PIL import Image

# Set OpenAI API Key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" 
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def generate_image(prompt, size="1024x1024"):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1  # Generate only 1 image
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def main():
    st.title("🖼️ Text-to-Image Generation ")
    st.markdown("Enter a prompt and generate AI images using OpenAI's DALL·E 3!")

    
    prompt = st.text_area("Enter your text prompt:")
    image_size = st.selectbox("Choose image size:", ["1024x1024", "1024x1792", "1792x1024"])

    if st.button("Generate Image"):
        if prompt.strip():
            with st.spinner("Generating image... 🚀"):
                image_url = generate_image(prompt, size=image_size)
                
                if image_url:
                    st.image(image_url, caption="Generated Image", use_column_width=True)

                    
                    img = download_image(image_url)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    byte_img = buf.getvalue()
                    st.download_button("Download Image", data=byte_img, file_name="generated_image.png", mime="image/png")
        else:
            st.warning("Please enter a prompt before generating an image.")

if __name__ == "__main__":
    main()
