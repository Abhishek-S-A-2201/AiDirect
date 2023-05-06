import torch
from diffusers import StableDiffusionPipeline
import openai
import cv2
import glob
import os
from PIL import Image
import numpy as np
import speech_recognition as sr

openai.api_key = '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'


def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source, timeout=5)

    try:
        transcription = r.recognize_google(audio)
        return transcription
    except (sr.UnknownValueError, sr.RequestError):
        return None


def generate_story(prompt):
    print("Generating story...")
    try:
        story = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a director and a video content writer."},
                {"role": "user", "content": f"""I will give you a short story idea prompt. 
              Give me a description of the different scenes for a 10 second video. 
              Seperate the scenes with a single newline character.
              Give me the scenes in the following format- 'Scene x: scene description'
              Prompt: {prompt}"""},
            ]
        )
    except:
        generate_story(prompt)
    story = story.choices[0].message.content
    story = story.split("\n")

    def clean(prompt):
        try:
            return prompt.split(":")[1].strip()
        except IndexError:
            return None

    story = list(map(clean, story))
    try:
        os.makedirs(f"static/output/{prompt[:10].replace(' ', '_')}.../")
    except FileExistsError:
        pass
    with open(f"static/output/{prompt[:10].replace(' ', '_')}.../transcript.txt", "w") as f:
        for item in story:
            f.write("%s\n" % item)
    return story


def generate_scenes(story):
    print("Generating scene prompts...")

    def prompt_generation(prompt):
        scene = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a prompt engineer."},
                {"role": "user", "content": f"""You will help me write prompts for an ai art generator called Stable diffusion.
                    THE PROMPTS SHOULD BE LESS THAN 75 WORDS.
                    I will provide you with short content ideas and your job is to elaborate these into full, coherent prompts.
                    Prompts involve describing the content and style of images in concise accurate language. 
                    It is useful to be explicit and use references to popular culture, artists 
                    Your focus needs to be on nouns and adjectives.
                    Here is a formula for you to use:
                    (content insert nouns here),
                    (style: insert references to genres, artists and popular culture here),
                    (colours reference color styles and palettes here),
                    (composition: reference cameras, specific lenses, shot types and positional elements here).
                    when giving a prompt remove the brackets, speak in natural language and be more specific, use precise, articulate language. 
                    Do not add the field names such as content medium etc. .
                    Content: {prompt}"""},
            ]
        )
        return scene.choices[0].message.content.replace("\n", "")

    scenes = list(map(prompt_generation, story))
    return scenes

# Online
pre_trained_model = "runwayml/stable-diffusion-v1-5"

#Local
# pre_trained_model = "pre_trained/stable_diffusion_safety_checker"

pipe = StableDiffusionPipeline.from_pretrained(pre_trained_model)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
pipe = pipe.to(device)
pipe.enable_attention_slicing()
generator = torch.Generator(device).manual_seed(1024)


def generate_images(scenes, prompt):
    print("Generating images...")
    try:
        os.makedirs(f"static/output/{prompt[:10].replace(' ', '_')}.../images")
    except FileExistsError:
        pass
    for idx, scene in enumerate(scenes):
        image = pipe(scene, generator=generator).images[0]
        cv2.imwrite(f"static/output/{prompt[:10].replace(' ', '_')}.../images/image{idx+1}.jpg", np.array(image))


def add_subs(story, prompt):
    print("Adding subtitles...")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2

    # Define the color of the text (yellow in this case)
    color = (0, 255, 255)
    try:
        os.makedirs(f"static/output/{prompt[:10].replace(' ', '_')}.../images_with_subtitles")
    except FileExistsError:
        pass
    # Loop through all the images in the folder
    for img_idx, img_path in enumerate(glob.glob(f"static/output/{prompt[:10].replace(' ', '_')}.../images/*.jpg")):
        # Load the image
        img = cv2.imread(img_path)

        # Get the subtitle for the current image
        subtitle = story[img_idx]

        # Get the size of the text
        text_size = cv2.getTextSize(subtitle, font, font_scale, font_thickness)[0]

        # Calculate the position of the text at the bottom of the image
        text_x = int((img.shape[1] - text_size[0]) / 2)
        text_y = int(img.shape[0] - text_size[1])

        # Add the text to the image
        cv2.putText(img, subtitle, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        # Save the image with the subtitle added
        cv2.imwrite(f"static/output/{prompt[:10].replace(' ', '_')}.../images_with_subtitles/{os.path.basename(img_path)}", img)


def generate_video(prompt):
    print("Your video is ready...")

    image_directory = f"static/output/{prompt[:10].replace(' ', '_')}.../images_with_subtitles"

    # List all the image files in the directory
    image_names = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.jpg')]

    # Open each image and append to a list
    images = []
    for name in sorted(image_names):
        images.append(Image.open(name))

    # Save the list of images as a GIF
    images[0].save(f"static/output/{prompt[:10].replace(' ', '_')}.../generation.gif", save_all=True, append_images=images[1:],
                   duration=5000, loop=0)

