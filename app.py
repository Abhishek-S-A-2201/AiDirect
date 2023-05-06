from flask import Flask, render_template, url_for, request
from generation_utils import *

app = Flask(__name__)
app.secret_key = 'super secret key'


@app.route('/')
def index():
    return render_template("index.html", error=None)


@app.route('/add_prompt', methods=['POST', 'GET'])
def add_prompt():
    prompt = ""
    story=""
    if request.method == "POST":
        prompt = request.form["prompt"]
        with open(f"static/output/{prompt[:10].replace(' ', '_')}.../transcript.txt") as f:
            story = f.readlines()
        # story= ["Scene1", "Scene2"]
        # story = generate_story(prompt)
        if story is None:
            return render_template("index.html", error="Server busy: try again after a while... Try again!")
        # scenes = generate_scenes(story)
        # generate_images(scenes, prompt)
        # add_subs(story, prompt)
        # generate_video(prompt)

    props = {
        "prompt": prompt,
        "story": story,
        "images": [i[7:] for i in list(glob.glob(f"static/output/{prompt[:10].replace(' ', '_')}.../images/*.jpg"))],
        "main_image": f"/output/{prompt[:10].replace(' ', '_')}.../generation.gif"
    }
    return render_template("output.html", props=props)


if __name__ == "__main__":
    app.run(debug=True)
