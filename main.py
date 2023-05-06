from generation_utils import *


def main():
    choice = input("Enter you choice:\n - S to speak\n - T to type \n - Q to quit ")
    prompt = ""
    if choice.lower() == "s":
        prompt = recognize_speech()
        print(f"Prompt: {prompt}")
    elif choice.lower() == "t":
        prompt = input("Prompt: ")
    elif choice.lower() == "q":
        exit(0)
    story = generate_story(prompt)
    scenes = generate_scenes(story)
    generate_images(scenes, prompt)
    add_subs(story, prompt)
    generate_video(prompt)


if __name__ == "__main__":
    main()
