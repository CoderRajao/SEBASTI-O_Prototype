#pip install deepface opencv-python ollama tf-keras

from deepface import DeepFace
import os
import cv2
import ollama
from PIL import Image

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def picture(adress, name, m):
    cam = cv2.VideoCapture(0)
    while True:
        sucess, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (0, 255, 0), thickness=2)

        cv2.waitKey(1)
        cv2.imwrite(adress, frame)

        if m == 1:
            try:
                result = DeepFace.verify(img1_path=r"face1.jpg",img2_path=r"a.jpg")
                return result['verified']
            except Exception as e:
                print("error: ", e)

    cv2.waitKey(0)
    cam.release()
    cv2.destroyAllWindows()


# His {emotion}
def template(name, context, user, emotion, person):
    template = f"""
    You are Sebastião, a personal Ai of {name}.
    Respond naturaly to him.
    Is it {name} talking to you?: {person}
    His kinda {emotion}

    Context:
    {context}

    Input:
    {user}
    """

    return template


end = None
name = input("Your name: ")
context = ""
i = 0
while end != 'true':
    i = i + 1
    user = input(f"{name}: ")

    print("Processing Visual Info...")
    if os.path.isfile(r"face1.jpg") == False:
        picture(r"face1.jpg", name, m=0)
        cv2.imwrite('a.jpg', cv2.imread('face1.jpg', cv2.IMREAD_COLOR))
    else:
        person = picture(r"a.jpg", name, m=1)
        print(f"Is it {name}?: ", person)

        Emotion = DeepFace.analyze(img_path=r"a.jpg")
        emotion = Emotion[0]['dominant_emotion']
        print(f"dominant emotion: {Emotion[0]['dominant_emotion']}")
        image = cv2.imread('a.jpg')
        cv2.imshow('image window', image)
        cv2.waitKey(5)

        print("Sebastião is thinking...")
        client = ollama.Client()
        image = Image.open("a.jpg")
        response = client.generate(
            model='gemma3:12b',
            prompt=template(context, user, name, emotion, person),
            images=["a.jpg"]
        )
        print(response['response'])

        context += f"User:{user}\nSebastião:{response['response']}\n"

    if input("end?: ") == "true":
        print(context)



