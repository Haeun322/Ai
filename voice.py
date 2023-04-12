import speech_recognition as sr
import pyttsx3

# 음성인식 엔진 초기화
r = sr.Recognizer()

# 음성합성 엔진 초기화
engine = pyttsx3.init()

# 음성합성 엔진 설정
engine.setProperty("rate", 150)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)

# 음성 입력
with sr.Microphone() as source:
    print("Speak now:")
    audio = r.listen(source)

# 음성 인식
try:
    text = r.recognize_google(audio, language="ko-KR")
    print("You said:", text)
except sr.UnknownValueError:
    print("Sorry, could not understand your speech.")
    text = "Sorry, could not understand your speech."
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    text = "Could not request results from Google Speech Recognition service."

# 음성합성
engine.say("You said: " + text)
engine.runAndWait()
