import pyttsx3
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', engine.getProperty('rate') - 30)
    engine.setProperty('volume', 1)
    engine.setProperty('voice', engine.getProperty('voices')[1].id)
    engine.say(text)
    engine.runAndWait()

speak("Hi, this is a test speak.")
speak("Hi, this is a test speak again.")  # ayrı ayrı engine başlatılıyor