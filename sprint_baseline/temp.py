from naoqi import ALProxy

#tts = ALProxy("ALTextToSpeech", "192.168.1.67", 9559)
tts = ALProxy("MovementGraph", "192.168.1.67", 9559)

#tts.say("Hello, world!")
tts.Move (x, y, theta)
