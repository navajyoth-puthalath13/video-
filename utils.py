import io
from google.cloud import videointelligence_v1 as videointelligence
from google.cloud import texttospeech
import openai

def analyze_frame(frame):
    client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.OBJECT_TRACKING]
    video_context = videointelligence.VideoContext()

    input_content = frame
    operation = client.annotate_video(
        input_content=input_content, 
        features=features, 
        video_context=video_context
    )
    result = operation.result(timeout=90)
    return result

def generate_narration(detection_result):
    objects = extract_objects(detection_result)
    prompt = f"Describe the scene with the following objects: {objects}"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50
    )

    narration = response.choices[0].text.strip()
    return narration

def extract_objects(detection_result):
    objects = []
    for annotation in detection_result.annotation_results[0].object_annotations:
        objects.append(annotation.entity.description)
    return ", ".join(objects)

def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=input_text, 
        voice=voice, 
        audio_config=audio_config
    )

    audio_file = "output.mp3"
    with open(audio_file, "wb") as out:
        out.write(response.audio_content)
    return audio_file
