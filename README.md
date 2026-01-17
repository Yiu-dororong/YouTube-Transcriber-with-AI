# Introduction

This project is for self use only, may not be suitable for scalable production environment. The goal is to extract transcripts from any YouTube videos, so that it can be more accessible to read the content.

Although there are already many tools or APIs to extract transcripts instantly, the painpoint is that they only works for videos with bulit-in captions. In other words, if the video does not come with bulit-in captions, those tools will fail to work.

*Disclaimer: This project is for educational and research purposes only.*

*Usage Policy:
This tool should only be used to download content that you own, or content that is explicitly licensed under Creative Commons or in the Public Domain.
The developer does not condone or encourage the downloading of copyrighted material in violation of YouTube's Terms of Service.
Users are solely responsible for any misuse of this software and for complying with their local copyright laws.*

*The developer of this project is not affiliated with YouTube or Google LLC. Use this software at your own risk.*

# Idea

To keep the input as easy as possible, the input would be a youtube video link instead of video ID. Next, it will extract some basic information, such as video title and duration. On one hand, this can make sure the video link corresponds to what the user wants, on the other hand, this allow backend operation to set pre-defined rules, such as disallowing transcribe excessively lengthy (say, 10 hours) videos.

Next, the main part is to transcribe. I still think the traditional API that download subtitles provided by youtube is useful, as it is fast and accurate. However, I leave a choice for the user to use AI to transcribe in case the subtitles are not satisfactory. 

Finally, there can be some post-processing, such as summarization or finding the insights. I consider the transcript as the original text, so I expect the user to read the final output.

<img width="609" height="692" alt="image" src="https://github.com/user-attachments/assets/3c322300-fcb7-4a8b-8cf5-4a4d9977a6bd" />

# Construction

## Preparations

To begin with, we need the following packages:

```
gradio==6.3.0
yt-dlp==2025.11.12
torch==2.9.1+cu126
transformers==4.49.0
pytubefix==10.3.6
mistralai==1.10.0
```

Please note that ffmpeg is also needed.

As mentioned, the first part would be extracting basic inforamtion using ```yt_dlp```. This would look like this:
```
import yt_dlp
ydl_opts = {
    'format': 'bestaudio/best',  
    'extract_audio': True,       
    'audioformat': 'webm',        # desired audio format (e.g., mp3, wav, aac)
    'outtmpl': 'temp_audio.%(ext)s',  # Save as local file
    #'outtmpl': f'./%(title)s.%(ext)s', # Standard naming
    'noplaylist': True,          # prevents downloading playlists
    'ffmpeg_location': "C:\\Program Files\\ffmpeg8\\bin", # Change to your ffmpeg location, though no need to use
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        final_filename = ydl.prepare_filename(info_dict)
        video_title = info_dict.get('title', None)
        thumbnail = info_dict.get('thumbnail', None)
        duration = info_dict.get('duration', None)
        #ydl.download([url])
```

This is the basic configuration with minimal changes, without setting the option to postprocess as other format. However, this can be a video downloader, it may be redundant to call it again. If the user feels comfortable without preview the video information to confirm, they can download it at the first place.


One more thing we can get right now is the subtitle availability: 
```
from pytubefix import YouTube
yt = YouTube(url)
yt.captions.keys()
```

Note: ```pytubefix``` and ```yt_dlp``` can perform the whole task in theory. You may choose one of them to complete. Besides, using YouTube Premium is a much safer way to download, and you may need to add a ```gr.File``` to ingest it and connect to the posterior process.

We can check what languages of subtitles dooes the video have, if it has. For example, one may have different versions of English, US, UK or even auto-generated.

## Transcript

After choosing the transcript languages, the transcript can be extracted by ```yt.captions[caption_language]```.

If you have a valid audio to transcribe, whisper model from openai maybe a good choice, as it support multiple languages (including Asian languages even for Cantonese) and lightweight with high accuracy, but its API is not free.

From their official examples, we can modified it to align with our task,

```
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe(final_filename, return_timestamps=True)
txt = str(result["text"])
```

This should complete the transcription workflow. Read more on their [documentation](https://huggingface.co/openai/whisper-large-v3) if you need extra help such as directly translate the transcript to other language.

## Post-process

Let's say I want a summary of it. We can pass the transcript to another AI to do it. Most LLMs can accpet multiple languages content, so I did not ask whisper to translate. With a simple prompt template, this will go pretty well without any extra explict instrcutions.

```
from mistralai import Mistral
import os

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as mistral:

    res = mistral.chat.complete(model="mistral-small-latest", messages=[
        {
            "content": Here is a transcript of a video: \
                        {}\
                    Give a summary of it within 400 words without telling me about this constraint.".format(txt),
            "role": "user",
        },
    ], stream=False)

    return res.choices[0].message.content
```

Here I used a [API](https://docs.mistral.ai/api) from mistralai which is pretty fast, you can use other LLM or even run on local as if the example of whisper shown above via huggingface. Yet, this API may not be the best choice as it may add context out of the transcript.

## Improvement

I assumed the transcript (original text) is not the focus, so it is just a huge chunk of text and hard to read. It may require some formatting. Also, AI transcribe is not prefect (whisper constantly make mistakes on Japanese songs claiming it is created by Hatsune Miku, and it cannot a single audio with multiple languages.), for gradio interface, please set it to be ```interactive=True``` to allow modification (human-in-the-loop).

Indeed, there is an all-in-one-model to complete this task, Voxtral-Mini-3B from mistralai. It can also auto-detect audio language and perform Q&A and summarization. However, it can only process up to 30 minutes for transcription and support limited languages (mostly European languages, but nearly no Asian languages). For details, please visit their [page](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).

Extend to other platforms or format can be a future direction to have a wider coverage.
