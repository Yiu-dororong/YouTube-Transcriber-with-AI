import yt_dlp
import gradio as gr
import os
from torch import cuda, float16, float32
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytubefix import YouTube
from mistralai import Mistral
# Replace with the actual path to your ffmpeg/bin directory
os.add_dll_directory("C:\\Program Files\\ffmpeg8\\bin")

ydl_opts = {
    'format': 'bestaudio/best',  # selects the best audio format
    'extract_audio': True,       # ensures only audio is extracted (though 'format' handles this)
    'audioformat': 'webm',        # desired audio format (e.g., mp3, wav, aac)
    'outtmpl': 'temp_audio.%(ext)s',  
    #'outtmpl': f'./%(title)s.%(ext)s', # Standard naming
    'noplaylist': True,          # prevents downloading playlists
    'ffmpeg_location': "C:\\Program Files\\ffmpeg8\\bin",
    'no_check_certificate': True,
}

def download_youtube_audio_yt_dlp(url, output_path='.'):

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        
        info_dict = ydl.extract_info(url, download=False)
        final_filename = ydl.prepare_filename(info_dict)
        video_title = info_dict.get('title', None)
        thumbnail = info_dict.get('thumbnail', None)
        duration = info_dict.get('duration', None)
        #ydl.download([url])
    
    describtion = f" Video Title: {video_title} \n Duration: {duration} seconds"
    yt = YouTube(url)
    caption_langs = list(yt.captions.keys())
    if not caption_langs:
        default_lang = ""
        describtion = describtion + "\n No subtitles provided by YouTube"
    else:
        default_lang = caption_langs[0]

    return final_filename , describtion, thumbnail, url, gr.Dropdown(choices=caption_langs, label="Select subtitle language provided by YouTube ",value=default_lang, allow_custom_value=True)

device = "cuda:0" if cuda.is_available() else "cpu"

torch_dtype = float16 if cuda.is_available() else float32

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

def remove_timestamps_from_srt(srt_content):
    lines = srt_content.strip().split('\n')
    text_only_lines = []
    # Keep track of whether the current line is actual caption text
    is_text_line = False

    for line in lines:
        # Lines containing '-->' are timestamps
        if '-->' in line:
            is_text_line = False
        # Empty lines separate caption blocks
        elif line.strip() == '':
            is_text_line = False
        # Numeric lines are sequence numbers
        elif line.strip().isdigit():
            is_text_line = False
        # Otherwise, it's likely a text line
        else:
            is_text_line = True

        if is_text_line:
            text_only_lines.append(line.strip())

    # Join the text lines into a single block of text
    return ' '.join(text_only_lines)



def transcribe(url, force_AI, choice, filename):
    if os.path.exists(filename):
        os.remove(filename)
    
    if force_AI or choice == "":
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        result = pipe(filename, return_timestamps=True)
        txt = str(result["text"])
        os.remove(filename)
    else:
        yt = YouTube(url)

        choice = choice.strip('<>').split(sep="code=")
        choice = choice[-1].strip('"') # Extract the language code

        caption = yt.captions[choice]
        srt_string = caption.generate_srt_captions()
        txt = remove_timestamps_from_srt(srt_string)
    
    return txt

def summarize(title, text):
    with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
    ) as mistral:

        res = mistral.chat.complete(model="mistral-small-latest", messages=[
            {
                "content": "This is a youtube video that its {}. \
                    Here is the transcript of it: \
                        {}\
                    Give a summary of it within 400 words without telling me about this constraint.".format(title, text),
                "role": "user",
            },
        ], stream=False)

        return res.choices[0].message.content


with gr.Blocks() as demo:
    gr.Markdown("## YouTube transcribe")
    
    with gr.Row():
        with gr.Column():
            inputss=[gr.Text(label="Video Link",placeholder="Input YouTube Video Link")]
            button1 = gr.Button("Get info")
            thumbnail = gr.Image(interactive=False)
            
            final_filename =gr.State()
            input_url =gr.State()
            
            describtion = gr.Text(label="Video Information", interactive=False, lines=3)

                
            force_AI = gr.Checkbox(label="Check to use AI transcribe anyway. \n Use youtube built-in subtitle by default, if exists")
            caption_lang = gr.Dropdown(choices=[], label="Select subtitle language provided by YouTube ", allow_custom_value=True)
                
            button2 = gr.Button("Confirm and Transcribe")

        with gr.Column():

            trans = gr.Text(label="Transcript", lines=10, max_lines=10, interactive=True)

            button3 = gr.Button("Summarize")

            summ = gr.Text(label="Summary", lines=25, max_lines=25)
        
        # Link Function to its components
        button1.click(fn=download_youtube_audio_yt_dlp, 
                        inputs=inputss, 
                        outputs=[final_filename, describtion, thumbnail, input_url, caption_lang])
        
        button2.click(fn=transcribe,
                        inputs=[input_url, force_AI, caption_lang, final_filename] ,
                        outputs=trans)
        
        button3.click(fn=summarize,
                        inputs=[input_url, describtion] ,
                        outputs=summ)

if __name__ == "__main__":
    demo.launch(max_threads=20) # Increase max_threads to reach maximum performance