import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from yt_dlp import YoutubeDL
import chromadb
import os
import whisper
import subprocess
from concurrent.futures import ThreadPoolExecutor

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "mistral:latest"
EMBED_MODEL     = "nomic-embed-text:latest"  
whisper_model=whisper.load_model("base")

chroma = chromadb.Client(
    chromadb.config.Settings(persist_directory="./chroma_db")
)
collection = chroma.create_collection("youtube_channel")

def ollama_chat(prompt: str) -> str:
    """Send a prompt to Ollama Mistral and return the response text."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def get_embedding(text: str) -> list:
    """Generate an embedding using Ollama's nomic-embed-text model."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_channel_video_ids(channel_url: str) -> list[str]:
    """Fetch all video IDs from a YouTube channel.

    yt-dlp can return a flat list OR a nested structure where the channel
    contains sub-playlists (Uploads, Shorts, Live…).  We walk both levels
    and collect every 'youtube' entry that has a real video id.
    """
    ydl_opts = {
        'extract_flat': 'in_playlist',   # flatten nested playlists too
        'quiet': True,
        'playlistend': 10,              # increase as needed
        'ignoreerrors': True,
    }

    def _collect(entries) -> list[str]:
        ids = []
        for entry in (entries or []):
            if entry is None:
                continue
            if entry.get('_type') == 'playlist':
                ids.extend(_collect(entry.get('entries', [])))
            else:
                vid_id = entry.get('id') or entry.get('url', '')
                if vid_id and len(vid_id) == 11 and '/' not in vid_id:
                    ids.append(vid_id)
        return ids

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        if info is None:
            raise ValueError(f"yt-dlp returned no data for URL: {channel_url}")
        
        entries = info.get('entries') or []
        video_ids = _collect(entries)

        if not video_ids:
            raise ValueError(
                "No video IDs found. Check the channel URL format.\n"
                "Try:  https://www.youtube.com/@ChannelName/videos"
            )

        # Deduplicate while preserving order
        seen = set()
        return [v for v in video_ids if not (v in seen or seen.add(v))]



def get_transcript(video_id: str):

    # Step 1: Try captions
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except:
        pass

    # Step 2: Try translate
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        for t in transcripts:
            try:
                translated = t.translate('en').fetch()
                return " ".join([x['text'] for x in translated])
            except:
                continue
    except:
        pass

    # Step 3: Whisper (FIXED)
    try:
        print(f"🎧 Using Whisper for: {video_id}")

        audio_file = f"{video_id}.mp3"

        subprocess.run([
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                f"https://www.youtube.com/watch?v={video_id}",
                "-o", f"{video_id}.%(ext)s"
            ])
        text = whisper_model.transcribe(audio_file)["text"]

        os.remove(audio_file)

        return text

    except Exception as e:
        print(f"❌ Whisper failed: {video_id} → {e}")
        return None

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split transcript into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def index_channel(channel_url: str):
    video_ids = get_channel_video_ids(channel_url)

    def process_video(video_id):
        transcript = get_transcript(video_id)
        if not transcript:
            return None

        chunks = chunk_text(transcript)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{video_id}_chunk_{i}"],
                metadatas=[{
                    "video_id": video_id,
                    "url": f"https://youtube.com/watch?v={video_id}"
                }]
            )
        return video_id

    with ThreadPoolExecutor(max_workers=5):  # 🔥 parallel
        results = list(filter(None, map(process_video, video_ids)))

    print(f" Indexed {len(results)} videos successfully")

def query_channel(question: str) -> dict:
    """Search across all indexed videos and generate an answer."""
    # Retrieve relevant chunks
    q_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=5
    )

    context = "\n\n".join(results["documents"][0])
    sources = [m["url"] for m in results["metadatas"][0]]

    prompt = f"""Answer the question using ONLY the YouTube transcript context below.
Include which video(s) the information came from.
if there is no innformation said no relevent data in document

Context:
{context}

Question: {question}
Answer:"""

    answer = ollama_chat(prompt)

    return {
        "answer": answer,
        "sources": list(set(sources))
    }
if __name__ == "__main__":
    # Index a channel - replace with any YouTube channel URL
    index_channel("https://www.youtube.com/@AndrejKarpathy/videos")

    result = query_channel("What does this creator say about transformers?")
    print(result["answer"])
    print("Sources:", result["sources"])