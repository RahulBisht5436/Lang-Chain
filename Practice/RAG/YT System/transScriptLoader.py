from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi


api = YouTubeTranscriptApi()


def get_youtube_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    video_ids = query_params.get("v")

    if not video_ids:
        raise ValueError("Invalid YouTube URL: video ID not found")

    return video_ids[0]


def video_transcript_loader(video_link: str) -> str:
    video_id = get_youtube_video_id(video_link)

    try:
        # Get all transcripts available for the video
        transcript_list = api.list(video_id)

        # Print available transcripts
        print("Available transcripts:")

        for transcript in transcript_list:
            print(
                f"- language={transcript.language_code}, "
                f"name={transcript.language}, "
                f"generated={transcript.is_generated}"
            )

        # Pick the first available transcript
        transcript = next(iter(transcript_list))

        print(
            f"\nUsing transcript: "
            f"{transcript.language} ({transcript.language_code})"
        )

        transcript_data = transcript.fetch()

        text = " ".join(
            snippet.text
            for snippet in transcript_data
        )

        return text

    except Exception as e:
        raise RuntimeError(
            f"Unable to fetch transcript for video {video_id}: {e}"
        )


if __name__ == "__main__":

    text = video_transcript_loader(
        "https://www.youtube.com/watch?v=J5_-l7WIO_w&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=17"
    )

    print("\nTranscript:\n")
    print(text)