import os
import time
import logging
import requests
import yt_dlp

logger = logging.getLogger("video-indexer")


class VideoIndexerService:
    def __init__(self):
        self.account_id = os.getenv("AZURE_VI_ACCOUNT_ID")
        self.location = os.getenv("AZURE_VI_LOCATION")
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.vi_name = os.getenv("AZURE_VI_NAME")

        # Service Principal credentials
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")

    def get_arm_token(self):
        """Gets ARM token using Service Principal credentials directly."""
        url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token"

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": "https://management.azure.com/"
        }

        response = requests.post(url, data=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to get ARM token: {response.text}")

        token = response.json().get("access_token")
        logger.info("✓ ARM token retrieved successfully")
        return token

    def get_vi_account_token(self, arm_token):
        """Exchanges ARM token for Video Indexer account token."""
        url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.VideoIndexer/accounts/{self.vi_name}"
            f"/generateAccessToken?api-version=2024-01-01"
        )

        headers = {"Authorization": f"Bearer {arm_token}"}
        payload = {"permissionType": "Contributor", "scope": "Account"}

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to get VI Account Token: {response.text}")

        token = response.json().get("accessToken")
        logger.info("✓ Video Indexer account token retrieved successfully")
        return token

    def get_access_token(self):
        """Full token chain: Service Principal -> ARM -> Video Indexer."""
        arm_token = self.get_arm_token()
        vi_token = self.get_vi_account_token(arm_token)
        return vi_token

    def download_youtube_video(self, url, output_path="temp_video.mp4"):
        """Downloads a YouTube video to a local file."""
        logger.info(f"Downloading YouTube video: {url}")

        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info("✓ Download complete.")
            return output_path
        except Exception as e:
            raise Exception(f"YouTube Download Failed: {str(e)}")

    def upload_video(self, video_path, video_name):
        """Uploads a local file to Azure Video Indexer."""
        vi_token = self.get_access_token()

        api_url = (
            f"https://api.videoindexer.ai/{self.location}"
            f"/Accounts/{self.account_id}/Videos"
        )

        params = {
            "accessToken": vi_token,
            "name": video_name,
            "privacy": "Private",
            "indexingPreset": "Default",
        }

        logger.info(f"Uploading {video_path} to Azure Video Indexer...")

        with open(video_path, 'rb') as video_file:
            files = {'file': video_file}
            response = requests.post(api_url, params=params, files=files)

        if response.status_code != 200:
            raise Exception(f"Azure Upload Failed: {response.text}")

        video_id = response.json().get("id")
        logger.info(f"✓ Upload successful. Azure Video ID: {video_id}")
        return video_id

    def wait_for_processing(self, video_id):
        """Polls Azure Video Indexer until processing is complete."""
        logger.info(f"Waiting for video {video_id} to process...")

        while True:
            vi_token = self.get_access_token()

            url = (
                f"https://api.videoindexer.ai/{self.location}"
                f"/Accounts/{self.account_id}/Videos/{video_id}/Index"
            )
            params = {"accessToken": vi_token}
            response = requests.get(url, params=params)
            data = response.json()

            state = data.get("state")
            logger.info(f"Processing status: {state}")

            if state == "Processed":
                logger.info("✓ Video processing complete!")
                return data
            elif state == "Failed":
                raise Exception("Video Indexing Failed in Azure.")
            elif state == "Quarantined":
                raise Exception("Video Quarantined (Copyright/Content Policy Violation).")

            logger.info("Waiting 30 seconds before next check...")
            time.sleep(30)

    def extract_data(self, vi_json):
        """Parses the Video Indexer JSON into our State format."""
        transcript_lines = []
        for v in vi_json.get("videos", []):
            for insight in v.get("insights", {}).get("transcript", []):
                transcript_lines.append(insight.get("text", ""))

        ocr_lines = []
        for v in vi_json.get("videos", []):
            for insight in v.get("insights", {}).get("ocr", []):
                ocr_lines.append(insight.get("text", ""))

        return {
            "transcript": " ".join(transcript_lines),
            "ocr_text": ocr_lines,
            "video_metadata": {
                "duration": vi_json.get("summarizedInsights", {}).get("duration", {}).get("seconds"),
                "platform": "youtube"
            }
        }