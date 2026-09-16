#!/usr/bin/env python3
"""
YouTube Scraper & Map Pipeline for FoodLoversTV
------------------------------------------------
Automates the full pipeline across:
1. YouTube video & transcript extraction (incremental)
2. Google Maps URL resolution & coordinate geocoding (incremental)
3. Gemini AI restaurant entity extraction & classification (incremental)
4. Interactive Folium map generation & downstream dataset exports
"""

import os
import sys
import re
import ast
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ----------------------------------------------------------------------
# Helper Functions & Key Loaders
# ----------------------------------------------------------------------

def load_api_key(env_var: str, file_path: str) -> Optional[str]:
    """Retrieve API key from environment variable or local file."""
    if os.environ.get(env_var):
        return os.environ[env_var].strip()
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
    return None


def extract_links(text: str) -> List[str]:
    """Extract all HTTP/HTTPS links from text."""
    if not isinstance(text, str):
        return []
    url_pattern = re.compile(r"(https?://\S+)")
    return url_pattern.findall(text)


def filter_gmaps_links(links: List[str]) -> List[str]:
    """Filter links to identify Google Maps URLs."""
    gmaps_pattern = re.compile(
        r"(https?://(?:goo\.gl/maps|maps\.google|google\.com/maps|maps\.app\.goo\.gl|g\.page)\S+)"
    )
    return [link.rstrip(".,;)\"'>") for link in links if gmaps_pattern.match(link)]


def map_to_veg_nonveg(category: Any) -> str:
    """Normalize raw category output into 'Veg' or 'Non-veg'."""
    category_str = str(category)
    non_veg_indicators = ['Non-veg', "['Non-veg', 'Veg']", "['Non-veg']", "['Veg', 'Non-veg']"]
    veg_indicators = ['Veg', 'Vegetarian', "['Veg']"]

    if any(ind in category_str for ind in non_veg_indicators):
        return 'Non-veg'
    elif any(ind in category_str for ind in veg_indicators):
        return 'Veg'
    return 'Non-veg'


# ----------------------------------------------------------------------
# Stage 1: YouTube Video & Transcript Scraper
# ----------------------------------------------------------------------

class YouTubeScraper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        channel_id: str = "UC-Lq6oBPTgTXT_K-ylWL6hg",
        output_csv: str = "youtube_videos.csv",
        no_gmaps_csv: str = "no_gmaps_links.csv",
    ):
        self.api_key = api_key or load_api_key("YOUTUBE_API_KEY", "youtube_api_key.txt")
        self.channel_id = channel_id
        self.output_csv = output_csv
        self.no_gmaps_csv = no_gmaps_csv

    def _get_youtube_service(self):
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. Please install with: "
                "pip install google-api-python-client"
            )
        if not self.api_key:
            raise ValueError(
                "YouTube API key not found. Set YOUTUBE_API_KEY environment variable "
                "or provide youtube_api_key.txt."
            )
        return build("youtube", "v3", developerKey=self.api_key)

    def extract_transcript(self, video_id: str) -> str:
        """Fetch English or translated transcript for a video."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
        except ImportError:
            logger.warning("youtube-transcript-api not installed; skipping transcript extraction.")
            return "Transcript API not installed"

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # 1. Try manual English
            try:
                transcript = transcript_list.find_transcript(['en'])
            except NoTranscriptFound:
                # 2. Try auto-generated English
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except NoTranscriptFound:
                    # 3. Try manual Hindi or Kannada
                    try:
                        transcript = transcript_list.find_transcript(['hi', 'kn'])
                    except NoTranscriptFound:
                        return "No transcript available"

            transcript_pieces = transcript.fetch()
            full_transcript = " ".join(t.get("text", "") for t in transcript_pieces).strip()

            if transcript.language_code != 'en':
                try:
                    from googletrans import Translator
                    translator = Translator()
                    translated = translator.translate(full_transcript, src=transcript.language_code, dest='en')
                    return translated.text.strip()
                except Exception as te:
                    logger.debug(f"Translation failed for {video_id}: {te}")
                    return full_transcript

            return full_transcript
        except Exception as e:
            return f"No transcript available ({e})"

    def run(self, full_scan: bool = False, limit: Optional[int] = None) -> None:
        """Run incremental scrape of the YouTube channel."""
        import pandas as pd

        logger.info("=== Stage 1: YouTube Video & Transcript Scraping ===")

        # Load existing dataset if available
        existing_df = None
        existing_ids = set()
        if os.path.exists(self.output_csv):
            try:
                existing_df = pd.read_csv(self.output_csv)
                if "video_id" in existing_df.columns:
                    existing_ids = set(existing_df["video_id"].dropna().astype(str))
                logger.info(f"Loaded {len(existing_df)} existing videos ({len(existing_ids)} unique IDs).")
            except Exception as e:
                logger.warning(f"Could not load existing {self.output_csv}: {e}")

        youtube = self._get_youtube_service()

        # Find uploads playlist ID
        logger.info(f"Querying channel: {self.channel_id}")
        channel_resp = youtube.channels().list(part="contentDetails", id=self.channel_id).execute()
        items = channel_resp.get("items", [])
        if not items:
            raise ValueError(f"No channel found with ID: {self.channel_id}")
        uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        new_videos = []
        next_page_token = None
        stop_pagination = False

        logger.info("Paginating uploads playlist (newest first)...")
        while True:
            playlist_req = youtube.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            playlist_resp = playlist_req.execute()
            batch_items = playlist_resp.get("items", [])

            for item in batch_items:
                snippet = item.get("snippet", {})
                vid_id = snippet.get("resourceId", {}).get("videoId")
                if not vid_id:
                    continue

                # Incremental stop: If not a full scan and we encounter a video already present
                if not full_scan and vid_id in existing_ids:
                    logger.info(f"Encountered already cached video {vid_id}. Halting pagination.")
                    stop_pagination = True
                    break

                title = snippet.get("title", "")
                description = snippet.get("description", "")
                link = f"https://www.youtube.com/watch?v={vid_id}"

                links_found = extract_links(description)
                gmaps = filter_gmaps_links(links_found)

                new_videos.append({
                    "video_id": vid_id,
                    "Title": title,
                    "Link": link,
                    "Description": description,
                    "Links": links_found,
                    "gmaps_links": gmaps if gmaps else None,
                })

                if limit and len(new_videos) >= limit:
                    logger.info(f"Reached user-specified limit of {limit} new videos.")
                    stop_pagination = True
                    break

            if stop_pagination:
                break

            next_page_token = playlist_resp.get("nextPageToken")
            if not next_page_token:
                break

        logger.info(f"Found {len(new_videos)} new videos to process.")

        if new_videos:
            new_df = pd.DataFrame(new_videos)
            logger.info("Fetching transcripts for new videos...")
            new_df["transcript"] = new_df["video_id"].apply(self.extract_transcript)

            if existing_df is not None:
                combined_df = pd.concat([new_df, existing_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["video_id"], keep="first")
            else:
                combined_df = new_df
        else:
            combined_df = existing_df if existing_df is not None else pd.DataFrame()

        # Save to disk
        if not combined_df.empty:
            combined_df.to_csv(self.output_csv, index=False)
            logger.info(f"Saved {len(combined_df)} total videos to {self.output_csv}.")

            no_gmaps = combined_df[combined_df["gmaps_links"].isnull()]
            no_gmaps[["Description"]].to_csv(self.no_gmaps_csv, index=False)
            logger.info(f"Saved {len(no_gmaps)} videos without GMaps links to {self.no_gmaps_csv}.")
        else:
            logger.info("No video data to save.")


# ----------------------------------------------------------------------
# Stage 2: Geocoding & Coordinate Resolution
# ----------------------------------------------------------------------

class GmapsGeocoder:
    def __init__(
        self,
        videos_csv: str = "youtube_videos.csv",
        lat_long_csv: str = "extracted_lat_long.csv",
        output_json: str = "gmaps_to_link.json",
    ):
        self.videos_csv = videos_csv
        self.lat_long_csv = lat_long_csv
        self.output_json = output_json

    @staticmethod
    def extract_lat_long_from_url(url: str) -> Tuple[Optional[float], Optional[float]]:
        """Resolve short URL redirects and extract (lat, long) via multiple regex patterns."""
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        final_url = url
        try:
            # Follow redirects to get final destination URL
            resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            final_url = resp.url
        except Exception as e:
            logger.debug(f"Request failed for {url}: {e}")

        patterns = [
            r"@(-?\d+\.\d+),(-?\d+\.\d+)",
            r"%40(-?\d+\.\d+),(-?\d+\.\d+)",
            r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",
            r"\?q=(-?\d+\.\d+),(-?\d+\.\d+)",
            r"ll=(-?\d+\.\d+),(-?\d+\.\d+)",
        ]

        for pat in patterns:
            match = re.search(pat, final_url)
            if match:
                try:
                    return float(match.group(1)), float(match.group(2))
                except ValueError:
                    pass

        # Try original url as fallback
        for pat in patterns:
            match = re.search(pat, url)
            if match:
                try:
                    return float(match.group(1)), float(match.group(2))
                except ValueError:
                    pass

        return None, None

    def run(self, limit: Optional[int] = None, force: bool = False) -> None:
        """Incrementally resolve Google Maps links and build gmaps_to_link.json."""
        import pandas as pd

        logger.info("=== Stage 2: Geocoding & Coordinate Resolution ===")

        if not os.path.exists(self.videos_csv):
            logger.warning(f"File {self.videos_csv} not found. Run stage 'scrape' first.")
            return

        videos_df = pd.read_csv(self.videos_csv)

        # Parse gmaps_links column
        valid_rows = videos_df[videos_df["gmaps_links"].notnull()].copy()

        def parse_links(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    try:
                        return ast.literal_eval(val)
                    except Exception:
                        pass
                return [val]
            return []

        valid_rows["parsed_gmaps"] = valid_rows["gmaps_links"].apply(parse_links)

        # Flatten list of all gmaps links
        all_links = set()
        for links_list in valid_rows["parsed_gmaps"]:
            for lk in links_list:
                clean_lk = lk.strip().rstrip(".,;)\"'>")
                if clean_lk:
                    all_links.add(clean_lk)

        logger.info(f"Found {len(all_links)} total unique Google Maps links across video records.")

        # Load existing geocache
        cached_coords = {}
        if os.path.exists(self.lat_long_csv):
            try:
                lat_df = pd.read_csv(self.lat_long_csv)
                for _, row in lat_df.iterrows():
                    lk = str(row["link"]).strip()
                    lat = row.get("latitude")
                    lng = row.get("longitude")
                    if pd.notnull(lat) and pd.notnull(lng):
                        try:
                            cached_coords[lk] = (float(lat), float(lng))
                        except ValueError:
                            cached_coords[lk] = (None, None)
                    else:
                        cached_coords[lk] = (None, None)
                logger.info(f"Loaded {len(cached_coords)} links from cache ({sum(1 for c in cached_coords.values() if c != (None, None))} with valid coordinates).")
            except Exception as e:
                logger.warning(f"Could not load {self.lat_long_csv}: {e}")

        # Determine unresolved links (only brand new links, or also failed links if force=True)
        if force:
            unresolved = [l for l in all_links if l not in cached_coords or cached_coords[l] == (None, None)]
        else:
            unresolved = [l for l in all_links if l not in cached_coords]
        logger.info(f"Unresolved links requiring geocoding: {len(unresolved)}")

        resolved_new = 0
        if unresolved:
            to_process = unresolved[:limit] if limit else unresolved
            for idx, lk in enumerate(to_process, start=1):
                logger.info(f"[{idx}/{len(to_process)}] Resolving: {lk}")
                lat, lng = self.extract_lat_long_from_url(lk)
                if lat is not None and lng is not None:
                    cached_coords[lk] = (lat, lng)
                    resolved_new += 1
                else:
                    cached_coords[lk] = (None, None)
                    logger.debug(f"Could not extract coordinates for: {lk}")
                time.sleep(0.3)  # Gentle rate limiting

            # Save updated lat_long CSV
            export_rows = [
                {"link": lk, "latitude": coords[0], "longitude": coords[1]}
                for lk, coords in cached_coords.items()
            ]
            pd.DataFrame(export_rows).to_csv(self.lat_long_csv, index=False)
            logger.info(f"Updated {self.lat_long_csv} with {len(export_rows)} total links ({resolved_new} newly resolved).")

        # Load existing gmaps_to_link.json to preserve any existing enrichments
        gmaps_to_link = {}
        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    gmaps_to_link = json.load(f)
                logger.info(f"Loaded existing {self.output_json} with {len(gmaps_to_link)} locations.")
            except Exception as e:
                logger.warning(f"Failed to read {self.output_json}: {e}")

        # Build / merge inverted index
        for _, row in valid_rows.iterrows():
            title = row.get("Title", "")
            yt_link = row.get("Link", "")
            desc = row.get("Description", "")
            transcript = row.get("transcript", "")
            for gmap_url in row["parsed_gmaps"]:
                clean_url = gmap_url.strip().rstrip(".,;)\"'>")
                if not clean_url:
                    continue

                lat, lng = cached_coords.get(clean_url, (None, None))

                if clean_url not in gmaps_to_link:
                    gmaps_to_link[clean_url] = []

                # Check if this video entry is already present under this gmap_url
                existing_entry = None
                for item in gmaps_to_link[clean_url]:
                    if item.get("link") == yt_link:
                        existing_entry = item
                        break

                if existing_entry is not None:
                    # Update lat/long if previously missing
                    if (existing_entry.get("lat") is None or pd.isnull(existing_entry.get("lat"))) and lat is not None:
                        existing_entry["lat"] = lat
                        existing_entry["long"] = lng
                else:
                    gmaps_to_link[clean_url].append({
                        "title": title,
                        "link": yt_link,
                        "description": desc,
                        "transcription": transcript,
                        "lat": lat,
                        "long": lng,
                    })

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(gmaps_to_link, f, indent=4)
        logger.info(f"Saved updated inverted index ({len(gmaps_to_link)} keys) to {self.output_json}.")


# ----------------------------------------------------------------------
# Stage 3: Gemini AI Enrichment
# ----------------------------------------------------------------------

class GeminiEnricher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
        input_json: str = "gmaps_to_link.json",
        output_json: str = "gmaps_to_link_w_labels_3.json",
        sleep_time: float = 2.0,
    ):
        self.api_key = api_key or load_api_key("GEMINI_API_KEY", "gemini_api.txt")
        self.model_name = model_name
        self.input_json = input_json
        self.output_json = output_json
        self.sleep_time = sleep_time
        self._model = None

    def _generate(self, prompt: str) -> Optional[str]:
        """Generate content via google-generativeai SDK if available, or direct REST API."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or provide gemini_api.txt."
            )

        system_instruction = (
            "You are a food and restaurant metadata extraction specialist. "
            "Extract structured restaurant details from the provided video title, description, and transcript.\n"
            "Required JSON fields:\n"
            "- rest_name: string (Restaurant Name)\n"
            "- address: string (Physical address)\n"
            "- ph_no: string (Phone number if mentioned, else empty string)\n"
            "- cuisine: list of strings (Cuisine classifications, including Indian state/region)\n"
            "- price: string ($, $$, $$$, $$$$, or $$$$$)\n"
            "- v_n_ng: string (Strictly one of: 'Veg', 'Non-veg', 'Vegan')\n"
            "- best_dishes: list of objects with keys [name, summary, price]\n"
            "- summary: string (Highlight summary of the restaurant and review)\n"
            "Return a JSON array containing a single object with these keys."
        )

        # 1. Try google.generativeai SDK if installed
        try:
            import google.generativeai as genai
            if self._model is None:
                genai.configure(api_key=self.api_key)
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096,
                    "response_mime_type": "application/json",
                }
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
            resp = self._model.generate_content(prompt)
            return resp.text
        except ImportError:
            pass

        # 2. Fallback to direct REST API via urllib with retry backoff
        import urllib.request
        model_path = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }

        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                return None
            except urllib.error.HTTPError as he:
                if he.code in (503, 429) and attempt < 3:
                    sleep_delay = attempt * 2
                    logger.warning(f"Gemini API returned {he.code}. Retrying in {sleep_delay}s (attempt {attempt}/3)...")
                    time.sleep(sleep_delay)
                else:
                    raise
            except Exception as e:
                if attempt < 3:
                    time.sleep(attempt * 2)
                else:
                    raise
        return None

    @staticmethod
    def parse_gemini_json(response_text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON response from Gemini."""
        if not response_text:
            return None
        text = response_text.strip()
        # Remove Markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try bracket slicing
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass

        return None

    def run(self, limit: Optional[int] = None) -> None:
        """Run incremental entity enrichment using Gemini."""
        logger.info("=== Stage 3: Gemini AI Restaurant Entity Extraction ===")

        # Base source data
        if not os.path.exists(self.input_json) and not os.path.exists(self.output_json):
            logger.warning(f"Neither {self.input_json} nor {self.output_json} found.")
            return

        # Prefer loading from output_json if it already exists (contains previous enrichments)
        data = {}
        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded existing enriched dataset {self.output_json} ({len(data)} entries).")
            except Exception as e:
                logger.warning(f"Could not read {self.output_json}: {e}")

        # Merge in any new keys from input_json
        if os.path.exists(self.input_json):
            try:
                with open(self.input_json, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                new_keys = 0
                for k, v in raw_data.items():
                    if k not in data:
                        data[k] = v
                        new_keys += 1
                if new_keys > 0:
                    logger.info(f"Merged {new_keys} new location keys from {self.input_json}.")
            except Exception as e:
                logger.warning(f"Could not read {self.input_json}: {e}")

        # Identify items requiring enrichment (missing rest_name)
        pending_items = []
        for gmap_url, entries in data.items():
            for idx, entry in enumerate(entries):
                if not entry.get("rest_name"):
                    pending_items.append((gmap_url, idx, entry))

        total_pending = len(pending_items)
        logger.info(f"Pending un-enriched entries: {total_pending}")

        if total_pending == 0:
            logger.info("All entries are already enriched. Nothing to do.")
            return

        process_queue = pending_items[:limit] if limit else pending_items
        logger.info(f"Processing {len(process_queue)} entries with Gemini ({self.model_name})...")

        processed_count = 0
        checkpoint_interval = 5

        try:
            for count, (gmap_url, idx, entry) in enumerate(process_queue, start=1):
                title = entry.get("title", "")
                desc = entry.get("description", "")
                transcription = entry.get("transcription", "")

                logger.info(f"[{count}/{len(process_queue)}] Enriching: {title[:60]}...")

                prompt = f"Video Title: {title}\n\nVideo Description:\n{desc}\n\nTranscript:\n{transcription}"

                try:
                    response_text = self._generate(prompt)
                    parsed_info = self.parse_gemini_json(response_text) if response_text else None
                    if parsed_info and isinstance(parsed_info, dict):
                        # Merge extracted attributes into entry
                        for field in ["rest_name", "address", "ph_no", "cuisine", "price", "v_n_ng", "best_dishes", "summary"]:
                            if field in parsed_info:
                                entry[field] = parsed_info[field]
                        processed_count += 1
                    else:
                        logger.warning(f"Could not parse Gemini JSON response for {title[:40]}")
                except Exception as call_err:
                    logger.error(f"Gemini API error for {title[:40]}: {call_err}")

                # Save checkpoint periodically
                if count % checkpoint_interval == 0:
                    with open(self.output_json, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                    logger.info(f"Checkpoint saved to {self.output_json} ({count} processed).")

                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)

        finally:
            # Always save final state
            with open(self.output_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved enriched data ({processed_count} newly enriched) to {self.output_json}.")


# ----------------------------------------------------------------------
# Stage 4: Interactive Map Generation & Downstream Export
# ----------------------------------------------------------------------

class MapGenerator:
    def __init__(
        self,
        input_json: str = "gmaps_to_link_w_labels_3.json",
        website_map_html: str = "Website/youtube_map_4.html",
        root_map_html: str = "youtube_map.html",
        final_data_json: str = "final_data.json",
        app_list_csv: str = "app/list2.csv",
    ):
        self.input_json = input_json
        self.website_map_html = website_map_html
        self.root_map_html = root_map_html
        self.final_data_json = final_data_json
        self.app_list_csv = app_list_csv

    def run(self) -> None:
        """Render Folium map with layers and export bot datasets."""
        import pandas as pd
        try:
            import folium
            from folium.plugins import Geocoder, LocateControl
            has_folium = True
        except ImportError:
            has_folium = False
            logger.warning("folium is not installed. Map HTML generation skipped. (Install with: pip install folium)")

        logger.info("=== Stage 4: Folium Map Generation & Downstream Exports ===")

        if not os.path.exists(self.input_json):
            logger.warning(f"Input file {self.input_json} not found. Run stage 'enrich' first.")
            return

        with open(self.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Processing data from {len(data)} location keys...")

        m = None
        category_groups = {}
        if has_folium:
            # Initialize Folium Map centered in Southern India / Bangalore
            map_center = [12.9716, 77.5946]
            m = folium.Map(
                location=map_center,
                zoom_start=7,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
            )
            Geocoder(position="topright").add_to(m)
            LocateControl().add_to(m)

            # Feature groups for Veg and Non-veg
            category_groups = {
                "Veg": folium.FeatureGroup(name="Veg", show=True),
                "Non-veg": folium.FeatureGroup(name="Non-veg", show=True),
            }

        total_markers = 0
        final_by_name: Dict[str, List[Dict[str, Any]]] = {}
        text_chunks: List[str] = []

        for gmap_url, info_list in data.items():
            for info in info_list:
                lat = info.get("lat")
                lng = info.get("long")
                if lat is None or lng is None or pd.isnull(lat) or pd.isnull(lng):
                    continue

                rest_name = info.get("rest_name") or info.get("title") or "Restaurant"
                diet_type = map_to_veg_nonveg(info.get("v_n_ng"))
                yt_link = info.get("link", "")
                address = info.get("address", "")
                price = info.get("price", "")
                cuisine = info.get("cuisine", "")
                summary = info.get("summary", "")

                # Clean popup HTML
                cuisine_str = ", ".join(cuisine) if isinstance(cuisine, list) else str(cuisine)
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px; max-width: 300px;">
                    <h3 style="margin: 0 0 5px 0; color: #2c3e50;">{rest_name}</h3>
                    <span style="background-color: {'#27ae60' if diet_type == 'Veg' else '#c0392b'}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;">{diet_type}</span>
                    {f"<span style='margin-left: 8px; font-weight: bold;'>{price}</span>" if price else ""}
                    <hr style="margin: 6px 0;">
                    {f"<p style='margin: 4px 0; font-size: 12px;'><b>Cuisine:</b> {cuisine_str}</p>" if cuisine_str else ""}
                    {f"<p style='margin: 4px 0; font-size: 12px;'><b>Address:</b> {address}</p>" if address else ""}
                    <div style="margin-top: 8px;">
                        {f"<a href='{yt_link}' target='_blank' style='margin-right: 12px; color: #e74c3c; text-decoration: none; font-weight: bold;'>▶ YouTube Review</a>" if yt_link else ""}
                        <a href='{gmap_url}' target='_blank' style='color: #2980b9; text-decoration: none; font-weight: bold;'>📍 Google Maps</a>
                    </div>
                    {f"<p style='margin-top: 8px; font-size: 11px; color: #555; max-height: 80px; overflow-y: auto;'>{summary}</p>" if summary else ""}
                </div>
                """

                if has_folium and m is not None:
                    folium.Marker(
                        location=[lat, lng],
                        popup=folium.Popup(popup_html, max_width=320),
                        tooltip=f"{rest_name} ({diet_type})",
                        icon=folium.Icon(
                            color="green" if diet_type == "Veg" else "red",
                            icon="cutlery",
                            prefix="fa",
                        ),
                    ).add_to(category_groups[diet_type])
                    total_markers += 1

                # Group by restaurant name for final_data.json
                if rest_name not in final_by_name:
                    final_by_name[rest_name] = []
                final_by_name[rest_name].append({
                    "title": info.get("title"),
                    "link": yt_link,
                    "lat": lat,
                    "long": lng,
                    "address": address,
                    "ph_no": info.get("ph_no"),
                    "cuisine": info.get("cuisine"),
                    "price": price,
                    "v_n_ng": diet_type,
                    "best_dishes": info.get("best_dishes"),
                    "summary": summary,
                })

                # Prepare text representation for bot dataset
                chunk = (
                    f"Restaurant: {rest_name}. Type: {diet_type}. Price: {price}. "
                    f"Cuisine: {cuisine_str}. Address: {address}. Summary: {summary}. "
                    f"Video Review: {yt_link}"
                )
                text_chunks.append(chunk)

        # Add groups and layer controls
        if has_folium and m is not None:
            for group in category_groups.values():
                group.add_to(m)
            folium.LayerControl().add_to(m)

            # Save HTML maps
            os.makedirs(os.path.dirname(self.website_map_html), exist_ok=True)
            m.save(self.website_map_html)
            m.save(self.root_map_html)
            logger.info(f"Exported map with {total_markers} markers to {self.website_map_html} and {self.root_map_html}.")

        # Export final_data.json
        with open(self.final_data_json, "w", encoding="utf-8") as f:
            json.dump(final_by_name, f, indent=2)
        logger.info(f"Saved {len(final_by_name)} restaurants to {self.final_data_json}.")

        # Export app/list2.csv
        os.makedirs(os.path.dirname(self.app_list_csv), exist_ok=True)
        pd.DataFrame(text_chunks, columns=["document"]).to_csv(self.app_list_csv, index=False)
        logger.info(f"Saved {len(text_chunks)} text records for bot to {self.app_list_csv}.")


# ----------------------------------------------------------------------
# Main CLI Orchestration
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Incremental Pipeline for FoodLoversTV YouTube Review Map"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run the complete end-to-end pipeline incrementally.",
    )
    parser.add_argument(
        "--stage",
        choices=["scrape", "geocode", "enrich", "map"],
        help="Run an individual stage of the pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing / re-resolving even if previously attempted or cached.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Stage 1: Force full YouTube channel scan instead of stopping at first cached video.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to process in the current execution (useful for testing).",
    )
    parser.add_argument(
        "--channel-id",
        type=str,
        default="UC-Lq6oBPTgTXT_K-ylWL6hg",
        help="YouTube channel ID to scrape.",
    )
    parser.add_argument(
        "--youtube-key",
        type=str,
        default=None,
        help="YouTube Data API key (defaults to YOUTUBE_API_KEY env or youtube_api_key.txt).",
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        default=None,
        help="Google Gemini API key (defaults to GEMINI_API_KEY env or gemini_api.txt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.6-flash",
        help="Gemini model name (e.g., gemini-3.6-flash, gemini-3.8-flash, gemini-flash-latest).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between Gemini API calls (default: 2.0s).",
    )

    args = parser.parse_args()

    if not args.run_all and not args.stage:
        parser.print_help()
        sys.exit(0)

    start_time = time.time()
    logger.info("Starting FoodLoversTV Pipeline...")

    # Stage 1: Scrape
    if args.run_all or args.stage == "scrape":
        scraper = YouTubeScraper(api_key=args.youtube_key, channel_id=args.channel_id)
        scraper.run(full_scan=args.full_scan, limit=args.limit)

    # Stage 2: Geocode
    if args.run_all or args.stage == "geocode":
        geocoder = GmapsGeocoder()
        geocoder.run(limit=args.limit, force=args.force)

    # Stage 3: Gemini Enrichment
    if args.run_all or args.stage == "enrich":
        enricher = GeminiEnricher(
            api_key=args.gemini_key,
            model_name=args.model,
            sleep_time=args.sleep,
        )
        enricher.run(limit=args.limit)

    # Stage 4: Map & Downstream Export
    if args.run_all or args.stage == "map":
        mapper = MapGenerator()
        mapper.run()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline finished successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
