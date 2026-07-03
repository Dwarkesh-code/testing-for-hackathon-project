"""
Vision & Image Analyzer — identifies candidate screenshots/images in a repo
and uses an actual vision-capable model to select the best product/UI screenshot.

Previously this "vision" step never looked at pixels — it was a text-only model
reasoning over filenames and README captions, which could easily pick a broken
link or a logo mislabeled as a screenshot. It now sends the candidate images
themselves to a multimodal model (config.VISION_PICKER_MODEL) so the pick is
based on what's actually in the image.
"""

import re
import json
from typing import List, Dict, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import config


IMAGE_REGEX = re.compile(
    r'!\[([^\]]*)\]\(((?!http)[^\)]+\.(?:png|jpg|jpeg|gif|webp)|https?://[^\)]+\.(?:png|jpg|jpeg|gif|webp))\)',
    re.IGNORECASE
)


def extract_candidate_images(readme: str, file_tree: List[str], repo_url: str, default_branch: str = "main") -> List[Dict[str, str]]:
    """
    Finds image URLs or repo paths from README markdown and file tree.
    Returns list of dicts: [{'url': '...', 'caption': '...'}]
    Limited to 5 candidates because the vision model supports a maximum of 5 images per request.
    """
    candidates = []

    matches = IMAGE_REGEX.findall(readme or "")
    for caption, path_or_url in matches[:6]:
        url = path_or_url.strip()
        if not url.startswith("http"):
            clean_path = url.lstrip("./")
            raw_base = repo_url.replace("github.com", "raw.githubusercontent.com")
            url = f"{raw_base}/{default_branch}/{clean_path}"
        candidates.append({"url": url, "caption": caption or "Project Screenshot"})

    img_exts = (".png", ".jpg", ".jpeg", ".webp")
    for f in file_tree:
        f_lower = f.lower()
        if f_lower.endswith(img_exts) and any(folder in f_lower for folder in ("doc", "asset", "img", "screen", "preview", "demo")):
            raw_base = repo_url.replace("github.com", "raw.githubusercontent.com")
            url = f"{raw_base}/{default_branch}/{f}"
            if not any(c["url"] == url for c in candidates):
                candidates.append({"url": url, "caption": f"Asset: {f.split('/')[-1]}"})

    return candidates[:5]


def analyze_best_screenshot(
    repo_name: str,
    repo_desc: str,
    readme: str,
    file_tree: List[str],
    repo_url: str
) -> Optional[Dict[str, Any]]:
    """
    Evaluates candidate images by actually looking at their content and
    selects the single coolest UI/product screenshot.
    """
    candidates = extract_candidate_images(readme, file_tree, repo_url)
    if not candidates:
        return None

    if len(candidates) == 1:
        return {
            "url": candidates[0]["url"],
            "caption": candidates[0]["caption"],
            "score": 0.90
        }

    system_prompt = (
        "You are a UI/UX art director. You will be shown several candidate images from a "
        "GitHub repository, in order. Judge them by what is actually visible in each image "
        "— not by filename or caption text.\n"
        "Pick the single best image that shows a real, working UI or product demo.\n"
        "Reject logos, badges, shields.io banners, and architecture diagrams if a real "
        "interface screenshot exists among the candidates.\n\n"
        'Return ONLY JSON: {"best_index": 0, "caption": "short cool description", "score": 0.95}\n'
        "best_index is the 0-based index of the chosen image in the order shown."
    )

    content = [
        {"type": "text", "text": f"Repository: {repo_name} - {repo_desc}\n\nCandidate images below, in order:"}
    ]
    for c in candidates:
        content.append({"type": "image_url", "image_url": {"url": c["url"]}})

    try:
        groq_key = config.GROQ.get()
        if groq_key:
            llm = ChatGroq(model=config.VISION_PICKER_MODEL, api_key=groq_key, temperature=0.1)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content)
            ])
            text = response.content.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)
            idx = int(data.get("best_index", 0))
            if 0 <= idx < len(candidates):
                return {
                    "url": candidates[idx]["url"],
                    "caption": data.get("caption", "Project UI Preview"),
                    "score": float(data.get("score", 0.90))
                }
    except Exception as e:
        print(f"[Vision] Could not evaluate via vision model for {repo_name}: {e}")

    # Fallback: pick first candidate if the vision call fails or returns garbage
    return {
        "url": candidates[0]["url"],
        "caption": candidates[0]["caption"],
        "score": 0.85
    }
