import base64
import json
import re
from pathlib import Path

import requests
from nacl.public import PublicKey, SealedBox
import truststore


ROOT = Path(__file__).resolve().parent
truststore.inject_into_ssl()
token_text = (ROOT / "Supplied files" / "Access Token Git Hub.txt").read_text(encoding="utf-8-sig")
matches = re.findall(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", token_text)
token = matches[-1] if matches else [line.strip() for line in token_text.splitlines() if line.strip()][-1].split()[-1]
config = json.loads((ROOT / "Supplied files" / "confignewproject4.txt").read_text(encoding="utf-8-sig"))

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})

owner = session.get("https://api.github.com/user").json()["login"]
# Active GitHub repository used for the Aurora Stays 120-token deployment.
repo = "aurora-stay-qna-120-token-option"
base = f"https://api.github.com/repos/{owner}/{repo}/codespaces/secrets"
key_response = session.get(f"{base}/public-key")
key_response.raise_for_status()
key_data = key_response.json()
box = SealedBox(PublicKey(base64.b64decode(key_data["key"])))

for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
    value = config.get(name)
    if not value:
        continue
    encrypted = base64.b64encode(box.encrypt(value.encode("utf-8"))).decode("ascii")
    response = session.put(
        f"{base}/{name}",
        json={"encrypted_value": encrypted, "key_id": key_data["key_id"]},
    )
    response.raise_for_status()
    print(f"Set Codespaces secret: {name}")

print(f"Repository: https://github.com/{owner}/{repo}")
