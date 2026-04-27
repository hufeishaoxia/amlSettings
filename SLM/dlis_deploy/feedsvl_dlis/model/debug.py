import requests
import json

def main():
    url = "http://localhost:8888"

    payload = {
        "Timestamp": "2025-10-02",
        "GemSnapshotId": "CEB576AE04-snapshot-4",
        "dsat_reason": "according to the policy guideline 2.1, the GEM thumbnail does not match the GEM title in relevance or context.",
        "label": 0,
        "gem_title": "Shubman Gill's captaincy sparks mixed expert opinions",
        "gem_summary": "Shubman Gill, recently appointed as India's ODI captain, has drawn both praise and constructive criticism from cricketing experts. While Gautam Gambhir lauded his leadership during challenging tours, Ian Bishop emphasised the need for time and guidance from veterans like Rohit Sharma and Virat Kohli to refine his captaincy. The discussions come ahead of India's ODI series against Australia, marking a pivotal moment in Gill's evolving leadership journey.",
        "image_title": "Guyana Amazon Warriors v Antigua & Barbuda Falcons - Men's 2025 Republic Bank Caribbean Premier League",
        "image_caption": "GEORGETOWN, GUYANA - SEPTEMBER 10: Commentator, Ian Bishop ahead of play during the Men's 2025 Republic Bank Caribbean Premier League match between Guyana Amazon Warriors v Antigua & Barbuda Falcons  at Providence Stadium on September 10, 2025 in Georgetown, Guyana. (Photo by Ashley Allen - CPL T20/CPL T20 via Getty Images)",
        "image": "https://media.gettyimages.com/id/2234748959/photo/georgetown-guyana-commentator-ian-bishop-ahead-of-play-during-the-mens-2025-republic-bank.jpg?b=1&s=612x612&w=0&k=20&c=c_oTqbuZy0yT_uwV1GGAB_BKL8E9rBvEymw6i-95U4E="
    }

    resp = requests.post(url, json=payload)

    print("Status Code:", resp.status_code)
    print("Response:")
    print(resp.text)

if __name__ == "__main__":
    main()
