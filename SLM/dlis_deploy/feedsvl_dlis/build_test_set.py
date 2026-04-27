import json
import sys

all_title = {}
idx = 0
with open(sys.argv[1]) as f_read, open(sys.argv[2], "w") as f_write:
    for item in f_read:
        infos = item.strip().split("\t")
        if len(infos) >= 6:
            title = infos[4].strip()
            body = infos[5].strip()
            if len(title.strip().split()) < 3:
                continue
            if title in all_title:
                continue
            all_title[title] = 0
            f_write.write(json.dumps({"title": title, "body": body}, ensure_ascii=False) + "\n")
            idx += 1
            if idx >= 10000:
                break