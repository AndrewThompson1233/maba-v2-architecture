import json
import os
import re
from datasets import load_dataset


def prepare_dialog_dataset(output_path: str = "assets/dialogs_clean.json", max_samples: int = 5000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Loading lightweight English dialogue dataset (knkarthick/dialogsum)...")
    ds = load_dataset("knkarthick/dialogsum", split="train")

    formatted_data = []
    # Pattern to parse #Person1#: ... #Person2#: ...
    turn_pattern = re.compile(r"(#Person\d+#):\s*(.*?)(?=(?:#Person\d+#:)|$)", re.DOTALL)

    for row in ds:
        raw_text = row["dialogue"].strip()
        matches = turn_pattern.findall(raw_text)
        if not matches:
            continue

        dialog_turns = []
        for speaker, content in matches:
            content_clean = content.strip()
            if not content_clean:
                continue
            role = "user" if speaker == "#Person1#" else "assistant"
            dialog_turns.append({"role": role, "content": content_clean})

        if len(dialog_turns) >= 2:
            formatted_data.append({"dialog": dialog_turns})
            if len(formatted_data) >= max_samples:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(formatted_data)} cleaned dialogue conversations to {output_path}")


if __name__ == "__main__":
    prepare_dialog_dataset()
