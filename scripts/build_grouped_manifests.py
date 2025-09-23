import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True, help=r"Folder with *_voice_pairs_avg.csv")
    args = ap.parse_args()

    root = Path(args.manifests)
    if not root.exists():
        raise SystemExit(f"Manifests folder not found: {root}")

    made_any = False
    for src in root.glob("*_voice_pairs_avg.csv"):
        df = pd.read_csv(src)

        # --- per-language: group key = label (kept in 'continent' column for the plotter) ---
        out_lang = src.with_name(src.stem.replace("_voice_pairs_avg","_per_language") + ".csv")
        df_lang = df.copy()
        df_lang["continent"] = df_lang["label"]
        df_lang = df_lang[["label","continent","active_json","passive_json"]]
        df_lang.to_csv(out_lang, index=False)
        print(f"Wrote {out_lang}")

        # --- by-voice-type: group key = voice_type (again stored in 'continent') ---
        out_type = src.with_name(src.stem.replace("_voice_pairs_avg","_by_voicetype") + ".csv")
        if "voice_type" not in df.columns:
            raise SystemExit(f"Missing 'voice_type' column in {src}")
        df_type = df.copy()
        df_type["continent"] = df_type["voice_type"]
        df_type = df_type[["label","continent","active_json","passive_json"]]
        df_type.to_csv(out_type, index=False)
        print(f"Wrote {out_type}")

        made_any = True

    if not made_any:
        print(f"No *_voice_pairs_avg.csv found in {root}")

if __name__ == "__main__":
    main()
