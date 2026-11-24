"""Build an explicitly automated narrated tour of actual code and saved results on macOS.

Requires ffmpeg, ffprobe and the optional local Kokoro narrator. It is a review aid, not a
claim that the student recorded or presented this material.
"""

from pathlib import Path
import json
import subprocess
import textwrap
from PIL import Image, ImageDraw, ImageFont
from generate_narration import Narrator

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "walkthrough/build"
FONT = "/System/Library/Fonts/Menlo.ttc"


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    narrator = Narrator(ROOT)
    runs = []
    for p in sorted((ROOT / "runs").glob("*/metrics.json")):
        if not (p.parent / "official_format_check.txt").exists():
            continue
        if (
            "Format is valid"
            not in (p.parent / "official_format_check.txt").read_text()
        ):
            continue
        m = json.loads(p.read_text())
        runs.append((p.parent.name, m))
    gpu = ROOT / 'experiments/gpu_extension_v1'
    gpu_runs = [(p.parent.name, json.loads(p.read_text()))
                for p in sorted((gpu / 'runs').glob('*/metrics.json'))
                if 'Format is valid' in (p.parent / 'official_format_check.txt').read_text()]
    protocol = json.loads((gpu / 'analysis/frozen_selection.json').read_text())
    primary = protocol['primary_final_run']
    best = next((rid,m) for rid,m in gpu_runs if rid == primary.removeprefix('FINAL_'))
    final_selected = next(m for rid,m in gpu_runs if rid == primary)
    final_sentence = (f"The prespecified hybrid primary scores {final_selected['mean_over_scored_queries']['ndcg_cut_5']:.4f} on the final follow up, "
                      f"over {final_selected['scored']} judged questions out of {final_selected['submitted']} inputs. ")
    summary = ["VALIDATION / FINAL — separate datasets, nDCG@5 means", "",
               "Validation: earlier BM25 + reranker 0.3898; hybrid fusion 0.4479", "",
               "FINAL COMPARISONS — same 332 judged / 507 inputs", ""]
    for rid,m in [*runs,*gpu_runs]:
        if m['dataset'] == 'mtragun':
            summary.append(f"{rid:<36} {m['mean_over_scored_queries']['ndcg_cut_5']:.4f}")
    summary += ["", "Primary: sparse + dense, original + rewritten query; no reranker",
                "Primary Recall@10: 0.6211; weighted nDCG@5: 0.4379",
                "Gain vs BM25 reranker: +0.0173; 95% interval [-0.0178, +0.0525]",
                "Public final set already used for CPU stage: follow-up, not fresh holdout",
                "MPS indexing: dense ~36 minutes; SPLADE ~156 minutes",
                "Historical leaderboard equivalence remains unverified"]
    demo = (
        json.loads((ROOT / "walkthrough/demo_output.json").read_text())
        if (ROOT / "walkthrough/demo_output.json").exists()
        else {}
    )
    chapters = [
        (
            "The task and the working system",
            "This is an automated narrated walkthrough of the CS 421 conversational retrieval project. Given a conversation ending in a user question, the program searches the assigned collection and returns ten original passages. It implements Subtask A, retrieval only. It does not generate a final answer. The repository includes code, reproducible experiment configurations, evaluation checks and measured results.",
            [
                "TASK: Conversation prefix + latest user question",
                "OUTPUT: Ten ranked original passage IDs",
                "DOMAINS: CLAPNQ | Cloud | FiQA | Government",
                "",
                "convrag prepare",
                "convrag index --backend bm25",
                "convrag run --config configs/E00_bm25_latest.json",
                "convrag report",
                "",
                "Personal question example:",
                "convrag search --domain cloud --question",
                '  "How can I deploy a Python application?" --rerank',
            ],
        ),
        (
            "Data separation and leakage prevention",
            "The data module pins the upstream IBM commit and preserves the original passage identifiers. It uses a field whitelist, so gold answers, reference contexts and labels do not enter the retrieval model. The development split is by whole conversation, not individual turns. Twenty two conversations are held out for validation. CPU and GPU final configurations were frozen separately. The GPU follow up reuses a public final set already evaluated in the CPU stage.",
            (ROOT / "src/convrag/data.py").read_text().splitlines()[43:67],
        ),
        (
            "Rewriting and preserving the original question",
            "The small local Qwen model rewrites the latest question using the supplied past conversation. The prompt asks it to resolve references while preserving names, dates and constraints, without answering the question. A rewrite can be wrong. The fusion experiment therefore searches both the original question and the rewrite. Each search produces a ranking, and reciprocal rank fusion combines their positions.",
            (ROOT / "src/convrag/queries.py").read_text().splitlines()[:25],
        ),
        (
            "Retrieval, fusion and reranking",
            "BM 25 provides a fast lexical baseline over the complete original collections. Reciprocal rank fusion combines query views without assuming their numeric scores are calibrated. A Mini L M cross encoder then scores the query with each candidate passage. We compare pure reranking and a fifty fifty blend of retrieval and reranker ranks, using the same candidate pool. The GPU study adds learned sparse SPLADE and B G E semantic search. Its selected hybrid combines original and rewritten query rankings without a cross encoder. This separates better retrieval from better ordering.",
            (ROOT / "src/convrag/retrieval.py").read_text().splitlines()[9:32],
        ),
        (
            "Scoring and verification",
            "The primary metric is n D C G at five. Recall at ten measures final retrieval coverage, and candidate recall at one hundred diagnoses whether a relevant passage was available to the reranker. The official metric function matches our wrapper on the baseline. The official format checker and automated tests check output coverage, duplicates, score validity, ID preservation and leakage. Both the direct average and the upstream style weighted aggregate are recorded.",
            (ROOT / "src/convrag/evaluation.py").read_text().splitlines()[7:30],
        ),
        (
            "Measured results and practical limits",
            f"The strongest GPU validation run is hybrid fusion, with n D C G at five of {best[1]['mean_over_scored_queries']['ndcg_cut_5']:.4f}. " + final_sentence + "Recall at ten is 0.6211. The improvement over plain B M 25 is clear. The smaller gain over B M 25 plus reranking has a paired uncertainty interval including zero. Both complete neural indexes and all twelve GPU validation runs finished in the user's Metal enabled Terminal. The agent process itself could not access Metal. These are follow up results on a public final set already used for the CPU study. Historical leaderboard equivalence remains unverified. Review the combined report and the actual course instructions. This synthetic voice video is an automated review aid, not a student recorded presentation.",
            summary,
        ),
    ]
    segments = []
    transcript = []
    timestamps = []
    position = 0
    for i, (heading, narration, lines) in enumerate(chapters):
        text = BUILD / f"{i:02d}.txt"
        audio = BUILD / f"{i:02d}.wav"
        png = BUILD / f"{i:02d}.png"
        mp4 = BUILD / f"{i:02d}.mp4"
        text.write_text(narration)
        narrator.create(narration, audio)
        duration = (
            float(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=nw=1:nk=1",
                        str(audio),
                    ],
                    text=True,
                )
            )
            + 0.5
        )
        image = Image.new("RGB", (1280, 720), "#f3f5f8")
        draw = ImageDraw.Draw(image)
        titlefont = ImageFont.truetype(FONT, 27)
        font = ImageFont.truetype(FONT, 16)
        small = ImageFont.truetype(FONT, 15)
        draw.rectangle((0, 0, 1280, 100), fill="#15324c")
        draw.text((32, 24), heading, font=titlefont, fill="white")
        draw.text(
            (32, 66),
            "CS421 | Actual implementation and recorded experiments",
            font=small,
            fill="#cbdbe7",
        )
        draw.rounded_rectangle((28, 120, 1252, 650), radius=12, fill="#18222e")
        visible = []
        for line in lines:
            visible.extend(
                textwrap.wrap(
                    line, 118, replace_whitespace=False, drop_whitespace=False
                )
                or [""]
            )
        for n, line in enumerate(visible[:25]):
            draw.text((44, 137 + 19 * n), line, font=font, fill="#e7eff6")
        draw.text(
            (32, 677),
            f"Automated narrated review aid | Chapter {i + 1} of {len(chapters)} | Not an official submission",
            font=small,
            fill="#435365",
        )
        image.save(png)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(png),
                "-i",
                str(audio),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-tune",
                "stillimage",
                "-r",
                "5",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-shortest",
                str(mp4),
            ],
            check=True,
        )
        duration = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(mp4),
                ],
                text=True,
            )
        )
        segments.append(mp4)
        transcript.append(f"## {heading}\n\n{narration}")
        timestamps.append(
            {
                "chapter": heading,
                "start_seconds": position,
                "duration_seconds": duration,
            }
        )
        position += duration
    listing = BUILD / "concat.txt"
    listing.write_text("".join("file '" + str(p) + "'\n" for p in segments))
    video = ROOT / "walkthrough/AUTOMATED_WALKTHROUGH.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    (ROOT / "walkthrough/NARRATION.md").write_text(
        "# Automated walkthrough transcript\n\n" + "\n\n".join(transcript) + "\n"
    )
    (ROOT / "walkthrough/chapters.json").write_text(
        json.dumps(timestamps, indent=2) + "\n"
    )
    print(video, "estimated_seconds", round(position))


if __name__ == "__main__":
    main()
