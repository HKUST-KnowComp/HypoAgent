import os
import json
import argparse
from collections import Counter

from akgr.utils.load_util import load_yaml, load_and_filter_query_patterns
from akgr.kgdata import load_kg


def inspect_jsonl_counts(path):
    total_lines = 0
    pattern_counter = Counter()
    bad_json = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                print(f"[BAD JSON] {path} line {line_no}")
                continue

            pattern = obj.get("pattern_str", None)
            if pattern is not None:
                pattern_counter[pattern] += 1

    return total_lines, pattern_counter, bad_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-sampling", default="akgr/configs/config-sampling.yml")
    parser.add_argument("-s", "--scale", required=True)
    parser.add_argument("-a", "--max-answer-size", type=int, default=32)
    parser.add_argument("--data_root", default="./sampled_data/")
    parser.add_argument("-r", "--reverse_edges_flag", action="store_true", default=False)
    args = parser.parse_args()

    config_sampling = load_yaml(args.config_sampling)

    pattern_filtered = load_and_filter_query_patterns(
        file_name=config_sampling["pattern_table_file"],
        max_dep=2,
        exclu=None,
        column="original"
    )
    pattern_list = pattern_filtered["pattern_str"].tolist()
    num_patterns = len(pattern_list)

    scaling_factor = config_sampling[args.scale]["scale"]
    datasets = config_sampling[args.scale]["datasets"]

    print("=" * 90)
    print("Check whether each split file is fully written")
    print("=" * 90)
    print(f"scale={args.scale}")
    print(f"max_answer_size={args.max_answer_size}")
    print(f"reverse_edges_flag={args.reverse_edges_flag}")
    print(f"num_patterns={num_patterns}")
    print()

    all_ok = True

    for dataname in datasets:
        print("-" * 90)
        print(f"Dataset: {dataname}")

        kg = load_kg(args.data_root, dataname, reverse_edges_flag=args.reverse_edges_flag)
        num_train_edges = kg.num_train_edges

        num_samples_perpattern = {
            "train": num_train_edges // scaling_factor,
            "valid": (num_train_edges // scaling_factor) // 8,
            "test":  (num_train_edges // scaling_factor) // 8,
        }

        expected_total = {
            split: num_patterns * num_samples_perpattern[split]
            for split in ["train", "valid", "test"]
        }

        print(f"num_train_edges={num_train_edges}")
        print(f"num_samples_perpattern={num_samples_perpattern}")
        print(f"expected_total={expected_total}")
        print()

        for split in ["train", "valid", "test"]:
            output_prefix = f"{dataname}-{args.scale}-{args.max_answer_size}-{split}"
            path = os.path.join(
                args.data_root,
                dataname,
                str(args.reverse_edges_flag),
                f"{output_prefix}-a2q.jsonl"
            )

            print(f"[{split}]")
            print(f"file={path}")

            if not os.path.exists(path):
                print("status=MISS_FILE")
                all_ok = False
                print()
                continue

            actual_total, pattern_counter, bad_json = inspect_jsonl_counts(path)
            expected = expected_total[split]
            expected_per_pattern = num_samples_perpattern[split]

            print(f"expected_lines={expected}")
            print(f"actual_lines={actual_total}")
            print(f"bad_json={bad_json}")

            split_ok = True

            if bad_json > 0:
                split_ok = False
                all_ok = False

            if actual_total != expected:
                print("line_check=FAIL")
                split_ok = False
                all_ok = False
            else:
                print("line_check=OK")

            bad_patterns = []
            for pattern in pattern_list:
                cnt = pattern_counter.get(pattern, 0)
                if cnt != expected_per_pattern:
                    bad_patterns.append((pattern, cnt, expected_per_pattern))

            if bad_patterns:
                print(f"pattern_check=FAIL ({len(bad_patterns)} mismatched)")
                for pattern, cnt, exp in bad_patterns[:10]:
                    print(f"  pattern={pattern} actual={cnt} expected={exp}")
                if len(bad_patterns) > 10:
                    print(f"  ... and {len(bad_patterns) - 10} more")
                split_ok = False
                all_ok = False
            else:
                print("pattern_check=OK")

            if split_ok:
                print("split_status=COMPLETE")
            else:
                print("split_status=INCOMPLETE")

            print()

    print("=" * 90)
    if all_ok:
        print("FINAL RESULT: all split files look complete.")
    else:
        print("FINAL RESULT: some split files are incomplete or abnormal.")
    print("=" * 90)


if __name__ == "__main__":
    main()