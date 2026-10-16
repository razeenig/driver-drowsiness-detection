import os
import re
from collections import defaultdict

DATASET_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

groups = {
    "Drowsy": defaultdict(list),
    "Non Drowsy": defaultdict(list)
}

for class_name in groups:

    folder = os.path.join(DATASET_DIR, class_name)

    for filename in os.listdir(folder):

        if not filename.lower().endswith(".png"):
            continue

        name = os.path.splitext(filename)[0]

        match = re.match(r"([A-Za-z]+)(\d+)", name)

        if match:
            prefix = match.group(1).lower()
            number = int(match.group(2))
            groups[class_name][prefix].append(number)


print("================================")
print("GROUP COMPARISON")
print("================================")

drowsy_groups = set(groups["Drowsy"].keys())
non_drowsy_groups = set(groups["Non Drowsy"].keys())

print("\nDrowsy groups:")
print(sorted(drowsy_groups))

print("\nNon-Drowsy groups:")
print(sorted(non_drowsy_groups))

print("\nGroups appearing in BOTH classes:")
common = sorted(drowsy_groups & non_drowsy_groups)
print(common)

print("\nNumber of common groups:", len(common))


print("\n================================")
print("NUMBER OVERLAP ANALYSIS")
print("================================")

for group in common:

    d_numbers = set(groups["Drowsy"][group])
    n_numbers = set(groups["Non Drowsy"][group])

    overlap = d_numbers & n_numbers

    print(
        f"{group}: "
        f"Drowsy={len(d_numbers)}, "
        f"Non-Drowsy={len(n_numbers)}, "
        f"Same numbers={len(overlap)}"
    )

    if overlap:
        print("   Example overlapping numbers:", sorted(overlap)[:20])