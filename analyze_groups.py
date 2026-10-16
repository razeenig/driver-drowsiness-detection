import os
import re
from collections import defaultdict

DATASET_DIR = r"C:\Users\razee\OneDrive\Desktop\DDD\Driver Drowsiness Dataset (DDD)"

for class_name in ["Drowsy", "Non Drowsy"]:

    folder = os.path.join(DATASET_DIR, class_name)

    groups = defaultdict(list)

    for filename in os.listdir(folder):

        if not filename.lower().endswith(".png"):
            continue

        name = os.path.splitext(filename)[0]

        # Separate letters from numbers
        match = re.match(r"([A-Za-z]+)(\d+)", name)

        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            groups[prefix].append(number)

    print("\n================================")
    print(class_name)
    print("================================")

    for prefix in sorted(groups):

        numbers = sorted(groups[prefix])

        print(
            f"{prefix}: "
            f"{len(numbers)} images | "
            f"range {numbers[0]}-{numbers[-1]}"
        )

        # Show first 10 numbers
        print("   First:", numbers[:10])

        # Show gaps
        gaps = []

        for i in range(1, len(numbers)):
            if numbers[i] != numbers[i - 1] + 1:
                gaps.append((numbers[i - 1], numbers[i]))

        if gaps:
            print("   Gaps:", gaps[:5])
        else:
            print("   Sequential: YES")