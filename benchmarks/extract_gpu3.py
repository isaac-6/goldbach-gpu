import re
import sys

pattern = re.compile(
    r"LIMIT=(\d+)\s+SEG_SIZE=(\d+)\s+P_SMALL=(\d+).*?"
    r"Total time\s+:\s+([\d\.]+)\s+ms",
    re.DOTALL
)

def parse_log(path):
    with open(path, "r") as f:
        text = f.read()

    rows = []
    for match in pattern.finditer(text):
        LIMIT = int(match.group(1))
        SEG = int(match.group(2))
        P = int(match.group(3))
        t_ms = float(match.group(4))
        rows.append((LIMIT, SEG, P, t_ms))

    return rows

if __name__ == "__main__":
    rows = parse_log(sys.argv[1])
    print("LIMIT,SEG_SIZE,P_SMALL,total_ms,evens_per_sec")
    for LIMIT, SEG, P, t_ms in rows:
        evens = (LIMIT - 4) // 2 + 1
        eps = evens / (t_ms / 1000)
        print(f"{LIMIT},{SEG},{P},{t_ms:.2f},{eps:.2f}")