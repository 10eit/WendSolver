from __future__ import annotations

import argparse
import os
from typing import List

from algorithm import Board, CoverageSolver, HeuristicSearch, Lexicon


def parse_board(text: str) -> List[str]:
    rows = [r.strip() for r in text.replace("/", ";").split(";") if r.strip()]
    if not rows:
        raise argparse.ArgumentTypeError("empty board")
    return rows


def parse_lengths(text: str) -> List[int]:
    try:
        vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from e
    if not vals or any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return vals


def render_solution(rows: List[str], solution) -> str:
    h, w = len(rows), len(rows[0])
    labels = [["##" if rows[r][c] == "#" else ".." for c in range(w)] for r in range(h)]
    for i, cand in enumerate(solution, 1):
        for r, c in cand.path:
            labels[r][c] = f"{i:02d}"
    return "\n".join(" ".join(row) for row in labels)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Heuristic solver for LinkedIn Wend-like word-path cover puzzles."
    )
    parser.add_argument(
        "--board",
        default="CAT;GOD;SUN",
        help="rows separated by ';' or '/', use # for masked cells (default: CAT;GOD;SUN)",
    )
    parser.add_argument(
        "--lengths",
        default="3,3,3",
        help="comma-separated target lengths (default: 3,3,3)",
    )
    parser.add_argument(
        "--dictionary",
        default=os.path.join(here, "dictionary.txt"),
        help="dictionary file: one word per line or 'word weight'",
    )
    parser.add_argument("--beam", type=int, default=32, help="beam width (default: 32)")
    parser.add_argument("--lookahead", type=int, default=3, help="lookahead depth (default: 3)")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=96,
        help="max complete candidates kept per length (default: 96)",
    )
    args = parser.parse_args()

    rows = parse_board(args.board)
    lengths = parse_lengths(args.lengths)
    board = Board(rows)
    lexicon = Lexicon.from_file(args.dictionary, min_len=min(lengths), max_len=max(lengths))

    search = HeuristicSearch(
        board,
        lexicon,
        beam_size=max(1, args.beam),
        lookahead_depth=max(0, args.lookahead),
    )
    solver = CoverageSolver(search, max_candidates_per_length=max(1, args.max_candidates))
    solution = solver.solve(lengths)

    print(f"dictionary words loaded: {lexicon.word_count}")
    if solution is None:
        print("No solution found within the current heuristic beam/candidate limits.")
        print("Try increasing --beam and --max-candidates.")
        return

    print("Solution:")
    for i, cand in enumerate(solution, 1):
        coords = " -> ".join(f"({r},{c})" for r, c in cand.path)
        print(f"  {i}. {cand.word.upper():<12} score={cand.score: .4f}  {coords}")
    print("\nCoverage map (number = chosen word):")
    print(render_solution(rows, solution))


if __name__ == "__main__":
    main()
