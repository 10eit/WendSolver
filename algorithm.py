from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    mass: float = 0.0
    terminal_weight: float = 0.0
    terminal_word: Optional[str] = None


class Lexicon:
    """Length-indexed weighted tries.

    dictionary.txt format:
        word
    or
        word<TAB>weight

    A missing weight defaults to 1.0, so node mass becomes the number of
    dictionary completions under a prefix.  With weights, it becomes their
    total prior mass.
    """

    def __init__(self) -> None:
        self.roots: Dict[int, TrieNode] = {}
        self.word_count = 0

    def add(self, word: str, weight: float = 1.0) -> None:
        word = word.strip().lower()
        if not word.isalpha() or weight <= 0:
            return
        root = self.roots.setdefault(len(word), TrieNode())
        node = root
        node.mass += weight
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
            node.mass += weight
        if node.terminal_weight == 0.0:
            self.word_count += 1
        node.terminal_weight += weight
        node.terminal_word = word

    @classmethod
    def from_file(cls, path: str, min_len: int = 2, max_len: Optional[int] = None) -> "Lexicon":
        lex = cls()
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                word = parts[0].lower()
                if not word.isalpha() or len(word) < min_len:
                    continue
                if max_len is not None and len(word) > max_len:
                    continue
                weight = 1.0
                if len(parts) >= 2:
                    try:
                        weight = float(parts[1])
                    except ValueError:
                        pass
                lex.add(word, weight)
        return lex


class Board:
    def __init__(self, rows: Sequence[str], mask_char: str = "#") -> None:
        if not rows:
            raise ValueError("board must contain at least one row")
        width = len(rows[0])
        if width == 0 or any(len(r) != width for r in rows):
            raise ValueError("board must be rectangular")
        self.h = len(rows)
        self.w = width
        self.rows = [r.lower() for r in rows]
        self.mask_char = mask_char
        self.cells: Set[Coord] = {
            (r, c)
            for r in range(self.h)
            for c in range(self.w)
            if self.rows[r][c] != mask_char
        }
        self._neighbors: Dict[Coord, Tuple[Coord, ...]] = {}
        for r, c in self.cells:
            ns = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if q in self.cells:
                    ns.append(q)
            self._neighbors[(r, c)] = tuple(ns)

    def char(self, v: Coord) -> str:
        return self.rows[v[0]][v[1]]

    def neighbors(self, v: Coord) -> Tuple[Coord, ...]:
        return self._neighbors[v]


@dataclass(frozen=True)
class Candidate:
    word: str
    path: Tuple[Coord, ...]
    score: float


@dataclass(frozen=True)
class _BeamState:
    path: Tuple[Coord, ...]
    node: TrieNode = field(compare=False, hash=False)
    score: float = 0.0


class HeuristicSearch:
    """Trie lexical prior + shallow spatial lookahead + beam search."""

    def __init__(
        self,
        board: Board,
        lexicon: Lexicon,
        *,
        beam_size: int = 16,
        lookahead_depth: int = 3,
        alpha: float = 1.0,
        beta: float = 0.7,
        gamma: float = 1.2,
    ) -> None:
        self.board = board
        self.lexicon = lexicon
        self.beam_size = beam_size
        self.lookahead_depth = lookahead_depth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = 1e-12

    def _reachable_count(self, start: Coord, allowed: Set[Coord]) -> int:
        if start not in allowed:
            return 0
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in self.board.neighbors(u):
                if v in allowed and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen)

    def _prefix_reachable(self, path: Tuple[Coord, ...], available: Set[Coord], target_len: int) -> bool:
        """Necessary condition: endpoint can still reach enough unused cells."""
        need_more = target_len - len(path)
        if need_more <= 0:
            return True
        endpoint = path[-1]
        used = set(path)
        allowed = (available - used) | {endpoint}
        return self._reachable_count(endpoint, allowed) >= need_more + 1

    def _lookahead_mass(
        self,
        cell: Coord,
        node: TrieNode,
        visited: Set[Coord],
        available: Set[Coord],
        depth: int,
    ) -> float:
        """Dictionary mass realizable by shallow self-avoiding walks on the board."""
        if depth <= 0:
            return node.mass
        total = 0.0
        for nxt in self.board.neighbors(cell):
            if nxt not in available or nxt in visited:
                continue
            child = node.children.get(self.board.char(nxt))
            if child is None:
                continue
            total += self._lookahead_mass(
                nxt, child, visited | {nxt}, available, depth - 1
            )
        return total

    def generate_candidates(
        self,
        length: int,
        available: Set[Coord],
        *,
        max_candidates: int = 64,
    ) -> List[Candidate]:
        root = self.lexicon.roots.get(length)
        if root is None or len(available) < length:
            return []

        states: List[_BeamState] = []
        for v in available:
            child = root.children.get(self.board.char(v))
            if child is None:
                continue
            lex_prob = child.mass / max(root.mass, self.eps)
            depth = min(self.lookahead_depth, length - 1)
            la_mass = self._lookahead_mass(v, child, {v}, available, depth)
            if la_mass <= 0 and length > 1:
                continue
            la_ratio = la_mass / max(child.mass, self.eps)
            score = self.alpha * math.log(lex_prob + self.eps)
            score += self.beta * math.log(la_ratio + self.eps)
            states.append(_BeamState((v,), child, score))

        states.sort(key=lambda s: s.score, reverse=True)
        states = states[: self.beam_size]

        for step in range(1, length):
            expanded: List[_BeamState] = []
            for state in states:
                endpoint = state.path[-1]
                visited = set(state.path)
                for nxt in self.board.neighbors(endpoint):
                    if nxt not in available or nxt in visited:
                        continue
                    child = state.node.children.get(self.board.char(nxt))
                    if child is None:
                        continue
                    new_path = state.path + (nxt,)
                    if not self._prefix_reachable(new_path, available, length):
                        continue

                    lex_prob = child.mass / max(state.node.mass, self.eps)
                    remaining = length - len(new_path)
                    depth = min(self.lookahead_depth, remaining)
                    if depth > 0:
                        la_mass = self._lookahead_mass(
                            nxt, child, visited | {nxt}, available, depth
                        )
                        if la_mass <= 0:
                            continue
                        la_ratio = la_mass / max(child.mass, self.eps)
                    else:
                        la_ratio = 1.0

                    score = state.score
                    score += self.alpha * math.log(lex_prob + self.eps)
                    score += self.beta * math.log(la_ratio + self.eps)
                    expanded.append(_BeamState(new_path, child, score))

            if not expanded:
                return []
            expanded.sort(key=lambda s: s.score, reverse=True)
            states = expanded[: self.beam_size]

        out: List[Candidate] = []
        for state in states:
            if state.node.terminal_weight <= 0 or not state.node.terminal_word:
                continue
            terminal_prob = state.node.terminal_weight / max(state.node.mass, self.eps)
            out.append(
                Candidate(
                    state.node.terminal_word,
                    state.path,
                    state.score + self.alpha * math.log(terminal_prob + self.eps),
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:max_candidates]


class CoverageSolver:
    """Outer exact-cover search with MRV and residual component pruning."""

    def __init__(
        self,
        search: HeuristicSearch,
        *,
        max_candidates_per_length: int = 48,
    ) -> None:
        self.search = search
        self.board = search.board
        self.max_candidates_per_length = max_candidates_per_length
        self._failed = set()

    def _components(self, available: Set[Coord]) -> List[int]:
        unseen = set(available)
        sizes: List[int] = []
        while unseen:
            start = next(iter(unseen))
            seen = {start}
            stack = [start]
            unseen.remove(start)
            while stack:
                u = stack.pop()
                for v in self.board.neighbors(u):
                    if v in unseen:
                        unseen.remove(v)
                        seen.add(v)
                        stack.append(v)
            sizes.append(len(seen))
        return sorted(sizes, reverse=True)

    @staticmethod
    def _lengths_fit_components(component_sizes: Sequence[int], lengths: Sequence[int]) -> bool:
        if sum(component_sizes) != sum(lengths):
            return False
        if not component_sizes:
            return not lengths
        if not lengths:
            return False

        caps = list(sorted(component_sizes, reverse=True))
        vals = list(sorted(lengths, reverse=True))
        if vals[0] > caps[0]:
            return False

        @lru_cache(maxsize=None)
        def assign(i: int, capacities: Tuple[int, ...]) -> bool:
            if i == len(vals):
                return all(c == 0 for c in capacities)
            x = vals[i]
            tried = set()
            for j, cap in enumerate(capacities):
                if cap < x or cap in tried:
                    continue
                tried.add(cap)
                new_caps = list(capacities)
                new_caps[j] -= x
                new_caps.sort(reverse=True)
                if assign(i + 1, tuple(new_caps)):
                    return True
            return False

        return assign(0, tuple(caps))

    def residual_feasible(self, available: Set[Coord], remaining_lengths: Sequence[int]) -> bool:
        if len(available) != sum(remaining_lengths):
            return False
        if not available:
            return not remaining_lengths
        components = self._components(available)
        return self._lengths_fit_components(components, remaining_lengths)

    def solve(self, lengths: Sequence[int]) -> Optional[List[Candidate]]:
        lengths = tuple(sorted((int(x) for x in lengths), reverse=True))
        if sum(lengths) != len(self.board.cells):
            raise ValueError(
                f"sum(lengths)={sum(lengths)} but board has {len(self.board.cells)} accessible cells"
            )
        self._failed.clear()
        return self._dfs(set(self.board.cells), lengths)

    def _dfs(self, available: Set[Coord], lengths: Tuple[int, ...]) -> Optional[List[Candidate]]:
        if not lengths:
            return [] if not available else None
        key = (frozenset(available), lengths)
        if key in self._failed:
            return None
        if not self.residual_feasible(available, lengths):
            self._failed.add(key)
            return None

        # MRV: generate candidates for every distinct remaining length and
        # branch on the length with the fewest surviving candidates.
        candidate_sets = []
        for length in sorted(set(lengths)):
            candidates = self.search.generate_candidates(
                length,
                available,
                max_candidates=self.max_candidates_per_length,
            )
            if not candidates:
                self._failed.add(key)
                return None
            candidate_sets.append((len(candidates), length, candidates))

        _, chosen_len, candidates = min(candidate_sets, key=lambda x: (x[0], -x[1]))
        rest = list(lengths)
        rest.remove(chosen_len)
        rest_t = tuple(sorted(rest, reverse=True))

        ranked: List[Tuple[float, Candidate, Set[Coord]]] = []
        for cand in candidates:
            new_available = available - set(cand.path)
            if not self.residual_feasible(new_available, rest_t):
                continue
            # Small bonus for preserving residual connectivity / feasibility.
            comps = self._components(new_available) if new_available else []
            geometry_bonus = -0.05 * max(0, len(comps) - len(rest_t))
            ranked.append((cand.score + self.search.gamma * geometry_bonus, cand, new_available))

        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, cand, new_available in ranked:
            suffix = self._dfs(new_available, rest_t)
            if suffix is not None:
                return [cand] + suffix

        self._failed.add(key)
        return None
