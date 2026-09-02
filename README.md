# WendSolver

WendSolver is an independent research and educational implementation of a generic dictionary-constrained grid path-cover solver. It is not affiliated with or endorsed by LinkedIn.

## Usage

```bash
python solver.py \
  --board 'XXXXXX;XXXXXX;XX##X##;XXXXXX;XXX##XX;XXXX#XXX' \
  --lengths 'X,X,X,X' \
  --beam 64 \
  --max-candidates 256
```

### Board format

Write down the board row by row and separate rows with `;`.

WendSolver supports arbitrary \(M \times N\) rectangular boards. Use `#` to denote masked or unavailable cells.

For example,

```text
ABCDEF
GHIJKL
MN##O#
PQRSTU
```

can be written as:

```bash
--board 'ABCDEF;GHIJKL;MN##O#;PQRSTU'
```

### Arguments

* `--board`: Board configuration. Rows are separated by `;`, and `#` denotes a masked cell.
* `--lengths`: Comma-separated target word lengths. Their sum must equal the number of unmasked cells.
* `--beam`: Beam size used during candidate-path search. Larger values explore more candidate states but require more computation.
* `--max-candidates`: Maximum number of candidate word paths retained for each target length.

## Dictionary

The bundled `dictionary.txt` is derived from the CMU Pronouncing Dictionary (CMUdict). The solver uses the dictionary only as a lexical constraint during path search; no language model or external AI service is required.
