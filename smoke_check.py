"""Run numerical gates without overwriting the published synthetic results."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import experiments as exp


def main():
    original = exp.OUT
    try:
        with TemporaryDirectory(prefix='lcrc-gates-') as temp:
            exp.OUT = Path(temp)
            exp.gate0()
            exp.s1()
            gate1 = json.loads((exp.OUT / 'gate1.json').read_text())
            assert gate1['passed'], gate1
    finally:
        exp.OUT = original
    for stage in ('S2', 'S3'):
        rows = exp.pd.read_parquet(original / f'{stage}_instances.parquet')
        assert len(rows) == 1120, (stage, len(rows))
        assert rows['method'].nunique() == 7
        assert not json.loads((original / f'{stage}_errors.json').read_text())
    assert len(exp.pd.read_csv(original / 's1.csv')) == 120
    print('PASS: numerical Gates 0/1 and final synthetic result inventory; saved results unchanged.')


if __name__ == '__main__':
    main()
