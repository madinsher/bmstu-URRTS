"""Общие функции скриптов экспериментов: разбор аргументов, запись таблиц."""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def parse(pr, extra=None):
    ap = argparse.ArgumentParser(description="Эксперименты %s: графики и таблицы для отчёта" % pr)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "reports", pr.lower()),
                    help="каталог результатов (по умолчанию reports/%s)" % pr.lower())
    ap.add_argument("--seed", type=int, default=0, help="зерно генератора; вариант бригады — своё зерно")
    ap.add_argument("--quick", action="store_true", help="уменьшенный объём для проверки, что всё запускается")
    if extra:
        extra(ap)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    return args


def write_table(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  таблица:", os.path.relpath(path))
    width = [max(len(str(h)), *(len(fmt(r[i])) for r in rows)) for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).ljust(width[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  " + "  ".join(fmt(r[i]).ljust(width[i]) for i in range(len(header))))


def fmt(x):
    if isinstance(x, float):
        return "%.4g" % x
    return str(x)


def done(path):
    print("  рисунок:", os.path.relpath(path))
