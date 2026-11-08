#!/usr/bin/env python3
"""Compare revised NTAR LaTeX Tables 7, 8, 10 with machine-generated JSON.

Example (run in the original simulation repository):
 python verify_manuscript_numbers.py --tex path/to/ntar_frontiers.tex --results results

Only checks numeric values that are displayed in the three revised tables.
No simulation or manuscript files are changed.
"""
from __future__ import annotations
import argparse, json, re, math, sys
from pathlib import Path

PAIR = re.compile(r'\$\s*([+-]?\d+(?:\.\d+)?)\\pm\s*([+-]?\d+(?:\.\d+)?)\s*\$')
TABLE_BEGIN = re.compile(r'\\begin\{table\}\[h!\]')
ERRORS=[]
MATCHES=0

def check(display, source, label):
    global MATCHES
    rendered, value = float(display), float(source)
    places = len(display.partition('.')[2]) if '.' in display else 0
    tolerance = 0.5 * 10**-places + 1e-10
    if not (math.isfinite(value) and abs(rendered-value) <= tolerance):
        ERRORS.append(f'{label}: manuscript {display} vs JSON {value:.10f}; tolerance {tolerance:g}')
    else:
        MATCHES+=1

def get_table(tex, label):
    needle = r'\label{' + label + '}'
    pos = tex.find(needle)
    if pos < 0: raise RuntimeError(f'Missing {needle}')
    start = tex.rfind(r'\begin{table}', 0, pos)
    end = tex.find(r'\end{table}', pos)
    if start < 0 or end < 0: raise RuntimeError(f'Malformed table {label}')
    return tex[start:end]

def rows_in_table(table):
    for line in table.splitlines():
        if '&' not in line or not line.strip().endswith(r'\\'):
            continue
        yield [c.strip() for c in line.split('&')]

def pairs(text, n, row):
    found=PAIR.findall(text)
    if len(found)!=n:
        ERRORS.append(f'{row}: expected {n} mean±CI pairs, found {len(found)}')
    return found

def audit7(tex, result):
    table=get_table(tex,'tab:decoupled')
    model=None; seen=set()
    metric_order=('energy','latency','f1','m1','pa1')
    for cols in rows_in_table(table):
        if len(cols)!=7: continue
        rowname=cols[1].replace('\\','').strip()
        if rowname not in ('Random','Fixed-Weight','NTAR','NTAR-R'): continue
        modeltxt=cols[0].strip()
        if modeltxt=='Independent': model='indep'
        if modeltxt=='Anti-correlated': model='anti'
        if model is None: ERRORS.append(f'Table 7: missing cost model for {rowname}'); continue
        key=(model,rowname)
        if key in seen: ERRORS.append(f'Table 7 duplicate {key}')
        seen.add(key)
        items=result['models'][model]['method_summary'][rowname]
        for metric, cell in zip(metric_order,cols[2:]):
            found=pairs(cell,1,f'Table 7 {model} {rowname} {metric}')
            if len(found)!=1: continue
            check(found[0][0],items[metric]['mean'],f'Table7 {key} {metric} mean')
            check(found[0][1],items[metric]['ci95_half_width'],f'Table7 {key} {metric} CI')
    if len(seen)!=8: ERRORS.append(f'Table7: expected 8 method rows; found {len(seen)}')

def audit8(tex,result):
    table=get_table(tex,'tab:scale')
    mapping=('all_candidates','yen16_candidates','ms_dfs_all','ms_yen_all','ms_yen16')
    indexed={(int(x['nodes']),float(x['degree'])):x for x in result['by_configuration']}
    seen=set()
    for cols in rows_in_table(table):
        if len(cols)!=7: continue
        try: n=int(cols[0]); degree=float(cols[1]); row=indexed[(n,degree)]
        except (ValueError,KeyError): continue
        if (n,degree) in seen: ERRORS.append(f'Table8 duplicate {(n,degree)}')
        seen.add((n,degree))
        for metric,cell in zip(mapping,cols[2:]):
            found=pairs(cell,1,f'Table8 {n},{degree} {metric}')
            if len(found)!=1: continue
            check(found[0][0],row[metric]['mean'],f'Table8 {(n,degree)} {metric} mean')
            check(found[0][1],row[metric]['ci95_half_width'],f'Table8 {(n,degree)} {metric} CI')
        if row.get('candidate_set_equal_dfs_vs_yen_all') is not True:
            ERRORS.append(f'Table8 candidate sets not equal for {(n,degree)}')
    if len(seen)!=8: ERRORS.append(f'Table8: expected 8 configurations; found {len(seen)}')

def audit10(tex,result):
    table=get_table(tex,'tab:energymatched')
    mapping=('energy','m1','pa1')
    labels=('Fixed-Weight','Random','NTAR-R','Energy-Matched Random')
    seen=set()
    for cols in rows_in_table(table):
        if len(cols)!=4 or cols[0] not in labels: continue
        label=cols[0]
        if label in seen: ERRORS.append(f'Table10 duplicate {label}')
        seen.add(label)
        for metric,cell in zip(mapping,cols[1:]):
            found=pairs(cell,1,f'Table10 {label} {metric}')
            if len(found)!=1: continue
            data=result['heldout_method_summary'][label][metric]
            check(found[0][0],data['mean'],f'Table10 {label} {metric} mean')
            check(found[0][1],data['ci95_half_width'],f'Table10 {label} {metric} CI')
    if len(seen)!=4: ERRORS.append(f'Table10: expected 4 methods; found {len(seen)}')
    calibration=result['calibration']
    if abs(calibration['beta']-7.771697)>0.0000005:
        ERRORS.append(f'Calibrated beta mismatch: {calibration["beta"]} != 7.771697')
    if result['design']['training_topologies']!=30 or result['design']['heldout_topologies']!=30:
        ERRORS.append('Baseline has unexpected train/test sample counts')
    if result['design'].get('test_recalibration') is not False:
        ERRORS.append('Baseline reports test-set recalibration')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex',type=Path,required=True)
    parser.add_argument('--results',type=Path,default=Path('results'))
    args=parser.parse_args()
    tex=args.tex.read_text(encoding='utf-8')
    filenames=['table7_revision.json','table8_revision.json','energy_matched_baseline.json']
    missing=[f for f in filenames if not (args.results/f).is_file()]
    if missing:
        sys.exit('MISSING JSON files under '+str(args.results)+': '+', '.join(missing))
    a,b,c=[json.loads((args.results/f).read_text(encoding='utf-8')) for f in filenames]
    audit7(tex,a); audit8(tex,b); audit10(tex,c)
    print(f'Checked {MATCHES} displayed means/CI half-widths against JSON results.')
    if ERRORS:
        print(f'FAILED: {len(ERRORS)} discrepancies:')
        for issue in ERRORS: print(' -',issue)
        sys.exit(1)
    print('PASS: All revised-table values agree with source JSON to the stated manuscript rounding.')

if __name__=='__main__': main()
