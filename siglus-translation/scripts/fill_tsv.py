# 翻译填充工具：将 JSON(索引->译文) 写入 TSV 第 4 列，前 3 列不动
# 用法: python fill_tsv.py <tsv路径> <json1> [json2 ...]
import json, sys, os

def main():
    tsv = sys.argv[1]
    jsons = sys.argv[2:]
    tr = {}
    for j in jsons:
        with open(j, encoding='utf-8') as f:
            tr.update(json.load(f))
    rows = []
    with open(tsv, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if not line.strip():
                rows.append(line)
                continue
            cols = line.split('\t')
            try:
                idx = int(cols[0])
            except ValueError:
                rows.append(line)
                continue
            key = str(idx)
            if key in tr:
                while len(cols) < 4:
                    cols.append('')
                cols[3] = tr[key]
            rows.append('\t'.join(cols))
    with open(tsv, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'{os.path.basename(tsv)}: 填充 {len(tr)} 条')

if __name__ == '__main__':
    main()
