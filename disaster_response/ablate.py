"""Run the fixed recipe's complete model and five ablations, without tuning."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from .training import write_csv
from . import protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--data',type=Path);group.add_argument('--raw-root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--split',choices=['original','purged'])
    p.add_argument('--epochs',type=int)
    args=p.parse_args();spec=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    common=['--config',str(args.config),'--device',args.device]
    common+=['--data',str(args.data)] if args.data else ['--raw-root',str(args.raw_root)]
    if args.split:common+=['--split',args.split]
    if args.epochs is not None:common+=['--epochs',str(args.epochs)]
    for kind in ('full','backbone','shared','mlp'):
        training=args.output/'training'/kind
        subprocess.run([sys.executable,'-m','disaster_response.cli','train',*common,
                        '--variant',kind,'--output',str(training)],check=True)
        checkpoint=training/spec['dataset']/'trials'/kind/'model.pt'
        subprocess.run([sys.executable,'-m','disaster_response.cli','evaluate',*common,
                        '--checkpoint',str(checkpoint),'--output',str(args.output/'evaluation'/kind)],check=True)
    variants=[('full_online','完整模型','full','online'),('without_response','去掉响应分支','backbone','online'),
       ('without_regional_adaptation','去掉地区自适应','shared','online'),('without_online','去掉在线更新','full','frozen'),
       ('response_mlp','响应函数换成MLP','mlp','online'),('backbone_frozen','仅冻结骨架','backbone','frozen')]
    rows=[]
    for variant,label,kind,mode in variants:
        with (args.output/'evaluation'/kind/'results_wide_cn.csv').open(encoding='utf-8-sig') as f:
            r=next(row for row in csv.DictReader(f) if row['版本']==mode)
        r['版本']=variant;r={'版本':variant,'说明':label,**{k:v for k,v in r.items() if k!='版本'}}
        rows.append(r)
    write_csv(args.output/'ablation_results_wide_cn.csv',rows)
    protocol.write_json(args.output/'completion.json',dict(complete=True,result_count=6,seed=spec['training']['seed']))
    print(args.output/'ablation_results_wide_cn.csv')


if __name__=='__main__':main()
