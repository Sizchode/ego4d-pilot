"""Release fixed main-evaluation RGB inputs for baselines, never for an agent."""
import argparse
import hashlib
import json
from pathlib import Path

from run_da3_population import identity, validate_plan, write
from run_da3_rgb import verify_input


METHODS = {'da3': 'DA3-GIANT', 'vggt': 'VGGT-1B-point',
           'pi3': 'Pi3-original-point', 'mapanything': 'MapAnything-point-unmasked',
           'amb3r': 'AMB3R-backend-depth+camera', 'ggpt': 'DA3-GIANT+GGPT'}


def main():
    parser=argparse.ArgumentParser()
    for name in ['inventory','source-manifests','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    inventory=json.loads(args.inventory.read_text())
    assert inventory['scope']=='prepared_formal_RGB_inventory_not_execution_release'
    source_index=json.loads((args.source_manifests/'index.json').read_text())['cases']
    assert [r['sequence_key'] for r in source_index]==[r['sequence_key'] for r in inventory['cases']]
    assert all(Path(r['manifest']).name==r['manifest'] for r in source_index)
    sources=[args.source_manifests/r['manifest'] for r in source_index]
    assert len(sources)==len(inventory['cases'])
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for index,(case,source) in enumerate(zip(inventory['cases'],sources)):
        assert source.name==f'{index:06d}.json'
        rgb=Path(case['rgb_directory']).resolve()
        rows=json.loads(source.read_text())
        assert len(rows)==case['views'] and [r['index'] for r in rows]==list(range(case['views']))
        public=[]
        for row,relative in zip(rows,case['images']):
            path=Path(row['path'])
            assert not path.is_symlink() and path.resolve()==(rgb/relative).resolve()
            assert path.resolve().is_relative_to(rgb)
            public.append({**row,'path':relative})
        manifest=rgb/'input_manifest.json'
        if manifest.exists():
            assert json.loads(manifest.read_text())==public
        else:
            write(manifest,public)
        verified,_=verify_input(rgb)
        assert verified==public
        cases.append({'case_id':f'{index:06d}','sequence_key':case['sequence_key'],
                      'dataset':case['dataset'],'input':str(rgb),'views':case['views'],
                      'input_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      'exposure':case['exposure'],'source_scene_id':case['source_scene_id'],
                      'source_scene_mapping':case['source_scene_mapping']})
        print(f'Verified main RGB sequence {index+1}/{len(sources)}',flush=True)
    for name,method in METHODS.items():
        plan={'schema':'rgb-population-plan-v1','method':method,'stage':'baseline_evaluation','cases':cases}
        release={'baseline_evaluation_released':True,'scope':'baseline_only',
                 'agent_execution_released':False,'plan_sha256':identity(plan),
                 'authorization':'User requested main experiment baseline evaluation before agent review',
                 'evaluation_protocol':'Fixed public canvas, least-squares Sim3, AUC5/AUC10; no test tuning'}
        validate_plan(plan,release,expected_method=method)
        write(args.output/(name+'-plan.json'),plan)
        write(args.output/(name+'-release.json'),release)
    write(args.output/'preparation.json',{'completed':True,'sequences':len(cases),
          'dataset_sequence_counts':inventory['dataset_sequence_counts'],
          'baseline_methods':METHODS,'agent_execution_released':False,'GT_access':False,
          'source_scene_mapping_resolved':False})


if __name__=='__main__':
    main()
