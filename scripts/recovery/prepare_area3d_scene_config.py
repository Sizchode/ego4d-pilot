"""Use AREA3D replay's sensor configuration for its active acquisition entrypoint.

Only combines the released agent, environment and scene YAMLs. It does not
change candidate sampling, perception models or view selection policy.
"""
import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


def prepare(source, scene, output):
    source, scene, output = map(lambda p: Path(p).resolve(), (source, scene, output))
    sys.path.insert(0, str(source))
    from utils.path_utils import resolve_cfg_paths
    inputs = [source/'configs/agent/default.yaml',
              source/'configs/env/scene_level_habitat.yaml', scene]
    config = OmegaConf.merge(*(OmegaConf.load(p) for p in inputs))
    config = resolve_cfg_paths(config, cfg_file=scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output)
    return config, inputs


def verify_entrypoint(source, scene, merged, audit_dir):
    """Capture real run()'s environment argument without constructing a simulator."""
    source, scene, merged = map(lambda p: Path(p).resolve(), (source, scene, merged))
    sys.path.insert(0, str(source))
    import active_recon
    from envs.settings import default_sim_settings
    keys = ['width', 'height', 'sensor_height', 'color_sensor',
            'depth_sensor', 'semantic_sensor', 'hfov', 'default_agent']
    expected = dict(default_sim_settings)
    for p in [source/'configs/agent/default.yaml', source/'configs/env/scene_level_habitat.yaml', scene]:
        expected.update(OmegaConf.to_container(OmegaConf.load(p), resolve=True))
    captured = {}
    class Captured(Exception):
        pass
    def capture(config):
        captured.update(default_sim_settings)
        captured.update(OmegaConf.to_container(config, resolve=True))
        raise Captured()
    original = active_recon.HabitatAREA3DEnv
    active_recon.HabitatAREA3DEnv = capture
    try:
        cfg = OmegaConf.create({'scene_cfg':str(merged), 'out_dir':str(audit_dir),
                                'env_level':'scene-level','env_backend':'habitat'})
        try:
            active_recon.run(cfg, None, None)
        except Captured:
            pass
    finally:
        active_recon.HabitatAREA3DEnv = original
    differences = {k: {'active':captured.get(k),'replay':expected.get(k)}
                   for k in keys if captured.get(k) != expected.get(k)}
    if not captured or differences:
        raise ValueError(f'Active/replay sensor mismatch: {differences}')
    return {k:captured[k] for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    config, inputs = prepare(args.source, args.scene, args.output)
    verified = verify_entrypoint(args.source, args.scene, args.output, args.output.parent)
    receipt = {'inputs':list(map(str, inputs)), 'merged_scene':str(args.output),
               'camera_config':verified, 'validation':'Actual active run() environment arguments match replay configuration; no simulator/model executed'}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
