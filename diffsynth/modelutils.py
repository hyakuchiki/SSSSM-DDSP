import hydra
import numpy as np
import torch
from diffsynth.modules.generators import SineOscillator
from diffsynth.processor import Add, Mix
from diffsynth.modules.fm import FM2, FM3
from diffsynth.modules.envelope import ADSREnvelope
from diffsynth.synthesizer import Synthesizer, FX
from diffsynth.modules.harmor import Harmor
from diffsynth.modules.delay import ModulatedDelay, ChorusFlanger
from diffsynth.modules.reverb import DecayReverb

def construct_synth_from_conf(synth_conf, ext_f0=False):
    dag = []
    for module_name, v in synth_conf.dag.items():
        module = hydra.utils.instantiate(v.config, name=module_name)
        conn = v.connections
        dag.append((module, conn))
    fixed_p = synth_conf.fixed_params
    fixed_p = {} if fixed_p is None else fixed_p
    fixed_p = {k: None if v is None else v*torch.ones(1) for k, v in fixed_p.items()}
    if ext_f0:
        fixed_p['BFRQ'] = None
    synth = Synthesizer(dag, fixed_params=fixed_p, static_params=synth_conf.static_params)
    return synth