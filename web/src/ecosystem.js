/** Source-reviewed code inventory, NOT a live connection/permission registry. */
export const ECOSYSTEM_APPS = Object.freeze([
  {
    id:'phistream',name:'PhiStream',role:'Experimental safetensors inference engine',
    status:'RESEARCH ONLY',stage:'research',
    detail:'Already linked to PhiServe through checkpoint, tokenizer and parity research. No pretrained weights bundled; no production chat backend.',
    direction:'Future optional backend → PhiServe',next:'Independent reference and real checkpoint validation, then a separately gated text-serving backend.',
    repo:'https://github.com/MichaelWave369/PhiStream',
  },
  {
    id:'pocketbrain',name:'PocketBrain',role:'Browser AI, local memory and Tiny experiments',
    status:'ADAPTER IN CODE',stage:'adapter',
    detail:'Has a dedicated same-device PhiServe provider. PocketBrain’s WebLLM and ONNX Tiny models are different runtimes, not GGUF files.',
    direction:'PocketBrain → PhiServe',next:'Same-machine browser/local daemon acceptance with explicit token and Local Network Access permission.',
    repo:'https://github.com/MichaelWave369/PocketBrain',
  },
  {
    id:'brainc',name:'BrainC',role:'Qwen persona, tools and memory',
    status:'MODEL RECIPE',stage:'planned',
    detail:'The Modelfile references qwen2.5:14b but does not bundle weights. A separately obtained compatible GGUF can use PhiServe.',
    direction:'BrainC → PhiServe (proposed)',next:'Explicit authenticated adapter, registered alias and continuity tests for its persona and memory boundaries.',
    repo:'https://github.com/MichaelWave369/BrainC',
  },
  {
    id:'vessie',name:'Vessie',role:'Governed agent orchestration and routing',
    status:'PROPOSED',stage:'planned',
    detail:'SuperPhiVessel has cognitive roles and authorization gates, not a standalone LLM-weight collection. Inference must never grant action authority.',
    direction:'Vessie → PhiServe (proposed)',next:'Scoped inference client preserving Reality Gate permissions and existing agent governance.',
    repo:'https://github.com/MichaelWave369/SuperPhiVessel',
  },
  {
    id:'gaia',name:'GAIA',role:'Symbolic logic, readings and local-brain experiments',
    status:'PROPOSED',stage:'planned',
    detail:'GAIA lives in GypsyAI, with separate symbolic and experimental browser logic. No authenticated production PhiServe text client yet.',
    direction:'GAIA → PhiServe (proposed)',next:'Opt-in text inference without automatic sharing of personal reading or ancestry context.',
    repo:'https://github.com/MichaelWave369/GypsyAI',
  },
  {
    id:'thinktank',name:'Think Tank',role:'Multi-agent deliberation and Reality Gate',
    status:'ADAPTER IN CODE',stage:'adapter',
    detail:'Merged PR #55 adds bearer-authenticated PhiServe mode to Think Tank’s local Node bridge. Council governance remains authoritative; hardware connection still needs field testing.',
    direction:'Think Tank → PhiServe',next:'Validate the merged adapter with an actual locally running PhiServe daemon, registered GGUF and busy-model handling.',
    repo:'https://github.com/MichaelWave369/think-tank',
    pr:'https://github.com/MichaelWave369/think-tank/pull/55',
  },
].map(item=>Object.freeze({
  ...item,liveConnectionObserved:false,grantsPermissions:false,
})));
export const INVENTORY_SCOPE='STATIC_CODE_REVIEW_NOT_LIVE_DISCOVERY';
export function inventoryCounts(entries=ECOSYSTEM_APPS){
  return {listed:entries.length,
    wired:entries.filter(x=>x.stage==='adapter').length,
    inReview:entries.filter(x=>x.stage==='review').length,
    research:entries.filter(x=>x.stage==='research').length,
    planned:entries.filter(x=>x.stage==='planned').length};
}
