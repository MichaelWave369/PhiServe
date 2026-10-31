import test from 'node:test';
import assert from 'node:assert/strict';
import {ECOSYSTEM_APPS,INVENTORY_SCOPE,inventoryCounts} from '../src/ecosystem.js';

test('inventory covers all six ecosystem projects with unique IDs and source links',()=>{
  assert.deepEqual(ECOSYSTEM_APPS.map(x=>x.id),
    ['phistream','pocketbrain','brainc','vessie','gaia','thinktank']);
  assert.equal(new Set(ECOSYSTEM_APPS.map(x=>x.id)).size,6);
  for(const entry of ECOSYSTEM_APPS){
    assert.ok(entry.repo.startsWith('https://github.com/MichaelWave369/'));
    assert.ok(entry.detail.length>30 && entry.next.length>20);
    assert.equal(entry.liveConnectionObserved,false);
    assert.equal(entry.grantsPermissions,false);
  }
});

test('research is never silently promoted to production backend',()=>{
  assert.equal(INVENTORY_SCOPE,'STATIC_CODE_REVIEW_NOT_LIVE_DISCOVERY');
  assert.equal(ECOSYSTEM_APPS.find(x=>x.id==='phistream').stage,'research');
  assert.equal(ECOSYSTEM_APPS.find(x=>x.id==='pocketbrain').stage,'adapter');
  assert.equal(ECOSYSTEM_APPS.find(x=>x.id==='thinktank').stage,'adapter');
  assert.deepEqual(inventoryCounts(),{listed:6,wired:2,inReview:0,research:1,planned:3});
});

test('BrainC recipe and PocketBrain browser weights are not treated as native GGUF assets',()=>{
  assert.match(ECOSYSTEM_APPS.find(x=>x.id==='brainc').detail,/does not bundle weights/);
  assert.match(ECOSYSTEM_APPS.find(x=>x.id==='pocketbrain').detail,/different runtimes/);
});
